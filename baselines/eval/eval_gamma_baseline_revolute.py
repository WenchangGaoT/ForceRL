from env.baseline_revolute_eval_env import BaselineEvalRevoluteEnv
import open3d as o3d
import robosuite as suite
import numpy as np
import matplotlib.pyplot as plt
import os 
import json
from scipy.spatial.transform import Rotation as R 
import utils.sim_utils as sim_utils
from grasps.aograsp.get_pointclouds import get_aograsp_ply_and_config
from grasps.aograsp.get_affordance import get_affordance_main
from grasps.aograsp.get_proposals import get_grasp_proposals_main
from gamma.get_joint_param import get_joint_param_main
from utils.baseline_utils import drive_to_grasp
import time

import termcolor

def eval_gamma_revolute(experiment_name, run_id, object_type="microwave", success_threshold=0.7):

    controller_name = "OSC_POSE"

    controller_cfg_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"controller_configs")
    controller_cfg_path = os.path.join(controller_cfg_dir, "osc_pose_absolute.json")
    controller_configs = suite.load_controller_config(custom_fpath = controller_cfg_path,default_controller=controller_name)

    with open(controller_cfg_path, 'r') as f:
        controller_configs = json.load(f)
    # print(controller_configs)


    env_kwargs = dict(
        # robots="Panda",
        robots="UR5e",
        object_name = "dishwasher-1",
        # obj_rotation=(-np.pi/2, -np.pi/2),
        # obj_rotation=(0, 0),
        obj_rotation=(-np.pi / 2, 0),
        scale_object = True,
        object_scale = 0.3,
        has_renderer=True,
        use_camera_obs=False,
        has_offscreen_renderer=True,
        camera_depths = True,
        camera_segmentations = "element",
        controller_configs=controller_configs,
        control_freq = 20,
        horizon=1000,
        camera_names = ["birdview", "agentview", "frontview", "sideview"],
        # render_camera = "birdview",
        camera_heights = [256,256,256,256],
        camera_widths = [256,256,256,256],
        move_robot_away = False,

        rotate_around_robot = True,
        object_robot_distance = (0.7,0.7),
        open_percentage = 0.3,

        cache_video = False,
        get_grasp_proposals_flag = True,
        skip_object_initialization=False
    )
    env_name = "BaselineEvalRevoluteEnv"


    trials_per_object = 20
    available_objects = BaselineEvalRevoluteEnv.available_objects()[object_type]
    
    baselines_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eval_results_dir = os.path.join(baselines_dir, f"eval_results/gamma_revolute_{experiment_name}_{run_id}")
    os.makedirs(eval_results_dir, exist_ok=True)

    object_success_dict = dict(
        object_type = object_type,
        obj_rotation = env_kwargs["obj_rotation"],
        object_robot_distance = env_kwargs["object_robot_distance"],
        open_percentage = env_kwargs["open_percentage"],
        trials_per_object = trials_per_object,
    )
    loosened_success_threshold = success_threshold

    for obj_name in available_objects: 

        if object_type in obj_name:
            env_kwargs["object_name"] = obj_name
        else:
            raise RuntimeError("Object not found in available objects")

        env:BaselineEvalRevoluteEnv = suite.make(
            env_name, 
            **env_kwargs
        )
        env._set_door_friction(3.0)

        success_list = []

        for exp_num in range(trials_per_object):
            print(f"Running experiment {exp_num} for object {obj_name}")

            # get the joint parameters
            joint_poses, joint_directions, joint_types, resuts_list = get_joint_param_main(
                env.pcd_wf_no_downsample_path, 
                env.camera_info_path, 
                viz=False, 
                return_results_list=True)

            # find the first joint in the list with joint_type = 0
            joint_pose_selected = None
            joint_direction_selected = None

            # randomly select a joint
            joint_idx = np.random.randint(0, len(joint_types))
            joint_pose_selected = joint_poses[joint_idx]
            joint_direction_selected = joint_directions[joint_idx]
            joint_direction_selected = joint_direction_selected / np.linalg.norm(joint_direction_selected)
            
            # print(termcolor.colored("joint pose selected: ", "green"), joint_pose_selected)
            # print(termcolor.colored("joint direction selected: ", "green"), joint_direction_selected)
            # joint_direction_selected = -joint_direction_selected
            env._set_door_friction(3.0)

            drive_to_grasp(
                env=env,
                reload_controller_config=True,
                use_gym_wrapper=False,
                render=False
            )

            obs = env._get_observations()
            last_grasp_pos = obs['robot0_eef_pos']
            rotation_vector = obs["grasp_rot"]


            h_point, f_point, h_direction = joint_pose_selected, obs['robot0_eef_pos'], joint_direction_selected
            obs = np.concatenate([
                                    h_direction, 
                                    (f_point-h_point-np.dot(f_point-h_point, h_direction)*h_direction)
                                                ])
            done = False
            # last_grasp_pos = final_grasp_pos
            for i in range(990):
                # action is vertical to both the hinge direction and the direction from the hinge to the finger
                # use cross product to get the vertical direction
                action = -np.cross(obs[:3], obs[3:])
                action = action / np.linalg.norm(action)
                
                
                # print("action: ", action)
                action = action * 0.01
                action = np.concatenate([last_grasp_pos +action, rotation_vector, [1]])
                next_obs, reward, done, _ = env.step(action)

                h_point, f_point, h_direction = joint_pose_selected, next_obs['robot0_eef_pos'], joint_direction_selected
                last_grasp_pos = next_obs['robot0_eef_pos']
                next_obs = np.concatenate([
                                                h_direction, 
                                                (f_point-h_point-np.dot(f_point-h_point, h_direction)*h_direction)
                                            ])
                obs = next_obs

                if env._check_success():
                    print("Success!")
                    success_list.append(1)
                    # env.save_video(video_path)
                    break
                elif loosened_success_threshold:
                    if env._get_observations()["open_progress"] > loosened_success_threshold:
                        print(loosened_success_threshold)
                        print("Partial Success!")
                        success_list.append(1)
                        # env.save_video(video_path)
                        break
                if done:
                    print("Failed!")
                    # env.save_video(video_path)
                    success_list.append(0)
                    break
                # env.render()

            # env.close()
        print("average success ", np.mean(success_list))
        object_success_dict[env_kwargs["object_name"]] = np.mean(success_list)
        # save the results
        with open(os.path.join(eval_results_dir, f"results_{object_type}_{success_threshold}.json"), "w") as f:
            json.dump(object_success_dict, f)


def delete_camera_frame_grasp_proposals():
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cam_frame_grasp_proposals_dir = os.path.join(project_dir, "outputs/grasp_proposals/camera_frame_proposals")
    for f in os.listdir(cam_frame_grasp_proposals_dir):
        os.remove(os.path.join(cam_frame_grasp_proposals_dir, f))
    # delete images as well
    img_dir = os.path.join(project_dir, "outputs/grasp_proposals/grasp_proposals_img")
    for f in os.listdir(img_dir):
        os.remove(os.path.join(img_dir, f))



if __name__ == "__main__":
    experiment_name = "baseline_revolute_trail"
    object_type = "microwave"
    success_threshold = 0.7
    # delete_camera_frame_grasp_proposals()
    for i in range(10):
        eval_gamma_revolute(experiment_name, i, object_type=object_type, success_threshold=success_threshold)
        delete_camera_frame_grasp_proposals()
        time.sleep(1)
