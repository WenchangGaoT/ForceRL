from env.baseline_revolute_eval_env import BaselineEvalRevoluteEnv
from env.wrappers import GymWrapper
import open3d as o3d
import robosuite as suite
import numpy as np
import os 
import json
from scipy.spatial.transform import Rotation as R 
from copy import deepcopy
import time
import xml.etree.ElementTree as ET
import time

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from utils.baseline_utils import drive_to_grasp

from stable_baselines3 import TD3

def eval_revolute(experiment_name, run_id):
    controller_name = "OSC_POSE"
    controller_configs = suite.load_controller_config(default_controller=controller_name)
    print(controller_configs)


    # dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # cfg_path = os.path.join(dir_path, "controller_configs/osc_pose_small_kp.json")
    # with open(cfg_path, "r") as f:
    #     controller_configs = json.load(f)


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

    object_type = "microwave"
    trials_per_object = 20
    available_objects = BaselineEvalRevoluteEnv.available_objects()[object_type]

    # load the policy
    training_exp_name = f"{experiment_name}_{run_id}"
    # training_exp_name = "baseline_revolute_test"
    model_name = f"final_{training_exp_name}.zip"
    # model_name = f"checkpoint_{training_exp_name}_2499600_steps.zip"
    baselines_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_dir = os.path.join(baselines_dir, f"checkpoints/{training_exp_name}")
    model_path = os.path.join(checkpoint_dir, model_name)

    project_dir = os.path.dirname(baselines_dir)
    video_dir = os.path.join(project_dir, "videos")
    video_path = os.path.join(video_dir, f"test_{training_exp_name}_failure.mp4")

    obs_keys = ["gripper_pos", "gripper_quat", "grasp_pos", "grasp_quat", "joint_position", "joint_direction", "open_progress"]

    eval_results_dir = os.path.join(baselines_dir, f"eval_results/{training_exp_name}")
    os.makedirs(eval_results_dir, exist_ok=True)

    object_success_dict = dict(
        object_type = object_type,
        obj_rotation = env_kwargs["obj_rotation"],
        object_robot_distance = env_kwargs["object_robot_distance"],
        open_percentage = env_kwargs["open_percentage"],
        trials_per_object = trials_per_object,
    )
    loosened_success_threshold = 0.4

    for obj_name in available_objects:
        if object_type in obj_name:
            env_kwargs["object_name"] = obj_name

        else:
            raise RuntimeError("Object not found in available objects")

        env:BaselineEvalRevoluteEnv = suite.make(
            env_name, 
            **env_kwargs
        )


        env = GymWrapper(env, keys=obs_keys)

        # obs = env.reset()


        model = TD3.load(model_path)
        success_list = []

        for exp_num in range(trials_per_object):
            print(f"Experiment {exp_num} for object {env_kwargs['object_name']}")
            drive_to_grasp(
                env = env,
                # grasp_pos= obs["grasp_pos"],
                # grasp_rot_vec=R.from_quat(obs["grasp_quat"]).as_rotvec(), 
                reload_controller_config=True, 
                use_gym_wrapper=True,render=False)

            env._set_door_friction(3.0)

            # action = np.array([0,1,0,0,0,0,-1])
            obs = env._flatten_obs(env._get_observations())

            for i in range(990):
                action, _states = model.predict(obs, deterministic=True)
                action[-1] = 1
                obs, reward, done,_,_ = env.step(action)
                # env.render()
                # time.sleep(0.05)

                if env._check_success():
                    print("Success!")
                    success_list.append(1)
                    # env.save_video(video_path)
                    break
                elif loosened_success_threshold:
                    if env._get_observations()["open_progress"] > loosened_success_threshold:
                        print("Partial Success!")
                        success_list.append(1)
                        # env.save_video(video_path)
                        break

                if done:
                    print("Failed!")
                    env.save_video(video_path)
                    success_list.append(0)
                    break
                
                
        
        print("average success ", np.mean(success_list))
        object_success_dict[env_kwargs["object_name"]] = np.mean(success_list)
        # save the results
        with open(os.path.join(eval_results_dir, f"results_{object_type}.json"), "w") as f:
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
    # delete_camera_frame_grasp_proposals()
    for i in range(3,10):
        eval_revolute(experiment_name, i)
        delete_camera_frame_grasp_proposals()
        time.sleep(1)
