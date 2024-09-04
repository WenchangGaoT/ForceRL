from env.baseline_revolute_training_env import BaselineTrainRevoluteEnv
from env.wrappers import GymWrapper, GraspStateWrapper
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
    horizon=500,
    camera_names = ["birdview", "agentview", "frontview", "sideview"],
    # render_camera = "birdview",
    camera_heights = [256,256,256,256],
    camera_widths = [256,256,256,256],
    move_robot_away = False,

    rotate_around_robot = True,
    object_robot_distance = (0.8,0.8),
    open_percentage = 0.3,

    cache_video = False ,
    get_grasp_proposals_flag = True,
    skip_object_initialization=False
)


custom_action_space = [np.concatenate([-0.4*np.ones(3), -0.1*np.ones(3), [-1]]), 
                           np.concatenate([0.4*np.ones(3), 0.1*np.ones(3), [1]])]

grasp_state_wrapper_kwargs = dict(number_of_grasp_states=4,
                                    use_wrapped_reward=True,
                                    reset_joint_friction = 3.0,
                                    reset_joint_damping = 0.1,
                                    filter_object_type = "microwave",
                                    end_episode_on_success=False, 
                        # use_lossend_success=True,
                        custom_action_space=custom_action_space)



env_name = "BaselineTrainRevoluteEnv"

object_type = "microwave"
trials_per_object = 20
available_objects = BaselineTrainRevoluteEnv.available_objects()[object_type]

# load the policy
training_exp_name = "baseline_revolute_trail_1_early_end"
# training_exp_name = "baseline_revolute_test"
model_name = f"final_{training_exp_name}.zip"
# model_name = f"checkpoint_{training_exp_name}_2499600_steps.zip"
baselines_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
checkpoint_dir = os.path.join(baselines_dir, f"checkpoints/{training_exp_name}")
model_path = os.path.join(checkpoint_dir, model_name)

project_dir = os.path.dirname(baselines_dir)
video_dir = os.path.join(project_dir, "videos")
video_path = os.path.join(video_dir, f"np_nov_test_{training_exp_name}_fail.mp4")

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

    env:BaselineTrainRevoluteEnv = suite.make(
        env_name, 
        **env_kwargs
    )


    # env = GraspStateWrapper(env, **grasp_state_wrapper_kwargs)
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
        env.reset()

        env._set_door_friction(3.0)

        # action = np.array([0,1,0,0,0,0,-1])
        obs = env._flatten_obs(env._get_observations())

        for i in range(990):
            action, _states = model.predict(obs, deterministic=True)
            action[-1] = 1
            # print(i)
            obs, reward, done,_,_ = env.step(action)
            # print("done", done)
            # env.render()
            # time.sleep(0.05)

            if env._check_success():
                print("Success!")
                success_list.append(1)
                env.save_video(video_path)
                break
            elif loosened_success_threshold:
                if env._get_observations()["open_progress"] > loosened_success_threshold:
                    print("Partial Success!")
                    success_list.append(1)
                    env.save_video(video_path)
                    break

            if done:
                print("Failed!")
                env.save_video(video_path)
                success_list.append(0)
                break
            
            
    
    print("average success ", np.mean(success_list))
    object_success_dict[env_kwargs["object_name"]] = np.mean(success_list)
    # save the results
    # with open(os.path.join(eval_results_dir, f"results_{object_type}.json"), "w") as f:
    #     json.dump(object_success_dict, f)
