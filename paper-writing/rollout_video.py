
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

controller_name = "OSC_POSE"

controller_cfg_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"baselines/controller_configs")
controller_cfg_path = os.path.join(controller_cfg_dir, "osc_pose_absolute.json")
controller_configs = suite.load_controller_config(custom_fpath = controller_cfg_path,default_controller=controller_name)

with open(controller_cfg_path, 'r') as f:
    controller_configs = json.load(f)
# print(controller_configs)


env_kwargs = dict(
    # robots="Panda",
    robots="UR5e",
    object_name = "microwave-3",
    # obj_rotation=(-np.pi/2, -np.pi/2),
    # obj_rotation=(0, 0),
    obj_rotation=(-np.pi / 2, -np.pi / 2),
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
    render_camera = "frontview",
    camera_heights = [256,256,256,256],
    camera_widths = [256,256,256,256],
    move_robot_away = False,

    rotate_around_robot = True,
    object_robot_distance = (0.7,0.7),
    open_percentage = 0.3,

    cache_video = True,
    get_grasp_proposals_flag = True,
    skip_object_initialization=False
)
env_name = "BaselineEvalRevoluteEnv"


env:BaselineEvalRevoluteEnv = suite.make(
    env_name, 
    **env_kwargs
)
env._set_door_friction(3.0)

success_list = []


# get the joint parameters
# joint_poses, joint_directions, joint_types, resuts_list = get_joint_param_main(
#     env.pcd_wf_no_downsample_path, 
#     env.camera_info_path, 
#     viz=False, 
#     return_results_list=True)

# find the first joint in the list with joint_type = 0
joint_pose_selected = None
joint_direction_selected = None

video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"microwave_opening.mp4")

# randomly select a joint
# use ground truth joint parameters


env._set_door_friction(3.0)

drive_to_grasp(
    env=env,
    reload_controller_config=True,
    use_gym_wrapper=False,
    render=False
)

obs = env._get_observations()

joint_pose_selected = obs["joint_position"]
joint_direction_selected = obs["joint_direction"]
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
    action = np.cross(obs[:3], obs[3:])
    action = action / np.linalg.norm(action)
    
    
    # print("action: ", action)
    action = action * 0.01
    action = np.concatenate([last_grasp_pos +action, rotation_vector, [1]])
    next_obs, reward, done, _ = env.step(action)
    # env.render()

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
        env.save_video(video_path)
        break
    if done:
        print("Failed!")
        env.save_video(video_path)
        success_list.append(0)
        break
    # env.render()

# env.close()