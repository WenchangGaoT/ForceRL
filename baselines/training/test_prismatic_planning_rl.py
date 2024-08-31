from env.baseline_revolute_training_env import BaselineTrainRevoluteEnv
from env.baseline_prismatic_training_env import BaselineTrainPrismaticEnv
from env.wrappers import GraspStateWrapper
import open3d as o3d
import robosuite as suite
import numpy as np
import matplotlib.pyplot as plt
import robosuite.utils.transform_utils as transform
import os 
import json
from scipy.spatial.transform import Rotation as R 
from agent.td3 import TD3, ReplayBuffer
from utils.logger import Logger
from copy import deepcopy
import time
import xml.etree.ElementTree as ET
import time

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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
    object_name = "train-drawer-1",
    # obj_rotation=(-np.pi/2, -np.pi/2),
    # obj_rotation=(0, 0),
    obj_rotation=(-np.pi *3/ 4, -np.pi / 4),
    scale_object = True,
    object_scale = 0.5,
    has_renderer=True,
    use_camera_obs=False,
    has_offscreen_renderer=True,
    camera_depths = True,
    camera_segmentations = "element",
    controller_configs=controller_configs,
    control_freq = 20,
    horizon=10000,
    camera_names = ["birdview", "agentview", "frontview", "sideview"],
    # render_camera = "birdview",
    camera_heights = [256,256,256,256],
    camera_widths = [256,256,256,256],
    move_robot_away = False,

    rotate_around_robot = True,
    object_robot_distance = (0.8,0.8),
    open_percentage = 0.4,

    cache_video = False,
    get_grasp_proposals_flag = True,
    skip_object_initialization=True
)

env_name = "BaselineTrainPrismaticEnv"
env = suite.make(
    env_name,
    **env_kwargs
)
env = GraspStateWrapper(env, number_of_grasp_states=4)
action = [0,0,0,0,0,0,1]
for i in range(4):
    obs = env.reset(i)
    gripper_quat = obs["gripper_quat"]
    gripper_rot = R.from_quat(gripper_quat).as_euler("zyx")
    for i in range(8):
        # action[3:6] = gripper_rot
        obs,rwd,_,_ = env.step(action)
        print(env.get_stage())
        # print(rwd)
        env.render()
        # time.sleep(0.5)
    print("Done")
