from env.baseline_revolute_training_env import BaselineTrainRevoluteEnv
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

env_kwargs = dict(
    # robots="Panda",
    robots="UR5e",
    object_name = "train-dishwasher-1",
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
    horizon=10000,
    camera_names = ["birdview", "agentview", "frontview", "sideview"],
    # render_camera = "birdview",
    camera_heights = [256,256,256,256],
    camera_widths = [256,256,256,256],
    move_robot_away = False,

    rotate_around_robot = True,
    object_robot_distance = (0.7,0.7),
    open_percentage = 0.8,

    cache_video = False,
    get_grasp_proposals_flag = True,
    skip_object_initialization=True
)

env_name = "BaselineTrainRevoluteEnv"
env = suite.make(
    env_name,
    **env_kwargs
)
env = GraspStateWrapper(env, number_of_grasp_states=2)

env.reset()
for i in range(1000):
    
    action = np.zeros(7)
    action[-1] = 1
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(1)
    if done:
        env.reset()
