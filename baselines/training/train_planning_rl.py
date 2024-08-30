from env.cips_baseline_revolute_training_env import CipsBaselineTrainRevoluteEnv
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

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)




controller_name = "OSC_POSE"
controller_configs = suite.load_controller_config(default_controller=controller_name)
print(controller_configs)

env_kwargs = dict(
    # robots="Panda",
    robots="UR5e",
    object_name = "train-microwave-1",
    obj_rotation=(-np.pi / 2, -np.pi / 2),
    # obj_rotation=(-np.pi/2, -np.pi/2),
    # obj_rotation=(-np.pi/2, -np.pi/2),
    # obj_rotation=(-np.pi*2/3, -np.pi*2/3),
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
    object_robot_distance = (0.8,0.8),
    open_percentage = np.pi/6,

    cache_video = False,
    get_grasp_proposals_flag = True,

    use_grasp_states = True,
    number_of_grasp_states=2,
)

env_name = "CipsBaselineTrainRevoluteEnv"

env:CipsBaselineTrainRevoluteEnv = suite.make(
    env_name,
    **env_kwargs
)
env.reset()
env_model_dir = env.env_model_dir
# env_model_path = os.path.join(env_model_dir, "RobotRevoluteOpening_0.xml")
# # env.reset_from_xml_string()
# obs = env.reset()
env.step(np.zeros(7))
# env.render()
grasp_state = env.get_grasp_states()
env.sim.set_state_from_flattened(grasp_state)
env.sim.forward()
env.render()
# time.sleep(1)
for i in range(100):
    action = np.zeros(7)
    action[-1] = 1
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(1)

