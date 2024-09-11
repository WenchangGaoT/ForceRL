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

if __name__ == "__main__":
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

    env = suite.make(
        "BaselineEvalRevoluteEnv",
        **env_kwargs,)