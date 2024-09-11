import numpy as np

import robosuite as suite
import yaml
import json

from agent.td3 import TD3, ReplayBuffer
from env.train_prismatic_env import TrainPrismaticEnv
# from env.wrappers import ActionRepeatWrapperNew
import time
import os
from scipy.spatial.transform import Rotation as R
import utils.sim_utils as s_utils


env_name = "TrainPrismaticEnv"

env_kwargs = dict(
    object_name = "train-drawer-1",
    init_object_angle = (-np.pi, -np.pi),
    has_renderer=True,
    has_offscreen_renderer=True,
    camera_names = ["birdview", "agentview", "frontview", "sideview"],
    camera_heights = [256,256,256,256],
    camera_widths = [256,256,256,256],
    render_camera = "sideview",
    # render_camera = "agentview",
    random_force_point = True,
    save_video = True,
)

env = suite.make(
    env_name,
    **env_kwargs
)



camera_euler = np.array([ 45., 22., 3.])
camera_euler_for_pos = np.array([camera_euler[-1], camera_euler[1], -camera_euler[0]])
camera_rotation = R.from_euler("xyz", camera_euler_for_pos, degrees=True)
camera_quat = R.from_euler("xyz", camera_euler, degrees=True).as_quat()
camera_forward = np.array([1, 0, 0])
world_forward = camera_rotation.apply(camera_forward)
distance = 3
camera_pos = -distance * world_forward




obs = env.reset()

s_utils.init_camera_pose(env, camera_pos,scale_factor=1, camera_quat=camera_quat)

# env.render()
action = np.array([2,0,0])

for _ in range(150):
    obs, reward, done, info = env.step(action)
    # env.render()

file_path = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(file_path, "drawer-training.mp4")
# save the video
env.save_video(video_path)





