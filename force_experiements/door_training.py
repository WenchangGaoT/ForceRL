import numpy as np

import robosuite as suite
import yaml
import json

from agent.td3 import TD3, ReplayBuffer
from env.original_door_env import OriginalDoorEnv 
from env.wrappers import ActionRepeatWrapper
import time


env:OriginalDoorEnv = suite.make(
    "OriginalDoorEnv",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)


obs = env.reset()
env.render()
action = np.array([0, 0, 0])
obs, reward, done, info = env.step(action)
env.render()



for i in range(1000):
    action = np.array([20, 20, 0])
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.5)
