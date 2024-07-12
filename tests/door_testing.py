import numpy as np

import robosuite as suite
import yaml
import json

from agent.td3 import TD3, ReplayBuffer
from env.original_door_env import OriginalDoorEnv 
from env.wrappers import ActionRepeatWrapperNew
import time


max_episodes = 100         # max num of episodes
max_timesteps = 200       # max timesteps in one episode
rollouts = 5

action_repeat = 1

raw_env = suite.make(
        "OriginalDoorEnv",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=max_timesteps, 
        reward_scale=1.0,
        debug_mode=True,
        )
    
env = ActionRepeatWrapperNew(raw_env, action_repeat)

for i in range(20):
    env.reset()
    env.step([0, 0, 0])
    done = False
    while not done:
        action = [0, 5, 0]
        # action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        env.render()
        time.sleep(0.1)