import numpy as np

import robosuite as suite
import yaml
import json

from agent.td3 import TD3, ReplayBuffer
from env.original_door_env import OriginalDoorEnv 
from env.wrappers import ActionRepeatWrapperNew
import time


def train(run_id):
    gamma = 0.99                # discount for future rewards
    batch_size = 100            # num of transitions sampled from replay buffer
    lr = 0.001
    exploration_noise = 1 
    polyak = 0.995              # target policy update parameter (1-tau)
    policy_noise = 0.2          # target policy smoothing noise
    noise_clip = 0.5
    policy_delay = 2            # delayed policy updates parameter
    max_episodes = 100         # max num of episodes
    max_timesteps = 200       # max timesteps in one episode
    rollouts = 5

    action_repeat = 1

    raw_env = suite.make(
        "OriginalDoorEnv",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        control_freq=20,
        horizon=max_timesteps, 
        reward_scale=1.0,
        )
    
    env = ActionRepeatWrapperNew(raw_env, action_repeat)

    # state_dim = 9
    state_dim = 6
    action_dim = 3
    max_action = float(5)

    policy = TD3(lr, state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()
    cur_ep_reward = 0

    for episode in range(1, max_episodes+1):
        cur_ep_rwds = [] 
        state = env.reset()

        # get the needed state information:
        # state = np.concatenate([state['force_point'],state['hinge_position'],  state["hinge_direction"]])
        state = np.concatenate([state['hinge_position'] - state['force_point'], state["hinge_direction"]])
        done = False
        for t in range(max_timesteps):
            action = policy.select_action(state)
            next_state, reward, done, _ = env.step(action)  
            # env.render()
            # next_state = np.concatenate([next_state['force_point'], next_state['hinge_position'], next_state["hinge_direction"]])
            next_state = np.concatenate([next_state['hinge_position'] - next_state['force_point'], next_state["hinge_direction"]])
            cur_ep_rwds.append(reward)

            replay_buffer.add((state, action, reward, next_state, float(done)))
            state = next_state

            # if episode is done then update policy:
            if done or t==(max_timesteps-1):
                policy.update(replay_buffer, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                print(f'Episode {episode}: rwd: {np.sum(cur_ep_rwds)}')
                cur_ep_rwds = [] 
                break
    
    for ep in range(rollouts):
        state = env.reset()
        # state = np.concatenate([state['force_point'],state['hinge_position'] , state["hinge_direction"]])

        state = np.concatenate([state['hinge_position'] - state['force_point'], state["hinge_direction"]])
        done = False
        while not done:
            action = policy.select_action(state)
            next_state, reward, done, _ = env.step(action)
            # next_state = np.concatenate([next_state['force_point'], next_state['hinge_position'], next_state["hinge_direction"]])
            next_state = np.concatenate([next_state['hinge_position'] - next_state['force_point'], next_state["hinge_direction"]])
            state = next_state
            env.render()


if __name__ == "__main__":
    train(0)
    
