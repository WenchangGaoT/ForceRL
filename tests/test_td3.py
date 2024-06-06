import numpy as np
import torch
import robosuite as suite
import yaml

from agent.td3 import TD3, ReplayBuffer
from env.drawer_opening import DrawerOpeningEnv 
from env.wrappers import ActionRepeatWrapper

def train():
    gamma = 0.99                # discount for future rewards
    batch_size = 100            # num of transitions sampled from replay buffer
    lr = 0.001
    exploration_noise = 1 
    polyak = 0.995              # target policy update parameter (1-tau)
    policy_noise = 0.2          # target policy smoothing noise
    noise_clip = 0.5
    policy_delay = 2            # delayed policy updates parameter
    max_episodes = 300         # max num of episodes
    max_timesteps = 200       # max timesteps in one episode
    
    env = suite.make(
        "ActionRepeatWrapper",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        # reward_shaping=True,
        control_freq=20,
        horizon=max_timesteps, 
        action_repeat=5
        )

    state_dim = 7
    action_dim = 3
    max_action = float(2)
    
    policy = TD3(lr, state_dim, action_dim, max_action)
    # policy.load(".", "td3_checkpoint_100.pth")
    replay_buffer = ReplayBuffer()
    cur_ep_reward = 0
    # training procedure:
    for episode in range(1, max_episodes+1):
        state = env.reset()
        state = np.concatenate([state['handle_pos'], state['handle_quat']]) 
        for t in range(max_timesteps):
            # select action and add exploration noise:
            action = policy.select_action(state)
            action = action + np.random.normal(0, exploration_noise, size=3)
            # action = action.clip(env.action_space.low, env.action_space.high)
            
            # take action in env:
            next_state, reward, done, _ = env.step(action) 
            next_state = np.concatenate([next_state['handle_pos'], next_state['handle_quat']]) 
            cur_ep_reward += reward
            replay_buffer.add((state, action, reward, next_state, float(done)))
            # state = next_state
            
            # if episode is done then update policy:
            if done or t==(max_timesteps-1):
                policy.update(replay_buffer, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay) 
                print(f'Episode {episode}: rwd: {cur_ep_reward}')  
                cur_ep_reward = 0
                break 
    for ep in range(10): 
        state = env.reset() 
        state = np.concatenate([state['handle_pos'], state['handle_quat']])
        done = False 
        cur_ep_reward = 0
        while not done: 
            action = policy.select_action(state)
            print(action)
            next_state, reward, done, _ = env.step(action) 
            if reward != 100:
                print(reward)
            next_state = np.concatenate([next_state['handle_pos'], next_state['handle_quat']])
            cur_ep_reward += reward
            state = next_state
            env.render() 
        print(f'Episode {ep}: rwd: {cur_ep_reward}') 

if __name__ == '__main__':
    train()
    