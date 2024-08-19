import numpy as np
import torch
import robosuite as suite
import yaml
import json

from agent.td3 import TD3, ReplayBuffer
from env.drawer_opening import DrawerOpeningEnv 
from env.wrappers import ActionRepeatWrapper

def train(run_id):
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

    cur_run_rwds = [] 
    cur_run_projections = []
    # training procedure:
    for episode in range(1, max_episodes+1): 
        cur_ep_rwds = [] 
        cur_ep_projections = []
        state = env.reset()
        state = np.concatenate([state['handle_pos'], state['handle_quat']]) 
        for t in range(max_timesteps):
            # select action and add exploration noise:
            action = policy.select_action(state)
            action = action + np.random.normal(0, exploration_noise, size=3) 
            cur_ep_projections.append(action[1]/(np.linalg.norm(action) + 1e-8))
            # action = action.clip(env.action_space.low, env.action_space.high)
            
            # take action in env:
            next_state, reward, done, _ = env.step(action) 
            next_state = np.concatenate([next_state['handle_pos'], next_state['handle_quat']]) 
            cur_ep_reward += reward 
            cur_ep_rwds.append(reward)
            replay_buffer.add((state, action, reward, next_state, float(done)))
            state = next_state
            
            # if episode is done then update policy:
            if done or t==(max_timesteps-1):
                policy.update(replay_buffer, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay) 
                print(f'Episode {episode}: rwd: {cur_ep_reward}')  
                cur_run_rwds.append(np.mean(cur_ep_rwds)) 
                cur_run_projections.append(np.mean(cur_ep_projections))

                cur_ep_rwds = [] 
                cur_ep_projections = []

                cur_ep_reward = 0
                break 
    
    with open(f'td3_reward_train_{run_id}.json', 'w') as f:
        json.dump(cur_run_rwds, f) 
    with open(f'td3_projection_train_{run_id}.json', 'w') as f:
        json.dump(cur_run_projections, f)

    cur_run_eval_rwds = [] 
    cur_run_eval_projections = []
    for ep in range(10): 
        cur_ep_rwds = [] 
        cur_ep_projections = []
        state = env.reset() 
        state = np.concatenate([state['handle_pos'], state['handle_quat']])
        done = False 
        cur_ep_reward = 0
        while not done: 
            action = policy.select_action(state) 
            cur_ep_projections.append(action[1]/(np.linalg.norm(action)+1e-8))
            print(action[1]/(np.linalg.norm(action) + 1e-8))
            next_state, reward, done, _ = env.step(action) 
            cur_ep_rwds.append(reward)
            if reward != 100:
                print(reward)
            next_state = np.concatenate([next_state['handle_pos'], next_state['handle_quat']])
            cur_ep_reward += reward
            state = next_state
            env.render() 
        print(f'Episode {ep}: rwd: {cur_ep_reward}') 
        cur_run_eval_rwds.append(np.mean(cur_ep_rwds))
        cur_run_eval_projections.append(np.mean(cur_ep_projections))

    with open(f'td3_reward_eval_{run_id}.json', 'w') as f:
        json.dump(cur_run_eval_rwds, f) 
    with open(f'td3_projection_eval_{run_id}.json', 'w') as f:
        json.dump(cur_run_eval_projections, f)
    

if __name__ == '__main__':
    for i in range(10):
        train(i)
    