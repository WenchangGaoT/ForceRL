import numpy as np
import torch
import robosuite as suite
import yaml
import json

from agent.ppo_singlehead import SingleheadPPO 
from env.drawer_opening import DrawerOpeningEnv 
from env.wrappers import ActionRepeatWrapper

def train(run_id):
    env = suite.make(
        "ActionRepeatWrapper",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        # reward_shaping=True,
        control_freq=20,
        horizon=500, 
        action_repeat=10
    )

    state_dim = 7
    action_dim = env.action_dim
    gamma = 0.99 
    lr_actor = 0.0003 
    lr_critic = 0.0001 
    K_epochs = 4 
    eps_clip = 0.2 
    action_std = 0.5 
    has_continuous_action_space = True 

    ppo_agent = SingleheadPPO( 
        state_dim, 
        action_dim, 
        lr_actor, 
        lr_critic, 
        gamma, 
        K_epochs, 
        eps_clip, 
        has_continuous_action_space, 
        action_std,
        max_val=100
        )  

    cur_run_rwds = [] 
    cur_run_projections = []

    for ep in range(300): 
        cur_ep_rwds = [] 
        cur_ep_projections = []
        state = env.reset() 
        done = False 
        cur_ep_reward = 0
        while not done: 
            state = np.concatenate([state['handle_pos'], state['handle_quat']]) 
            action = ppo_agent.select_action(state)
            state_, reward, done, _ = env.step(action) 
            cur_ep_rwds.append(reward)
            cur_ep_projections.append(action[1] / (np.linalg.norm(action)+1e-8))
            cur_ep_reward += reward
            ppo_agent.buffer.rewards.append(reward) 
            ppo_agent.buffer.is_terminals.append(done)
            state = state_
        ppo_agent.update() 
        cur_run_rwds.append(np.mean(cur_ep_rwds)) 
        cur_ep_rwds = []
        cur_run_projections.append(np.mean(cur_ep_projections)) 
        cur_ep_projections = []
        print(f'Episode {ep}: rwd: {cur_ep_reward}') 

    # ppo_agent.save('direction_100.pth')
    # ppo_agent.save('magnitude_100.pth') 

    ppo_agent.eval() 

    cur_run_eval_rwds = [] 
    cur_run_eval_projections = []

    for ep in range(10): 
        cur_ep_rwds = [] 
        cur_ep_projections = []
        state = env.reset() 
        done = False 
        cur_ep_reward = 0
        while not done: 
            state = np.concatenate([state['handle_pos'], state['handle_quat']]) 
            action = ppo_agent.select_action(state)
            print(action)
            state_, reward, done, _ = env.step(action) 
            cur_ep_rwds.append(reward) 
            cur_ep_projections.append(action[1]/(np.linalg.norm(action)+1e-8))
            cur_ep_reward += reward
            ppo_agent.buffer.rewards.append(reward) 
            ppo_agent.buffer.is_terminals.append(done)
            state = state_
            env.render() 
        print(f'Episode {ep}: rwd: {cur_ep_reward}')  
        cur_run_eval_rwds.append(np.mean(cur_ep_rwds))
        cur_run_eval_projections.append(np.mean(cur_ep_projections)) 
        cur_ep_rwds = [] 
        cur_ep_projections = []


    with open(f'ppo_singlehead_reward_train_{run_id}.json', 'w') as f:
        json.dump(cur_run_rwds, f) 
    with open(f'ppo_singlehead_projection_train_{run_id}.json', 'w') as f:
        json.dump(cur_run_projections, f)
    with open(f'ppo_singlehead_reward_eval_{run_id}.json', 'w') as f:
        json.dump(cur_run_eval_rwds, f) 
    with open(f'ppo_singlehead_projection_eval_{run_id}.json', 'w') as f:
        json.dump(cur_run_eval_projections, f)
if __name__ == '__main__':
    for i in range(10):
        train(i)