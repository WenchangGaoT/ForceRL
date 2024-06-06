import numpy as np
import torch
import robosuite as suite
import yaml

from agent.ppo import PPO 
from env.drawer_opening import DrawerOpeningEnv 
from env.wrappers import ActionRepeatWrapper


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

# with open('configs.yaml', 'r') as f:
#     config = yaml.safe_load(f)

ppo_direction_agent = PPO( 
    state_dim, 
    action_dim, 
    lr_actor, 
    lr_critic, 
    gamma, 
    K_epochs, 
    eps_clip, 
    has_continuous_action_space, 
    action_std 
    )  
ppo_magnitude_agent = PPO( 
    state_dim, 
    1, 
    lr_actor, 
    lr_critic, 
    gamma, 
    K_epochs, 
    eps_clip, 
    True, 
    action_std,
    max_val=4
)  

# ppo_direction_agent.load('direction_100.pth')  
# ppo_direction_agent.eval()
# ppo_magnitude_agent.load('magnitude_100.pth') 
# ppo_magnitude_agent.eval()

for ep in range(100): 
    state = env.reset() 
    done = False 
    cur_ep_reward = 0
    while not done: 
        state = np.concatenate([state['handle_pos'], state['handle_quat']]) 
        action_direction = ppo_direction_agent.select_action(state)
        action_magnitude = ppo_magnitude_agent.select_action(state) 
        action = action_direction / np.linalg.norm(action_direction) * action_magnitude * 2
        # print(action)
        # print(action_magnitude)
            # if np.linalg.norm(action_magnitude) > 0.01 else np.zeros_like(action_direction) 
        # print(action_magnitude)
        # action = action_direction*action_magnitude
        # action = action_direction
        state_, reward, done, _ = env.step(action) 
        cur_ep_reward += reward
        ppo_direction_agent.buffer.rewards.append(reward) 
        ppo_magnitude_agent.buffer.rewards.append(reward)
        ppo_direction_agent.buffer.is_terminals.append(done)
        ppo_magnitude_agent.buffer.is_terminals.append(done)
        state = state_
        # if ep % 50 == 0:
        #     env.render()
        # env.render() 
    ppo_direction_agent.update() 
    # ppo_magnitude_agent.update()
    print(f'Episode {ep}: rwd: {cur_ep_reward}') 

ppo_direction_agent.save('direction_100.pth')
ppo_magnitude_agent.save('magnitude_100.pth') 

ppo_direction_agent.eval() 
ppo_magnitude_agent.eval()

for ep in range(10): 
    state = env.reset() 
    done = False 
    cur_ep_reward = 0
    while not done: 
        state = np.concatenate([state['handle_pos'], state['handle_quat']]) 
        action_direction = ppo_direction_agent.select_action(state)
        action_magnitude = ppo_magnitude_agent.select_action(state) 
        action = action_direction / (np.linalg.norm(action_direction)+1e-7) * action_magnitude * 2
        # print(action)
        # print(action_magnitude)
        # print(action_magnitude)
        # action = action_direction*action_magnitude
        # action = action_direction
        state_, reward, done, _ = env.step(action) 
        cur_ep_reward += reward
        ppo_direction_agent.buffer.rewards.append(reward) 
        ppo_magnitude_agent.buffer.rewards.append(reward)
        ppo_direction_agent.buffer.is_terminals.append(done)
        ppo_magnitude_agent.buffer.is_terminals.append(done)
        state = state_
        # if ep % 50 == 0:
        #     env.render()
        env.render() 
    ppo_direction_agent.update() 
    # ppo_magnitude_agent.update()
    print(f'Episode {ep}: rwd: {cur_ep_reward}') 