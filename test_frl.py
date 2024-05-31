import numpy as np
import robosuite as suite

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

ppo_agent = PPO( 
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

# ppo_agent.load('checkpoint_100.pth')

for ep in range(100): 
    state = env.reset() 
    # print(state.items())
    # state = np.concatenate([state['cube_pos'], state['cube_quat']]) 
    # print(state) 
    done = False 
    cur_ep_reward = 0
    while not done: 
        state = np.concatenate([state['handle_pos'], state['handle_quat']]) 
        # state = state['handle_pos']
        action = ppo_agent.select_action(state) 
        # action[3:] = 0
        # action[0] = np.clip(action[0], -1, 1) 
        # action[1] = np.clip(action[1], -1, 1)
        state_, reward, done, _ = env.step(action) 
        # print(reward)
        # print(action)
        cur_ep_reward += reward
        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(done)
        state = state_
        env.render() 
    ppo_agent.update() 
    print(f'Episode {ep}: rwd: {cur_ep_reward}') 

ppo_agent.save('checkpoint_100.pth')