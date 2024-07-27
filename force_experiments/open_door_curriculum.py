import numpy as np

import robosuite as suite
import yaml
import json

from agent.td3 import TD3, ReplayBuffer
from env.original_door_env import OriginalDoorEnv 
from env.curri_door_env import CurriculumDoorEnv
from env.wrappers import ActionRepeatWrapperNew
import time
import os


def train(run_id, json_dir, algo_name):
    gamma = 0.99                # discount for future rewards
    batch_size = 100            # num of transitions sampled from replay buffer
    lr = 0.001
    exploration_noise = 1 
    polyak = 0.995              # target policy update parameter (1-tau)
    policy_noise = 0.2          # target policy smoothing noise
    noise_clip = 1
    policy_delay = 2            # delayed policy updates parameter
    max_episodes = 200         # max num of episodes
    max_timesteps = 200       # max timesteps in one episode
    rollouts = 5
    action_repeat = 1 
    # state_dim = 9
    state_dim = 6
    action_dim = 3
    max_action = float(5)

    cur_run_rewards = []
    cur_run_projections = []
    cur_run_eval_rwds = [] 
    cur_run_eval_projections = []


    policy = TD3(lr, state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer() 

    policy_noise_schedule = [
        1, 
        0.4, 
        0.2, 
        0.1
    ]
    # cur_ep_reward = 0

    curriculas = [
        (-0.25, 0.25),
        (-np.pi / 2.0, 0), 
        (-np.pi/2.0 - 0.25, -np.pi/2.0 + 0.25),
        # (-np.pi, np.pi/2.0 + 0.25),
        (-np.pi / 4.0, np.pi / 2.0),
        (-np.pi / 2.0, np.pi / 2.0),
        (0, np.pi),
        # (np.pi / 2.0, np.pi * 1.5),
        (-np.pi, np.pi)
    ] 

    episodes_schedule = [
        100,
        100,
        100,
        200, 
        200, 
        300, 
        300
    ]


    for c_idx, c in enumerate(curriculas):
        # Environment initialization
        raw_env = suite.make(
            "CurriculumDoorEnv", 
            init_door_angle=c, 
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=False,
            control_freq=20,
            horizon=max_timesteps, 
            reward_scale=1.0, 
            random_force_point=False
            )
        
        env = ActionRepeatWrapperNew(raw_env, action_repeat) 
        cur_curriculum_rewards = []
        cur_curriculum_successes = []
        cur_curriculum_projections = []
        # env.env.debug_mode = True
        env.env.debug_mode = False 
        cur_noise_idx = 0

        for episodes in range(1, 1+episodes_schedule[c_idx]):
            env.env.debug_mode = False
            cur_ep_rwds = [] 
            cur_ep_projections = []
            state = env.reset()
            # state = np.concatenate([state['hinge_position'] - state['force_point'], state["hinge_direction"]])
            h_point, f_point, h_direction = state['hinge_position'], state['force_point'], state['hinge_direction']
            state = np.concatenate([h_direction, 
                                             f_point-h_point-np.dot(f_point-h_point, h_direction)*h_direction
                                             ])
            done = False 
            if episodes % 20 == 0:
                cur_noise_idx = min(cur_noise_idx + 1, len(policy_noise_schedule)-1)
            for t in range(max_timesteps):
                # env.render()
                action = policy.select_action(state)
                next_state, reward, done, _ = env.step(action)  
                # if c_idx == 1:
                #     print("reward: ", reward)
                action_projection = env.current_action_projection
                h_point, f_point, h_direction = next_state['hinge_position'], next_state['force_point'], next_state['hinge_direction']
                next_state = np.concatenate([h_direction, 
                                             f_point-h_point-np.dot(f_point-h_point, h_direction)*h_direction
                                             ])
                
                cur_ep_rwds.append(reward)
                cur_ep_projections.append(action_projection)

                replay_buffer.add((state, action, reward, next_state, float(done)))
                state = next_state

                # if episode is done then update policy:
                if done or t==(max_timesteps-1): 
                    policy.update(replay_buffer, t, batch_size, gamma, polyak, policy_noise_schedule[cur_noise_idx], noise_clip, policy_delay)
                    print(f'Episode {episodes}: rwd: {np.sum(cur_ep_rwds)}') 
                    cur_curriculum_rewards.append(np.sum(cur_ep_rwds))
                    cur_curriculum_successes.append(cur_curriculum_rewards[-1] >= 800) # The task is considered success if reward >= 800.
                    cur_curriculum_projections.append(np.mean(cur_ep_projections))
                    cur_ep_rwds = [] 
                    cur_ep_projections = []
                    break
                replay_buffer.add((state, action, reward, next_state, float(done)))
                state = next_state
            # env.save_video(f'videos/{algo_name}_run_{run_id}_curriculum_{c_idx}_episode_{episodes}.gif')

            print(f'Curriculum {c_idx} average reward becomes: {np.mean(cur_curriculum_rewards[max(-10, -episodes):])}')
            print(f'Curriculum {c_idx} success rate becomes {np.mean(cur_curriculum_successes)}')
            if (episodes > 30 and np.mean(cur_curriculum_successes[max(-20, -episodes):]) > 0.7 and c_idx < (len(curriculas)-1)) or episodes>=(episodes_schedule[c_idx]-1):
                print(f'Curriculum {c_idx} is finished. Proceed to next task.')
                # env.env.debug_mode = True
                for ep in range(rollouts):
                    state = env.reset()
                    # state = np.concatenate([state['force_point'],state['hinge_position'] , state["hinge_direction"]])

                    h_point, f_point, h_direction = state['hinge_position'], state['force_point'], state['hinge_direction']
                    state = np.concatenate([h_direction, 
                                                    f_point-h_point-np.dot(f_point-h_point, h_direction)*h_direction
                                                    ])                    
                    done = False
                    cur_ep_eval_ep_rwds = [] 
                    cur_ep_eval_projections = []
                    while not done:
                        action = policy.select_action(state)
                        next_state, reward, done, _ = env.step(action)
                        # next_state = np.concatenate([next_state['force_point'], next_state['hinge_position'], next_state["hinge_direction"]])
                        h_point, f_point, h_direction = next_state['hinge_position'], next_state['force_point'], next_state['hinge_direction']
                        next_state = np.concatenate([h_direction, 
                                                    f_point-h_point-np.dot(f_point-h_point, h_direction)*h_direction
                                ])  
                        action_projection = env.current_action_projection
                        cur_ep_eval_ep_rwds.append(reward)
                        cur_ep_eval_projections.append(action_projection)                      
                        state = next_state
                        # env.render()
                    cur_run_eval_rwds.append(np.sum(cur_ep_eval_ep_rwds)) 
                    cur_run_eval_projections.append(np.mean(cur_ep_eval_projections))
                
                break

        cur_run_rewards.extend(cur_curriculum_rewards)
        cur_run_projections.extend(cur_curriculum_projections)
        with open(f'{json_dir}/{algo_name}_reward_train_{run_id}.json', 'w') as f:
            json.dump(cur_run_rewards, f) 
        with open(f'{json_dir}/{algo_name}_projection_train_{run_id}.json', 'w') as f:
            json.dump(cur_run_projections, f)
        with open(f'{json_dir}/{algo_name}_reward_eval_{run_id}.json', 'w') as f:
            json.dump(cur_run_eval_rwds, f) 
        with open(f'{json_dir}/{algo_name}_projection_eval_{run_id}.json', 'w') as f:
            json.dump(cur_run_eval_projections, f)

    print('----------------------------------------------')
    print('Switching to random force point environment') 
    env.random_force_point = True
    for ep in range(1000):
        env.env.debug_mode = False
        cur_ep_rwds = [] 
        cur_ep_projections = []
        state = env.reset()
        # state = np.concatenate([state['hinge_position'] - state['force_point'], state["hinge_direction"]])
        h_point, f_point, h_direction = state['hinge_position'], state['force_point'], state['hinge_direction']
        state = np.concatenate([h_direction, 
                                            f_point-h_point-np.dot(f_point-h_point, h_direction)*h_direction
                                            ])
        done = False 
        if episodes % 20 == 0:
            cur_noise_idx = min(cur_noise_idx + 1, len(policy_noise_schedule)-1)
        for t in range(max_timesteps):
            # env.render()
            action = policy.select_action(state)
            next_state, reward, done, _ = env.step(action)  
            # if c_idx == 1:
            #     print("reward: ", reward)
            action_projection = env.current_action_projection
            h_point, f_point, h_direction = next_state['hinge_position'], next_state['force_point'], next_state['hinge_direction']
            next_state = np.concatenate([h_direction, 
                                            f_point-h_point-np.dot(f_point-h_point, h_direction)*h_direction
                                            ])
            
            cur_ep_rwds.append(reward)
            cur_ep_projections.append(action_projection)

            replay_buffer.add((state, action, reward, next_state, float(done)))
            state = next_state

            # if episode is done then update policy:
            if done or t==(max_timesteps-1): 
                policy.update(replay_buffer, t, batch_size, gamma, polyak, policy_noise_schedule[cur_noise_idx], noise_clip, policy_delay)
                print(f'Episode {episodes}: rwd: {np.sum(cur_ep_rwds)}') 
                cur_curriculum_rewards.append(np.sum(cur_ep_rwds))
                cur_curriculum_successes.append(cur_curriculum_rewards[-1] >= 800) # The task is considered success if reward >= 800.
                cur_curriculum_projections.append(np.mean(cur_ep_projections))
                cur_ep_rwds = [] 
                cur_ep_projections = []
                break
            replay_buffer.add((state, action, reward, next_state, float(done)))
            state = next_state



if __name__ == "__main__":

    file_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(file_dir, "outputs")
    algo_name = "door_increasing_random_td3"

    train(0, output_dir, algo_name)
    
