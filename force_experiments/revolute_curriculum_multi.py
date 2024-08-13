import numpy as np

import robosuite as suite
import yaml
import json

from agent.td3 import TD3, ReplayBuffer
from env.original_door_env import OriginalDoorEnv 
from env.train_multiple_revolute_env import MultipleRevoluteEnv
from env.wrappers import ActionRepeatWrapperNew
import time
import os


def train(run_id, json_dir, algo_name, checkpoint_dir = "outputs"):
    gamma = 0.99                # discount for future rewards
    batch_size = 100            # num of transitions sampled from replay buffer
    lr = 0.001
    polyak = 0.995              # target policy update parameter (1-tau)
    noise_clip = 1
    policy_delay = 2            # delayed policy updates parameter
    max_timesteps = 200       # max timesteps in one episode
    rollouts = 5
    action_repeat = 1 
    state_dim = 6
    action_dim = 3
    max_action = float(5)

    cur_run_rewards = []
    cur_run_projections = [] 
    cur_run_success_rates = []
    cur_run_eval_rwds = [] 
    cur_run_eval_projections = [] 
    cur_run_eval_success_rates = []
    cur_run_eval_curricula_success_rates = [] 
    cur_run_eval_curricula_rewards = []


    policy = TD3(lr, state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer() 

    policy_noise_schedule = [
        1, 
        0.4, 
        0.2, 
        0.1
    ]

    # each curriculum is tuple (revolute_obj_pose, random_force_point, object_name)
    # curriculas = [
    #      ((-0.25, 0.25), True, "door-like"),
    #      ((-np.pi / 2.0, 0), False, "single-obj"),
    #      ((-np.pi / 2.0, np.pi / 2.0), False, "single-obj"),
    #      ((-np.pi, 0), False, "single-obj"),
    #      ((-np.pi, np.pi), False, "single-obj"),
    #      ((-np.pi, np.pi), True, "single-obj"),
    #      ((-np.pi, np.pi), True, "door-like"),
    #      ((-np.pi, np.pi), True, "dishwasher-like"),
    #      ((-np.pi, np.pi), True, "all"),
    # ]

    curriculas = [
         ((-0.25, 0.25), True, "door-like"),
         ((-np.pi / 2.0, 0), True, "door-like"),
         ((-np.pi / 2.0, np.pi / 2.0), True, "door-like"),
         ((-np.pi, 0), True, "door-like"),
         ((-np.pi, np.pi), True, "single-obj"),
         ((-np.pi, np.pi), True, "single-obj"),
         ((-np.pi, np.pi), True, "door-like"),
         ((-np.pi, np.pi), True, "dishwasher-like"),
         ((-np.pi, np.pi), True, "all"),
    ]

    episodes_schedule = [
        100,
        100,
        200, 
        200, 
        300, 
        300,
        300, 
        300,
        300,
    ]

    assert len(curriculas) == len(episodes_schedule), "The length of curriculas and episodes_schedule should be the same."


    for curriculum_idx, current_curriculum in enumerate(curriculas):


        # some list to record training data
        cur_curriculum_rewards = []
        cur_curriculum_successes = []
        cur_curriculum_projections = []
        cur_curriculum_success_rates = []

        cur_noise_idx = 0

        for episodes in range(episodes_schedule[curriculum_idx]):
            

            available_objects_dict = MultipleRevoluteEnv.available_objects()
            current_curriculum_available_objects = available_objects_dict[current_curriculum[2]]
            # randomly choose an object from the available objects
            current_curriculum_object = np.random.choice(current_curriculum_available_objects)
            print(f'episode {episodes} is training with object {current_curriculum_object}')


            curriculum_env_kwargs = {
                "init_door_angle": current_curriculum[0],
                "has_renderer": False,
                "has_offscreen_renderer": False,
                "use_camera_obs": False,
                "control_freq": 20,
                "horizon": max_timesteps,
                "reward_scale": 1.0,
                "random_force_point": current_curriculum[1],
                "object_name": current_curriculum_object
            }

            # Environment initialization
            raw_env = suite.make(
                "MultipleRevoluteEnv", 
                **curriculum_env_kwargs
                )
        
            env: MultipleRevoluteEnv = ActionRepeatWrapperNew(raw_env, action_repeat) 
            cur_ep_rwds = [] 
            cur_ep_projections = []
            
            state = env.reset()
            h_point, f_point, h_direction = state['hinge_position'], state['force_point'], state['hinge_direction']
            state = np.concatenate([
                                    h_direction, 
                                    f_point-h_point-np.dot(f_point-h_point, h_direction)*h_direction
                                    ])
            done = False 

            if episodes % 20 == 0:
                cur_noise_idx = min(cur_noise_idx + 1, len(policy_noise_schedule)-1) 

            # training for each episodes
            for t in range(max_timesteps):

                action = policy.select_action(state)

                next_state, reward, done, _ = env.step(action)  
                
                # make our observation
                h_point, f_point, h_direction = next_state['hinge_position'], next_state['force_point'], next_state['hinge_direction']
                next_state = np.concatenate([
                                             h_direction, 
                                             f_point-h_point-np.dot(f_point-h_point, h_direction)*h_direction
                                            ])
                
                cur_ep_rwds.append(reward)
                # env.render()

                action_projection = env.current_action_projection
                cur_ep_projections.append(action_projection)

                replay_buffer.add((state, action, reward, next_state, float(done)))
                state = next_state

                if done or t==(max_timesteps-1):

                    policy.update(replay_buffer, t, batch_size, gamma, polyak, policy_noise_schedule[cur_noise_idx], noise_clip, policy_delay)

                    print(f'Episode {episodes}: rwd: {np.sum(cur_ep_rwds)}') 

                    episode_success = env.success
                    # add current episode data to the list
                    cur_curriculum_rewards.append(np.sum(cur_ep_rwds))
                    cur_curriculum_successes.append(episode_success) # The task is considered success if reward >= 800.
                    cur_curriculum_projections.append(np.mean(cur_ep_projections))

                    env.close()
                    break

            cur_curriculum_latest_rwd = np.mean(cur_curriculum_rewards[max(-20, -episodes):]) 
            cur_curriculum_latest_sr = np.mean(cur_curriculum_successes[max(-20, -episodes):])
            print(f'Curriculum {curriculum_idx} average reward (past 20 episode): {cur_curriculum_latest_rwd}')
            print(f'Curriculum {curriculum_idx} success rate (past 20 episode): {cur_curriculum_latest_sr}')


            cur_curriculum_success_rates.append(np.mean(cur_curriculum_successes))

            # do the rollouts
            if episodes >= episodes_schedule[curriculum_idx] - 1: # Current curriculum is finished
                print(f'Curriculum {curriculum_idx} is finished training. Evaluating.') 

                for ep in range(rollouts):

                    # make the environments  
                    available_objects_dict = MultipleRevoluteEnv.available_objects()

                    current_curriculum_available_objects = available_objects_dict[current_curriculum[2]]
                    # randomly choose an object from the available objects
                    current_curriculum_object = np.random.choice(current_curriculum_available_objects)
                    print(f'episode {episodes} is training with object {current_curriculum_object}')


                    curriculum_env_kwargs = {
                        "init_door_angle": current_curriculum[0],
                        "has_renderer": True,
                        "has_offscreen_renderer": True,
                        "use_camera_obs": False,
                        "control_freq": 20,
                        "horizon": max_timesteps,
                        "reward_scale": 1.0,
                        "random_force_point": current_curriculum[1],
                        "object_name": current_curriculum_object
                    }

                    # Environment initialization
                    raw_env = suite.make(
                        "MultipleRevoluteEnv", 
                        **curriculum_env_kwargs
                        )
                
                    env: MultipleRevoluteEnv = ActionRepeatWrapperNew(raw_env, action_repeat) 

                    cur_ep_eval_ep_rwds = [] 
                    cur_ep_eval_projections = []

                    state = env.reset()

                    h_point, f_point, h_direction = state['hinge_position'], state['force_point'], state['hinge_direction']
                    state = np.concatenate([
                                            h_direction, 
                                            f_point-h_point-np.dot(f_point-h_point, h_direction)*h_direction
                                            ])                    
                    done = False
                    
                    for t in range(max_timesteps):

                        action = policy.select_action(state)

                        next_state, reward, done, _ = env.step(action)

                        # make our observation
                        h_point, f_point, h_direction = next_state['hinge_position'], next_state['force_point'], next_state['hinge_direction']
                        next_state = np.concatenate([
                                                    h_direction, 
                                                    f_point-h_point-np.dot(f_point-h_point, h_direction)*h_direction
                                                    ]) 
                         
                        action_projection = env.current_action_projection
                        cur_ep_eval_ep_rwds.append(reward)
                        cur_ep_eval_projections.append(action_projection) 
                        env.render()

                        state = next_state

                        if done or t == max_timesteps-1:

                            current_episode_success = env.success

                            cur_run_eval_rwds.append(np.sum(cur_ep_eval_ep_rwds)) 
                            cur_run_eval_projections.append(np.mean(cur_ep_eval_projections))
                            cur_run_eval_success_rates.append(current_episode_success)
                            env.close()
                            break

                print(f'Curriculum {curriculum_idx} gets {np.mean(cur_run_eval_rwds[-rollouts:])} average rewards per episode, {np.mean(cur_run_eval_success_rates[-rollouts:])} success rates')

        cur_run_rewards.extend(cur_curriculum_rewards)
        cur_run_projections.extend(cur_curriculum_projections)

        with open(f'{json_dir}/{algo_name}_reward_train_{run_id}_curriculum_{curriculum_idx}.json', 'w') as f:
            json.dump(cur_curriculum_rewards, f) 
        
        with open(f'{json_dir}/{algo_name}_success_rate_train_{run_id}_curriculum_{curriculum_idx}.json', 'w') as f:
            json.dump(cur_curriculum_success_rates, f) 

        # save the checkpoint
        policy.save(
            directory=checkpoint_dir,
            name=f"{algo_name}_{run_id}_curriculum_{curriculum_idx}")

    # record the evaluation results
    with open(f'{json_dir}/{algo_name}_reward_eval_{run_id}.json', 'w') as f:
        json.dump(cur_run_eval_rwds, f)
    with open(f'{json_dir}/{algo_name}_projection_eval_{run_id}.json', 'w') as f:
        json.dump(cur_run_eval_projections, f)
    with open(f'{json_dir}/{algo_name}_success_rate_eval_{run_id}.json', 'w') as f:
        json.dump(cur_run_eval_success_rates, f)




if __name__ == "__main__":

    file_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(file_dir)
    output_dir = os.path.join(file_dir, "force_training_outputs")
    checkpoint_dir = os.path.join(project_dir, "checkpoints/force_policies")

    # create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # create the checkpoint directory if it does not exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    algo_name = "curriculum_door_continuous_random_point_td3"
    for trial in range(20):
        train(trial, output_dir, algo_name, checkpoint_dir)
    
