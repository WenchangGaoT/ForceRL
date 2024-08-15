import numpy as np

import robosuite as suite
import yaml
import json

from agent.td3 import TD3, ReplayBuffer
from env.train_prismatic_env import TrainPrismaticEnv
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
    obs_dim = 6
    action_dim = 3
    max_action = float(5)

    cur_run_rewards = []
    cur_run_projections = [] 
    cur_run_eval_rwds = [] 
    cur_run_eval_projections = [] 
    cur_run_eval_success_rates = []


    policy = TD3(lr, obs_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer() 

    policy_noise_schedule = [
        1, 
        0.4, 
        0.2, 
        0.1
    ]

    curriculas = [
        # ((-np.pi / 2 - 0.25, -np.pi / 2), True, "prismatic"),
         ((-0.25, 0.25), True, "prismatic"),
         ((-np.pi / 2.0, 0), True, "prismatic"),
         ((-np.pi / 2.0, np.pi / 2.0), True, "prismatic"),
         ((-np.pi, 0), True, "prismatic"),
         ((-np.pi, np.pi), True, "prismatic"),
    ]

    episodes_schedule = [
        # 30,
        50,
        50,
        100, 
        100, 
        100, 
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
            

            available_objects_dict = TrainPrismaticEnv.available_objects()
            current_curriculum_available_objects = available_objects_dict[current_curriculum[2]]
            print(f'available objects: {current_curriculum_available_objects}')
            # randomly choose an object from the available objects
            current_curriculum_object = np.random.choice(current_curriculum_available_objects)
            print(f'episode {episodes} is training with object {current_curriculum_object}')


            curriculum_env_kwargs = {
                "init_object_angle": current_curriculum[0],
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
                "TrainPrismaticEnv", 
                **curriculum_env_kwargs
                )
        
            env: TrainPrismaticEnv = ActionRepeatWrapperNew(raw_env, action_repeat) 
            cur_ep_rwds = [] 
            cur_ep_projections = []
            
            obs = env.reset()
            # the prismatic training only needs the joint direction
            obs = np.concatenate([obs["joint_direction"], obs["force_point_relative_to_start"]])

            done = False 

            if episodes % 20 == 0:
                cur_noise_idx = min(cur_noise_idx + 1, len(policy_noise_schedule)-1) 

            # training for each episodes
            for t in range(max_timesteps):

                action = policy.select_action(obs)

                next_obs, reward, done, _ = env.step(action)  
                
                next_obs = np.concatenate([next_obs["joint_direction"], next_obs["force_point_relative_to_start"]])
                
                cur_ep_rwds.append(reward)
                # env.render()

                action_projection = env.current_action_projection
                cur_ep_projections.append(action_projection)

                replay_buffer.add((obs, action, reward, next_obs, float(done)))
                obs = next_obs

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
                    available_objects_dict = TrainPrismaticEnv.available_objects()

                    current_curriculum_available_objects = available_objects_dict[current_curriculum[2]]
                    # randomly choose an object from the available objects
                    current_curriculum_object = np.random.choice(current_curriculum_available_objects)
                    print(f'episode {episodes} is training with object {current_curriculum_object}')


                    curriculum_env_kwargs = {
                        "init_object_angle": current_curriculum[0],
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
                        "TrainPrismaticEnv", 
                        **curriculum_env_kwargs
                        )
                
                    env: TrainPrismaticEnv = ActionRepeatWrapperNew(raw_env, action_repeat) 

                    cur_ep_eval_ep_rwds = [] 
                    cur_ep_eval_projections = []

                    obs = env.reset()
                    obs = np.concatenate([obs["joint_direction"], obs["force_point_relative_to_start"]])               
                    
                    done = False
                    
                    for t in range(max_timesteps):

                        action = policy.select_action(obs)

                        next_obs, reward, done, _ = env.step(action)

                        # make our observation
                        next_obs = np.concatenate([next_obs["joint_direction"], next_obs["force_point_relative_to_start"]])
                         
                        action_projection = env.current_action_projection
                        cur_ep_eval_ep_rwds.append(reward)
                        cur_ep_eval_projections.append(action_projection) 
                        env.render()

                        obs = next_obs

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

        # make the current run json directory
        current_run_json_dir = os.path.join(json_dir, f"run_{run_id}")
        if not os.path.exists(current_run_json_dir):
            os.makedirs(current_run_json_dir)


        with open(f'{json_dir}/run_{run_id}/{algo_name}_reward_train_{run_id}_curriculum_{curriculum_idx}.json', 'w') as f:
            json.dump(cur_curriculum_rewards, f) 
        
        with open(f'{json_dir}/run_{run_id}/{algo_name}_success_rate_train_{run_id}_curriculum_{curriculum_idx}.json', 'w') as f:
            json.dump(cur_curriculum_success_rates, f) 


        # make the current curriculum policy checkpoint directory
        current_curriculum_checkpoint_dir = os.path.join(checkpoint_dir, f"run_{run_id}/curriculum_{curriculum_idx}")
        if not os.path.exists(current_curriculum_checkpoint_dir):
            os.makedirs(current_curriculum_checkpoint_dir)
        # save the checkpoint
        policy.save(
            directory=current_curriculum_checkpoint_dir,
            name=f"{algo_name}_{run_id}_curriculum_{curriculum_idx}")

    # record the evaluation results
    with open(f'{json_dir}/run_{run_id}/{algo_name}_reward_eval_{run_id}.json', 'w') as f:
        json.dump(cur_run_eval_rwds, f)
    with open(f'{json_dir}/run_{run_id}/{algo_name}_projection_eval_{run_id}.json', 'w') as f:
        json.dump(cur_run_eval_projections, f)
    with open(f'{json_dir}/run_{run_id}/{algo_name}_success_rate_eval_{run_id}.json', 'w') as f:
        json.dump(cur_run_eval_success_rates, f)




if __name__ == "__main__":

    file_dir = os.path.dirname(os.path.abspath(__file__))

    algo_name = "prismatic_td3"

    project_dir = os.path.dirname(file_dir)
    output_dir = os.path.join(file_dir, f"force_training_outputs/{algo_name}")
    checkpoint_dir = os.path.join(project_dir, f"checkpoints/force_policies/{algo_name}")

    # create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # create the checkpoint directory if it does not exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    
    for trial in range(20):
        train(trial, output_dir, algo_name, checkpoint_dir)
    
