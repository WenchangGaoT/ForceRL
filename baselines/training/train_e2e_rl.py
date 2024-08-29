from env.baseline_prismatic_training_env import BaselineTrainPrismaticEnv
import open3d as o3d
import robosuite as suite
import numpy as np
import matplotlib.pyplot as plt
import robosuite.utils.transform_utils as transform
import os 
import json
from scipy.spatial.transform import Rotation as R 
from agent.td3 import TD3, ReplayBuffer
from utils.logger import Logger
import warnings
from copy import deepcopy
warnings.filterwarnings("ignore", category=FutureWarning)


def train(run_id, logdir, algo_name, checkpoint_dir = "outputs", max_episodes = 100_000):
    controller_name = "OSC_POSE"
    controller_configs = suite.load_controller_config(default_controller=controller_name)
    print(controller_configs)
    # with open(controller_cfg_path, 'r') as f:
    #     controller_configs = json.load(f)


    env_kwargs = dict(
        # robots="Panda",
        robots="UR5e",
        object_name = "train-drawer-1",
        # obj_rotation=(-np.pi/2, -np.pi/2),
        # obj_rotation=(0, 0),
        obj_rotation=(-np.pi/2, -np.pi/2),
        scale_object = True,
        object_scale = 0.5,
        has_renderer=False,
        use_camera_obs=False,
        has_offscreen_renderer=False,
        camera_depths = True,
        camera_segmentations = "element",
        controller_configs=controller_configs,
        control_freq = 20,
        horizon=10000,
        camera_names = ["birdview", "agentview", "frontview", "sideview"],
        # render_camera = "birdview",
        camera_heights = [256,256,256,256],
        camera_widths = [256,256,256,256],
        move_robot_away = False,

        rotate_around_robot = True,
        object_robot_distance = (0.7,0.7),
        open_percentage = 0.4,

        cache_video = False,
        get_grasp_proposals_flag = True,
    )
    env_name = "BaselineTrainPrismaticEnv"

    env:BaselineTrainPrismaticEnv = suite.make(
        env_name,
        **env_kwargs
    )

    obs = env.reset()



    # set the open percentage
    gamma = 0.99                # discount for future rewards
    batch_size = 100            # num of transitions sampled from replay buffer
    lr = 0.001
    polyak = 0.995              # target policy update parameter (1-tau)
    noise_clip = 1
    policy_delay = 2            # delayed policy updates parameter

    obs_dim = 18
    action_dim = 7
    gamma = 0.99 
    lr = 0.001
    lr_critic = 0.0001 
    K_epochs = 4 
    eps_clip = 0.2 
    action_std = 0.5 
    has_continuous_action_space = True
    max_action = float(1)
    max_timesteps = 1000
    rollout_timesteps = 100

    rollout_every = 100
    rollouts = 1
    log_every = 20

    policy = TD3(lr, obs_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer() 

    log_path = os.path.join(logdir, f"{algo_name}_{run_id}")
    logger = Logger(log_path, step=0)
    average_episode_reward = []

    for ep in range(max_episodes):
        
        obs = env.reset()
        print(obs["open_progress"])
        obs = np.concatenate([obs["gripper_pos"], obs["gripper_quat"], obs["grasp_pos"],  obs["grasp_quat"], obs["joint_direction"], np.array([obs["open_progress"]])])
        done = False
        episode_reward = 0
        for t in range(max_timesteps):
            action = policy.select_action(obs)
            next_obs, reward, done, _= env.step(action)
            next_obs = np.concatenate([next_obs["gripper_pos"], next_obs["gripper_quat"], next_obs["grasp_pos"],  next_obs["grasp_quat"], next_obs["joint_direction"], np.array([next_obs["open_progress"]])])
            replay_buffer.add((obs, action, reward, next_obs, float(done)))
            obs = next_obs
            episode_reward += reward
            # print(t, reward)
            if done or t == max_timesteps - 1:
                average_episode_reward.append(episode_reward)
                policy.update(replay_buffer, t, batch_size,gamma, polyak, 0.2, noise_clip, policy_delay)
                break
        print(f"Episode: {ep}, Reward: {episode_reward}")

        # log the reward
        if ep % log_every == 0:
            logger.scalar("reward", np.mean(np.array(average_episode_reward)))
            logger.write(step=ep)
            average_episode_reward = []

        if ep % rollout_every == 0:

            rollout_env_kwargs = deepcopy(env_kwargs)
            rollout_env_kwargs["has_renderer"] = True
            rollout_env_kwargs["use_camera_obs"] = True
            rollout_env_kwargs["has_offscreen_renderer"] = True
            rollout_env_kwargs["cache_video"] = True

            rollout_env:BaselineTrainPrismaticEnv = suite.make(
                env_name,
                **rollout_env_kwargs
            )
            rollout_rewards = []
            for _ in range(rollouts):
                
                obs = rollout_env.reset()
                obs = np.concatenate([obs["gripper_pos"], obs["gripper_quat"], obs["grasp_pos"],  obs["grasp_quat"], obs["joint_direction"], np.array([obs["open_progress"]])])
                done = False
                episode_reward = 0
                for t in range(rollout_timesteps):
                    action = policy.select_action(obs)
                    next_obs, reward, done, _= rollout_env.step(action)
                    next_obs = np.concatenate([next_obs["gripper_pos"], next_obs["gripper_quat"], next_obs["grasp_pos"],  next_obs["grasp_quat"], next_obs["joint_direction"], np.array([next_obs["open_progress"]])])
                    obs = next_obs
                    episode_reward += reward
                    # print(t, reward)
                    if done or t == rollout_timesteps - 1:
                        rollout_rewards.append(episode_reward)
                        break
            average_reward = np.mean(rollout_rewards)
            print(f"Episode: {ep}, Reward: {episode_reward}")
            logger.scalar("eval avg reward", average_reward)
            # get the video
            rollout_video = rollout_env.frames
            logger.video("rollout video", rollout_video)
            logger.write(step=ep, fps=120)

if __name__ == "__main__":
    baseline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(baseline_dir, "logs")
    train(1, logs_dir, "baseline_td3_test",max_episodes = 10_000)
    




