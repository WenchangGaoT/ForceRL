from env.baseline_revolute_training_env import BaselineTrainRevoluteEnv
from env.baseline_prismatic_training_env import BaselineTrainPrismaticEnv
from env.wrappers import GraspStateWrapper, GymWrapper, make_vec_env_baselines
from stable_baselines3.common.vec_env import SubprocVecEnv
import robosuite as suite
import numpy as np
import robosuite.utils.transform_utils as transform
import os 
import json
from scipy.spatial.transform import Rotation as R 
from utils.logger import Logger
import time
import xml.etree.ElementTree as ET
import time
import tensorboard

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import ProgressBarCallback, CallbackList
from utils.baseline_utils import VideoRecorderCallback, TensorboardCallback, get_eval_env_kwargs, make_eval_env


if __name__ == "__main__":
    controller_name = "OSC_POSE"
    controller_configs = suite.load_controller_config(default_controller=controller_name)
    # print(controller_configs)

    env_kwargs = dict(
        # robots="Panda",
        robots="UR5e",
        object_name = "train-dishwasher-1",
        # obj_rotation=(-np.pi/2, -np.pi/2),
        # obj_rotation=(0, 0),
        obj_rotation=(-np.pi / 2, 0),
        scale_object = True,
        object_scale = 0.3,
        has_renderer=False,
        use_camera_obs=False,
        has_offscreen_renderer=False,
        camera_depths = True,
        camera_segmentations = "element",
        controller_configs=controller_configs,
        control_freq = 20,
        horizon=500,
        camera_names = ["birdview", "agentview", "frontview", "sideview"],
        # render_camera = "birdview",
        camera_heights = [256,256,256,256],
        camera_widths = [256,256,256,256],
        move_robot_away = False,

        rotate_around_robot = True,
        object_robot_distance = (0.7,0.7),
        open_percentage = 0.3,

        cache_video = False,
        get_grasp_proposals_flag = True,
        skip_object_initialization=True
    )

    env_name = "BaselineTrainRevoluteEnv"

    grasp_state_wrapper_kwargs = dict(number_of_grasp_states=4,
                                    use_wrapped_reward=True,
                                    reset_joint_friction = 3.0,
                                    reset_joint_damping = 1.0,)

    num_envs = 12

    obs_keys = ["gripper_pos", "gripper_quat", "grasp_pos", "grasp_quat", "joint_position", "joint_direction", "open_progress"]

    # logging parameters
    experiment_name = "baseline_revolute_test"
    baselines_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(baselines_dir, f"logs/{experiment_name}")
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_dir = os.path.join(baselines_dir, f"checkpoints/{experiment_name}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    env = make_vec_env_baselines(
        env_name=env_name,
        obs_keys=obs_keys,
        env_kwargs=env_kwargs,
        n_envs=num_envs,
        grasp_state_wrapper=True,
        grasp_state_wrapper_kwargs=grasp_state_wrapper_kwargs,
        log_dir=log_dir
    )

    checkpoint_callback = CheckpointCallback(save_freq=int(50_000/num_envs), save_path=checkpoint_dir, name_prefix=f"checkpoint_{experiment_name}")
    
    eval_env_kwargs = get_eval_env_kwargs(env_kwargs)
    eval_env = make_eval_env(
        env_name=env_name,
        obs_keys=obs_keys,
        env_kwargs=eval_env_kwargs,
        grasp_state_wrapper=True,
        grasp_state_wrapper_kwargs=grasp_state_wrapper_kwargs,
        )

    video_recorder = VideoRecorderCallback(eval_env, render_freq=int(50_000/num_envs), n_eval_episodes=1)
    tensorboard_callback = TensorboardCallback(check_freq=20)

    callback_list = CallbackList([checkpoint_callback, video_recorder, tensorboard_callback])
    # callback_list = CallbackList([checkpoint_callback])

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1, tensorboard_log=log_dir)
    model.learn(total_timesteps=3_000_000, 
                log_interval=10, 
                tb_log_name=experiment_name,
                progress_bar=True, 
                callback=callback_list)
    model.save(os.path.join(checkpoint_dir, f"final_{experiment_name}"))




