import robosuite as suite
from agent.ppo import PPO
from env.robot_env_cube_move import RobotCubeMove
import numpy as np


def initialize_robot_position(env, desired_position):
    obs, reward, done, _ = env.step(np.zeros(env.action_dim))
    # print(obs["gripper_pos"])
    while np.square(obs["gripper_pos"]*10 - desired_position*10).sum() > 0.1:
        action = -(obs["gripper_pos"] - desired_position) * 1
        action = np.concatenate([action, np.zeros(4)])
        obs, reward, done, _ = env.step(action)
        print(obs["gripper_pos"])
        env.render()
    return obs





controller_name = "OSC_POSE"
controller_configs = suite.load_controller_config(default_controller=controller_name)
env = suite.make(
    "RobotCubeMove",
    robots="Panda",
    has_renderer=True,
    use_camera_obs=False,
    has_offscreen_renderer=False,
    controller_configs=controller_configs,
    control_freq = 20
)

state_dim = 3
action_dim = 6
gamma = 0.99 
lr_actor = 0.0003 
lr_critic = 0.0001 
K_epochs = 4 
eps_clip = 0.2 
action_std = 0.5 
has_continuous_action_space = True 
trained_agent =  PPO(
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
trained_agent.load("model.pth")

horizon = 1000   
steps = 0
action_dim = env.action_dim
gripper_offset = np.array([0, 0, 0.8])
env.reset()
obs = initialize_robot_position(env, np.zeros(3) + gripper_offset)
done = False
while not done and steps < horizon:
    # state = env._get_observation()
    # state = obs["gripper_pos"] - gripper_offset
    state = obs["cube_pos"]
    state[2] = 0.3
    action = trained_agent.select_action(state)
    action[3:] = 0
    action[0] = np.clip(action[0], -1, 1) 
    action[1] = np.clip(action[1], -1, 1)
    action[2] = 0
    action = action / np.sqrt((action**2).sum()) * 0.5
    # print(action)   
    action = np.concatenate([action, [0]])
    # obs, reward, done, _ = env.step(action if steps % 3 == 0 else np.zeros(action_dim))
    obs, reward, done, _ = env.step(action)
    steps += 1
    env.render() 





