# grasp the cube and roll out the policy
import open3d as o3d
import robosuite as suite
from agent.ppo import PPO
from env.robot_env_cube_move import RobotCubeMove
from utils.sim_utils import  get_pointcloud, initialize_robot_position, close_gripper, open_gripper
from utils.gpd_utils import gpd_get_grasp_pose
import numpy as np
import time

controller_name = "OSC_POSE"
controller_configs = suite.load_controller_config(default_controller=controller_name)
env = suite.make(
    "RobotCubeMove",
    robots="Panda",
    randomize_drawer=False,
    has_renderer=True,
    use_camera_obs=True,
    has_offscreen_renderer=True,
    camera_depths = True,
    camera_segmentations = "element",
    controller_configs=controller_configs,
    control_freq = 20,
    horizon=10000,
    camera_names = ["birdview", "agentview", "frontview", "sideview"],
    camera_heights = 256,
    camera_widths = 256
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

# reset env
obs = env.reset()
env.render()
# get pointcloud
pointcloud = get_pointcloud(env, obs, "agentview", 256, 256, ["cube", "table"])
# o3d.visualization.draw_geometries([pointcloud])

# save pointcloud to pcd file
o3d.io.write_point_cloud("point_clouds/demo_cube_pointcloud.pcd", pointcloud)

# get grasp pose
x = gpd_get_grasp_pose('demo_cube_pointcloud.pcd',cfg_file_name='test.cfg')
print(x.pos_x[0], x.pos_y[0], x.pos_z[0])
goal_pose = np.array([x.pos_x[0], x.pos_y[0], x.pos_z[0]])
# initialize robot position
obs = initialize_robot_position(env, goal_pose)
obs = open_gripper(env)
goal_pose_grasp = np.array([x.pos_x[0], x.pos_y[0], x.pos_z[0] - 0.04])
obs = initialize_robot_position(env, goal_pose_grasp)

# close gripper
obs = close_gripper(env)

for i in range(10):
    obs, reward, done, _ = env.step(np.zeros(env.action_dim))
    env.render()
# time.sleep(5)


horizon = 1000   
steps = 0
action_dim = env.action_dim
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
