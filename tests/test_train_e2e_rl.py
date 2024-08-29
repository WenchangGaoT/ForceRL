from env.baseline_prismatic_training_env import BaselineTrainPrismaticEnv
import open3d as o3d
import robosuite as suite
from utils.sim_utils import get_pointcloud, flip_image
from utils.gpd_utils import gpd_get_grasp_pose
import numpy as np
import matplotlib.pyplot as plt
from utils.sim_utils import initialize_robot_position
import robosuite.utils.transform_utils as transform
from utils.control_utils import PathPlanner, Linear, Gaussian
import os 
import json
from scipy.spatial.transform import Rotation as R 
import utils.sim_utils as sim_utils
from grasps.aograsp.get_pointclouds import get_aograsp_ply_and_config
from grasps.aograsp.get_affordance import get_affordance_main
from grasps.aograsp.get_proposals import get_grasp_proposals_main
from gamma.get_joint_param import get_joint_param_main
from agent.td3 import TD3
from termcolor import colored, cprint
import termcolor
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



controller_name = "OSC_POSE"
controller_cfg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"controller_configs")
controller_cfg_path = os.path.join(controller_cfg_dir, "osc_pose_absolute.json")
controller_configs = suite.load_controller_config(default_controller=controller_name)

with open(controller_cfg_path, 'r') as f:
    controller_configs = json.load(f)


env_kwargs = dict(
    # robots="Panda",
    robots="UR5e",
    object_name = "train-drawer-1",
    obj_rotation=(-np.pi/2, -np.pi/2),
    scale_object = True,
    object_scale = 0.5,
    has_renderer=True,
    use_camera_obs=True,
    has_offscreen_renderer=True,
    camera_depths = True,
    camera_segmentations = "element",
    controller_configs=controller_configs,
    control_freq = 20,
    horizon=10000,
    camera_names = ["birdview", "agentview", "frontview", "sideview"],
    # render_camera = "birdview",
    camera_heights = [1024,256,512,1024],
    camera_widths = [1024,1024,1024,1024],
    move_robot_away = False,

    rotate_around_robot = True,
    object_robot_distance = (0.7,0.7),
    open_percentage = 0.4,

    cache_video = True,
    get_grasp_proposals_flag = True,
)
env_name = "BaselineTrainPrismaticEnv"

env:BaselineTrainPrismaticEnv = suite.make(
    env_name,
    **env_kwargs
)

obs = env.reset()



# set the open percentage

obs_dim = 6
action_dim = 3
gamma = 0.99 
lr = 0.001
lr_critic = 0.0001 
K_epochs = 4 
eps_clip = 0.2 
action_std = 0.5 
has_continuous_action_space = True
max_action = float(0.05)



final_grasp_pos = obs["grasp_pos"]
robot_gripper_pos = obs["gripper_pos"]
rotation_vector = obs["grasp_rot"]
prepaer_grasp_pos = final_grasp_pos + np.array([-0., 0, 0.15])

action = np.concatenate([robot_gripper_pos, rotation_vector, [-1]])

for i in range(80):
    # action = np.zeros_like(env.action_spec[0])
    action = np.concatenate([prepaer_grasp_pos, rotation_vector, [-1]])
    env.step(action)
    
    env.render()
    print(env.frames[0].shape)


for i in range(50):
    # action = np.zeros_like(env.action_spec[0])
    action = np.concatenate([final_grasp_pos, rotation_vector, [-1]])
    env.step(action)
    env.render()

# close the gripper
for i in range(20):
    action = np.concatenate([final_grasp_pos, rotation_vector, [1]])
    env.step(action)
    env.render()

# move the robot according to the policy
last_grasp_pos = final_grasp_pos
action_direrction = np.array([0,0.05,0])
for i in range(50):
    action = np.concatenate([last_grasp_pos + action_direrction, rotation_vector, [1]])
    # print("action: ", action)
    obs,_,_,_ = env.step(action)
    env.render()
    last_grasp_pos = obs["gripper_pos"]

done = False
last_grasp_pos = final_grasp_pos



