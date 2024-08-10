from env.robot_revolute_env import RobotRevoluteOpening 
import open3d as o3d
import robosuite as suite
from env.robot_drawer_opening import RobotDrawerOpening
from env.drawer_opening import DrawerOpeningEnv
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


object_name = "temp_door" # A microwave called "drawer"!!!

controller_name = "OSC_POSE"
controller_cfg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"controller_configs")
controller_cfg_path = os.path.join(controller_cfg_dir, "osc_pose_absolute.json")
controller_configs = suite.load_controller_config(custom_fpath = controller_cfg_path,default_controller=controller_name)

with open(controller_cfg_path, 'r') as f:
    controller_configs = json.load(f)
# print(controller_configs)

env_kwargs = dict(
    robots="Panda",
    object_type = "microwave",
    obj_rotation=(np.pi*2/3, np.pi*2/3),
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
    x_range = (-1,-1),
    y_range = (0,0),
)
env_name = "RobotRevoluteOpening"

env:RobotDrawerOpening = suite.make(
    env_name,
    **env_kwargs
)

obs = env.reset()

sim_utils.init_camera_pose(env, camera_pos=np.array([-0.77542536, -0.02539806,  0.30146208]), scale_factor=3) 

pcd_wf_path, camera_info_path = get_aograsp_ply_and_config(env_name = env_name, 
                        env_kwargs=env_kwargs,
                        object_name=env_kwargs["object_type"], 
                        camera_pos=np.array([-0.77542536, -0.02539806,  0.30146208]),
                        camera_quat=np.array([-0.005696068282031459, 0.19181093598117868, 0.02913152799094475, 0.9809829120433564]),
                        scale_factor=3,
                        pcd_wf_path='point_clouds/world_frame_pointclouds/world_frame_temp_door.ply',
                        viz=True
                        )

print(pcd_wf_path)
print(camera_info_path)

get_affordance_main(pcd_wf_path, camera_info_path)

# print(type(obs["gripper_quat"]))
# env.render()
camera_name = "frontview"
depth_image = obs['{}_depth'.format(camera_name)]
depth_image = flip_image(depth_image)

proposals = np.load(f'outputs/grasp_proposals/world_frame_proposals/world_frame_{object_name}_grasp.npz', allow_pickle=True)
print(proposals.keys())

proposal_pos = proposals['pos'] # np.array(np.array(3), ...)
proposal_quat = proposals['quat'] # np.array(np.array(4), ...)
proposal_scores = proposals['scores'] # np.array(np.float32, ...)

print(type(proposal_pos[0])) 
print(type(proposal_scores[0])) 

for i in range(30000):
    action = np.zeros_like(env.action_spec[0])
    env.step(action)
    env.render()

hand_pose = np.array([-0.2030102014541626, -0.540092134475708, 1.2524129152297974]) 
# Robosuite have an x pointing out of screen to you and an Z facing upward 
# rotation_euler = np.array([0.005762052722275257, -0.984636664390564,0.17452049255371094])
rotation_euler = np.array([0.005762052722275257, -0.984636664390564,0.17452049255371094])
# rotation_euler = np.array([0,0,0])
rotation_matrix = transform.euler2mat(rotation_euler)

rotation_matrix = np.array([
    [-0.0073337797075510025, -0.5499632358551025,0.8351566791534424],
    [0.0029766582883894444, -0.8351874351501465,-0.5499573349952698],          
    [0.9999686479568481, -0.001547289895825088,0.00776213314384222]
    ])
rotation_matrix = np.array([[-0.10221315920352936, -0.9264970421791077,-0.3621542751789093],
                            [-0.3834446370601654, 0.37262317538261414,-0.845057487487793],
                            [0.9178903102874756, 0.05249011516571045,-0.3933473229408264]])
# rotation_matrix = np.transpose(rotation_matrix)

sci_rotation = R.from_matrix(rotation_matrix)
rotation_axis_angle = sci_rotation.as_rotvec()

# target_quat = transform.mat2quat(rotation_matrix)
# obs = env.reset()
# env.render()
# obs = initialize_robot_position(env, hand_pose)
target_euler = transform.mat2euler(rotation_matrix)

state_dim = 7
action_dim = 3
gamma = 0.99 
lr_actor = 0.0003 
lr_critic = 0.0001 
K_epochs = 4 
eps_clip = 0.2 
action_std = 0.5 
has_continuous_action_space = True 
env.close()


