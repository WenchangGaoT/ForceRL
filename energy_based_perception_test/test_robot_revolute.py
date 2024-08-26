from env.robot_revolute_env import RobotRevoluteOpening 
import open3d as o3d
import robosuite as suite
from env.robot_revolute_env import RobotRevoluteOpening
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
from agent.td3 import TD3
from state_inference import StateInference
import termcolor



controller_name = "OSC_POSE"
controller_cfg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"controller_configs")
controller_cfg_path = os.path.join(controller_cfg_dir, "osc_pose_absolute_small_kp.json")
controller_configs = suite.load_controller_config(custom_fpath = controller_cfg_path,default_controller=controller_name)

with open(controller_cfg_path, 'r') as f:
    controller_configs = json.load(f)
# print(controller_configs)


env_kwargs = dict(
    robots="Panda",
    # robots="Kinova3",
    # object_type = "dishwasher",
    # object_name = "dishwasher-3",
    object_type = "microwave",
    object_name = "microwave-3",
    scale_object = True,
    object_scale = 0.3,
    obj_rotation=(-np.pi/2, -np.pi/2),
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
    x_range = (0.8,0.8),
    y_range = (0.5, 0.5),
    # y_range = (-2, -2),
)
env_name = "RobotRevoluteOpening"

env:RobotRevoluteOpening = suite.make(
    env_name,
    **env_kwargs
)

obs = env.reset()

reset_joint_qpos = np.pi/6
# # set the joint qpos of the microwave
env.sim.data.qpos[env.slider_qpos_addr] = np.pi/6
env.sim.forward()

robot_gripper_pos = obs["robot0_eef_pos"]

sim_utils.init_camera_pose(env, camera_pos=np.array([-0.77542536, -0.02539806,  0.30146208]), scale_factor=3) 



viz_imgs = False
need_o3d_viz = False
run_cgn = False
use_env_state = True

pcd_wf_path = f'point_clouds/world_frame_pointclouds/world_frame_{env_kwargs["object_name"]}.ply'
pcd_wf_no_downsample_path = f'point_clouds/world_frame_pointclouds/world_frame_{env_kwargs["object_name"]}_no_downsample.ply'
camera_info_path = f'infos/camera_info_{env_kwargs["object_name"]}.npz'


camera_euler = np.array([ 45., 22., 3.])
camera_euler_for_pos = np.array([camera_euler[-1], camera_euler[1], -camera_euler[0]])
camera_rotation = R.from_euler("xyz", camera_euler_for_pos, degrees=True)
camera_quat = R.from_euler("xyz", camera_euler, degrees=True).as_quat()
print("camera quat: ", camera_quat)
camera_forward = np.array([1, 0, 0])
world_forward = camera_rotation.apply(camera_forward)
distance = 1.5
camera_pos = -distance * world_forward

reset_x_range = (1,1)
reset_y_range = (1,1)

pcd_wf_path, pcd_wf_no_downsample_path,camera_info_path = get_aograsp_ply_and_config(env_name = env_name, 
                        env_kwargs=env_kwargs,
                        object_name=env_kwargs["object_name"], 
                        # camera_pos=np.array([-0.77542536, -0.02539806,  0.30146208]),
                        camera_pos=camera_pos,
                        camera_quat=camera_quat,
                        scale_factor=1,
                        pcd_wf_path=pcd_wf_path,
                        pcd_wf_no_downsample_path=pcd_wf_no_downsample_path,
                        camera_info_path=camera_info_path,
                        viz=viz_imgs, 
                        need_o3d_viz=need_o3d_viz,
                        reset_joint_qpos=reset_joint_qpos,
                        reset_x_range = reset_x_range, 
                        reset_y_range = reset_y_range,
                        )

print(pcd_wf_path)
print(camera_info_path)

# get the affordance heatmap
pcd_cf_path, affordance_path = get_affordance_main(pcd_wf_path, camera_info_path, 
                                                   viz=False)

# get the grasp proposals
world_frame_proposal_path, top_k_pos_wf, top_k_quat_wf = get_grasp_proposals_main(
    pcd_cf_path, 
    affordance_path, 
    camera_info_path, 
    run_cgn=run_cgn, 
    viz=need_o3d_viz, 
    save_wf_pointcloud=True,
    object_name=env_kwargs["object_name"],
    top_k=-1,
)



obs_dim = 6
action_dim = 3
gamma = 0.99 
lr = 0.001
lr_critic = 0.0001 
K_epochs = 4 
eps_clip = 0.2 
action_std = 0.5 
has_continuous_action_space = True
max_action = float(5)


# load the trained policy
policy_dir = 'checkpoints/force_policies'
policy_name = 'curriculum_door_continuous_random_point_td3_0_curriculum_8'
policy = TD3(lr, obs_dim, action_dim, max_action)
policy.load(policy_dir, policy_name)


# world frame proposals need to be offset by the object's position
print("object pos: ", env.obj_pos)
object_pos = np.array(env.obj_pos)

# scale the proposals
print("raw grasp pose: ", top_k_pos_wf[0])
# top_k_pos_wf = np.array(top_k_pos_wf) * 3

print("scaled grasp pose: ", top_k_pos_wf[0])

top_k_pos_wf = top_k_pos_wf + object_pos


grasp_pos = top_k_pos_wf[0]
grasp_quat = top_k_quat_wf[0]

print("Grasp Pos: ", grasp_pos)

# change quat to euler
sci_rotation = R.from_quat(grasp_quat)
further_rotation = R.from_euler('z', 90, degrees=True)
sci_rotation = sci_rotation * further_rotation
rotation_vector = sci_rotation.as_rotvec()

mid_point_pos = (grasp_pos + robot_gripper_pos) / 2
prepaer_grasp_pos = grasp_pos + np.array([0., 0, 0.1])

action = np.concatenate([robot_gripper_pos, rotation_vector, [-1]])

# if state file exists, load it
object_name = env_kwargs["object_name"]
object_rotation = env_kwargs["obj_rotation"][0]
env_state_file_name  = f"revolute_{object_name}_{object_rotation}.npz"
file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "env_states")

if use_env_state and os.path.exists(os.path.join(file_dir, env_state_file_name)):
    state = np.load(os.path.join(file_dir, env_state_file_name))["state"]
    env.sim.set_state_from_flattened(state)
    env.sim.forward()

    final_grasp_pos = grasp_pos + np.array([0, 0, -0.05])
    obs,_,_,_ = env.step(np.concatenate([final_grasp_pos, rotation_vector, [1]]))

else:
    for i in range(80):
        action = np.concatenate([prepaer_grasp_pos, rotation_vector, [-1]])
        obs,_,_,_ = env.step(action)
        # print("gripper qpos", obs['robot0_gripper_qpos'])
        env.render()


    final_grasp_pos = grasp_pos + np.array([0, 0, -0.05])
    # final_grasp_pos = grasp_pos

    for i in range(50):
        action = np.concatenate([final_grasp_pos, rotation_vector, [-1]])
        obs,_,_,_ = env.step(action)
        env.render()

    # close the gripper
    for i in range(20):
        action = np.concatenate([final_grasp_pos, rotation_vector, [1]])
        obs,_,_,_ = env.step(action)
        env.render()

    # save the state of the environment
    object_name = env_kwargs["object_name"]
    object_rotation = env_kwargs["obj_rotation"][0]
    env_state_file_name  = f"revolute_{object_name}_{object_rotation}.npz"
    file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "env_states")

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    # check if the file exists
    if not os.path.exists(os.path.join(file_dir, env_state_file_name)):
        # get the env state
        state = env.sim.get_state().flatten()
        np.savez(os.path.join(file_dir, env_state_file_name), state=state)


last_grasp_pos = obs['robot0_eef_pos']

# pull the door, direction is towards the robot
robot_xpos = env.robot_base_xpos
action_direction = robot_xpos - last_grasp_pos
action_direction[2] = 0
action_direction = action_direction / np.linalg.norm(action_direction)


interaction_action_sequence = []
eef_pose_sequence = []


for i in range(80):
    action_direction = robot_xpos - last_grasp_pos
    # action_direction = [final_grasp_pos[0],final_grasp_pos[1],1]
    # action_direction = np.array([10,final_grasp_pos[1],final_grasp_pos[2]])
    action_direction = action_direction / np.linalg.norm(action_direction)
    if i <=5:
        action = action_direction * 0.01
    else: action = action_direction * 0.2
    print("action: ", action)
    action = np.concatenate([last_grasp_pos + action, rotation_vector, [1]])
    next_obs,_,_,_=env.step(action)
    env.render()
    last_grasp_pos = next_obs['robot0_eef_pos']
    # add the action to the sequence if the gripper is not fully closed
    if next_obs['robot0_gripper_qpos'][0] > 0.002:
        interaction_action_sequence.append(action)
        eef_pose_sequence.append(obs['robot0_eef_pos'])
    obs = next_obs

# print("interaction action sequence: ", interaction_action_sequence)
print("eef pose sequence: ", eef_pose_sequence)

eef_pose_sequence = np.array(eef_pose_sequence)
# calculate eef pose change each step
delta_eef_pose = eef_pose_sequence[1:] - eef_pose_sequence[:-1]

critic = policy.critic_2_target

axis_position_bonds = np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
# bounds relative to grasp position
axis_position_bonds = axis_position_bonds + final_grasp_pos
print("Axis Position Bounds: ", axis_position_bonds)

state_inference = StateInference(critic, axis_position_bonds, num_samples=4096,
                                 device="cuda")

# infer the state
state = state_inference.infer_state_from_critic(eef_pose_sequence)
print("Inferred State: ", state)
# get the ground truth state
print("ground truth hing position: ", env.hinge_position)
hinge_direction = np.array([0,0,1])
ground_truth_axis = np.concatenate([env.hinge_position, hinge_direction])
gt_axis_value= state_inference.get_ground_truth_state_value(eef_pose_sequence,
                                                            ground_truth_axis)
print("Ground Truth Axis Value: ", gt_axis_value)




