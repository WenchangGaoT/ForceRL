from gamma.visual_model.object_articulation_part import gamma_model_net 
import utils.gamma_utils.data_utilts as g_d_utils
import os 
import argparse 
import torch 
import numpy as np 
import robosuite as suite 
from env.robot_revolute_env import RobotRevoluteOpening
from robosuite.utils.transform_utils import * 
from scipy.spatial.transform import Rotation as R 
import robosuite.utils.camera_utils as robo_cam_utils
import matplotlib.pyplot as plt 
from utils.sim_utils import get_pointcloud 
import open3d as o3d

parser = argparse.ArgumentParser() 
parser.add_argument('--pcd_path', type=str, default='point_clouds/temp_door.ply') 
parser.add_argument('--camera_pos', default=[-0.77542536, -0.02539806,  0.30146208]) 
parser.add_argument('--scale_factor', type=float, default=1)
args = parser.parse_args()

def set_camera_pose(env, camera_name, position, quaternion):
    sim = env.sim
    cam_id = sim.model.camera_name2id(camera_name)
    sim.model.cam_pos[cam_id] = position
    sim.model.cam_quat[cam_id] = quaternion
    sim.forward()

model = gamma_model_net(in_channel=3, num_point=4096, num_classes=3, device='cuda:0').to('cuda:0')
model.load_state_dict(torch.load('/home/wgao22/projects/ForceRL/checkpoints/perception_models/gamma_best.pth')) 

controller_name = "OSC_POSE"
controller_configs = suite.load_controller_config(default_controller=controller_name)
env = suite.make(
    'RobotRevoluteOpening',
    robots="Panda",
    has_renderer=True,
    use_camera_obs=True,
    has_offscreen_renderer=True,
    camera_depths = True,
    camera_segmentations = "element",
    controller_configs=controller_configs,
    control_freq = 20,
    horizon=10000,
    camera_names = ['frontview', 'sideview'], 
    camera_heights = [448, 448], 
    camera_widths = [448, 448], 
    # obj_rotation=(-np.pi/2, -np.pi/2),
    obj_rotation=(0, 0),
    x_range = (0.8,0.8),
    y_range = (0.4, 0.4),
    object_type = "microwave",
) 

obs = env.reset() 
env.sim.data.qpos[env.slider_qpos_addr] = np.pi / 10
# env.slider_qpos_addr = 0.5
enc=env.sim.forward()
print('rotation matrix for [-0.5, -0.5, 0.5, 0.5]: ') 
m1 = quat2mat(np.array([-0.5, -0.5, 0.5, 0.5])) # Camera local frame to world frame front, set camera fram

obj_quat = env.obj_quat 
obj_quat = convert_quat(obj_quat, to='xyzw')
rotation_mat_world = quat2mat(obj_quat)
rotation_euler_world = mat2euler(rotation_mat_world)
rotation_euler_cam = np.array([rotation_euler_world[2], 0,0])
m3_world = quat2mat(obj_quat)
# obj_quat = np.array([0.383, 0, 0, 0.924])

m3 = euler2mat(rotation_euler_cam)# Turn camera and microwave simultaneously
# m3 = np.eye(3)
 
m2 = quat2mat(np.array([-0.005696068282031459, 0.19181093598117868, 0.02913152799094475, 0.9809829120433564])) # Turn camera to microwave

M = np.dot(m1,m2)
M = np.dot(M, m3.T) 
quat = R.from_matrix(M).as_quat() 
print('Corresponding quaternion: ', quat)

obj_pos = env.obj_pos 
camera_pos = np.array(args.camera_pos)
camera_trans = args.scale_factor*camera_pos 
camera_trans = np.dot(m3_world, camera_trans) 

set_camera_pose(env, 'sideview', obj_pos + camera_trans, quat) 

extrinsic = robo_cam_utils.get_camera_extrinsic_matrix(env.sim, 'sideview')
# extrinsic[:3,:3] = M.T

print('quaternion rotation matrix: \n', M)
print('exported extrinsic: \n', extrinsic)

low, high = env.action_spec
obs, reward, done, _ = env.step(np.zeros_like(low))
plt.imshow(obs['sideview_image']) 
plt.show()
plt.imshow(obs['sideview_depth'], cmap='gray')
plt.show() 
pointcloud = get_pointcloud(env, obs, ['sideview'], [448], [448], ['microwave'])  
# pointcloud = pointcloud.farthest_point_down_sample(55000)

o3d.io.write_point_cloud('point_clouds/microwave_open_wf.ply', pointcloud)
print('pointcloud object: ', pointcloud)

camera_pcd_arr = np.array(pointcloud.points).astype(np.float32) 
camera_pcd_arr = g_d_utils.translate_pc_world_to_camera(camera_pcd_arr, extrinsic)
print(camera_pcd_arr.dtype)
env.close()

with torch.no_grad():
    model.eval() 
    results, labels, camera_pcd = model.online_inference(camera_pcd=camera_pcd_arr, view_res=True, ignore_label=2)