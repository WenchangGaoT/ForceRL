'''
Generate the point cloud and camera config from a single camera in the system that AO-grasp supports. 
'''

import torch 
import argparse
import os
import pickle

import robosuite as suite
import open3d as o3d 
import numpy as np
import aograsp.model_utils as m_utils 
import aograsp.viz_utils as v_utils 
import aograsp.data_utils as d_utils 
import aograsp.rotation_utils as r_utils 
import matplotlib.pyplot as plt

from utils.sim_utils import get_pointcloud 
from env.curri_door_env import CurriculumDoorEnv
from env.robot_revolute_env import RobotRevoluteOpening 
from env.robot_drawer_opening import RobotDrawerOpening 
from env.original_door_env import OriginalDoorEnv 
from env.camera_door_env import CameraDoorEnv
from scipy.spatial.transform import Rotation as R 
from robosuite.utils.transform_utils import *

def set_camera_pose(env, camera_name, position, quaternion):
    sim = env.sim
    cam_id = sim.model.camera_name2id(camera_name)
    sim.model.cam_pos[cam_id] = position
    sim.model.cam_quat[cam_id] = quaternion
    sim.forward() 

def display_camera_pose(env, camera_name):
    cam_id = env.sim.model.camera_name2id(camera_name)
    print(f'{camera_name} pose: {env.sim.model.cam_pos[cam_id]}') 
    print(f'{camera_name} quat: {env.sim.model.cam_quat[cam_id]}') 

def get_aograsp_ply_and_config(environment_name, object_name, camera_pos, camera_quat, scale_factor=3, cfg_path='temp.npz', pcd_path='temp.ply', device='cuda:0', denoise=False):

    # aograsp_model = m_utils.load_model(
    #     model_conf_path='models/grasp_models/ao-grasp/conf.pth', 
    #     ckpt_path='models/grasp_models/ao-grasp/770-network.pth'
    # ) 
    # aograsp_model.to(torch.device(device))

    controller_name = "OSC_POSE"
    controller_configs = suite.load_controller_config(default_controller=controller_name)
    env = suite.make(
        environment_name,
        robots="Panda",
        has_renderer=True,
        use_camera_obs=True,
        has_offscreen_renderer=True,
        camera_depths = True,
        camera_segmentations = "element",
        controller_configs=controller_configs,
        control_freq = 20,
        horizon=10000,
        camera_names = ['sideview'], 
        camera_heights = 256, 
        camera_widths = 256, 
        obj_rotation=(np.pi/6, np.pi/6)
    )

    obs = env.reset() 
    print('rotation matrix for [-0.5, -0.5, 0.5, 0.5]: ') 
    m1 = quat2mat(np.array([-0.5, -0.5, 0.5, 0.5])) # Camera local frame to world frame front, set camera fram
    print(m1)

    # obj_quat = env.obj_quat
    # obj_quat = convert_quat(obj_quat, to='xyzw')
    # m3 = quat2mat(obj_quat)# Turn camera and microwave simultaneously

    # print('rotation matrix for quat: ') 
    # m2 = quat2mat(np.array(camera_quat)) # Turn camera to microwave
    # print(m2)

    obj_quat = env.obj_quat 
    obj_quat = convert_quat(obj_quat, to='xyzw')
    rotation_mat_world = quat2mat(obj_quat)
    rotation_euler_world = mat2euler(rotation_mat_world)
    rotation_euler_cam = np.array([rotation_euler_world[2], 0,0])
    m3_world = quat2mat(obj_quat)
    # obj_quat = np.array([0.383, 0, 0, 0.924])

    m3 = euler2mat(rotation_euler_cam)# Turn camera and microwave simultaneously
    # m3 = np.eye(3)

    print('rotation matrix for quat: ') 
    m2 = quat2mat(np.array([-0.005696068282031459, 0.19181093598117868, 0.02913152799094475, 0.9809829120433564])) # Turn camera to microwave
    print(m2)
    print("m3: ", m3)
    M = np.dot(m1,m2)
    M = np.dot(M, m3.T) 
    quat = R.from_matrix(M).as_quat() 
    print('Corresponding quaternion: ', quat)

    obj_pos = env.obj_pos 
    camera_pos = np.array(camera_pos)
    camera_trans = scale_factor*camera_pos 
    camera_trans = np.dot(m3_world, camera_trans) 

    set_camera_pose(env, 'sideview', obj_pos + camera_trans, quat) 
    display_camera_pose(env, 'sideview') 
    low, high = env.action_spec
    obs, reward, done, _ = env.step(np.zeros_like(low))
    plt.imshow(obs['sideview_image']) 
    plt.show()
    plt.imshow(obs['sideview_depth'], cmap='gray')
    plt.show() 
    pointcloud = get_pointcloud(env, obs, ['sideview'], [256], [256], [object_name]) 
    print(f'Before denoising: {len(pointcloud.points)}')
    if denoise:
        pointcloud, ind = pointcloud.remove_statistical_outlier(nb_neighbors=200, std_ratio=2.0)
    print(f'After denoising: {len(pointcloud.points)}') 
    pointcloud = pointcloud.farthest_point_down_sample(4096) 
    obs_arr = np.asarray(pointcloud.points) - np.array(env.obj_pos)
    obs_arr = obs_arr / scale_factor

    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(obs_arr)
    pts_arr = np.array(pointcloud.points)

    o3d.io.write_point_cloud(pcd_path, pointcloud) 
    print(f'point cloud saved to {pcd_path}') 

    pts_npz = {
        'data': {
            'pts': pts_arr, 
            'obj_pos': np.array(env.obj_pos), 
            'obj_quat': np.array(env.obj_quat)
            }
        }
    
    np.save(f"{pcd_path}.npz", pts_npz) 
    with open(f'{pcd_path}.npz', 'wb') as f:
        pickle.dump(pts_npz, f)
    print(f'point cloud npz saved to {pcd_path}.npz')

    camera_config = {'data': {
                        'camera_config': {
                            # 'trans': camera_pos*scale_factor, 
                            'trans': camera_pos, 
                            'quat': camera_quat
                        }
                    }
                } 
    with open(cfg_path, 'wb') as f:
        pickle.dump(camera_config, f)
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--environment_name', type=str, default='RobotRevoluteOpening') 
    parser.add_argument('--object_name', type=str, default='drawer')
    parser.add_argument('--camera_pos', default=[-0.77542536, -0.02539806,  0.30146208])
    parser.add_argument('--camera_quat', default=[-0.005696068282031459, 0.19181093598117868, 0.02913152799094475, 0.9809829120433564])
    # parser.add_argument('--output_dir', type=str, default='outputs/')
    parser.add_argument('--scale_factor', default=3)
    parser.add_argument('--pcd_path', type=str, default='point_clouds/temp_door.ply') 
    parser.add_argument('--cfg_path', type=str, default='temp_door_camera.npz')
    parser.add_argument('--device', type=str, default='cuda:0') 
    parser.add_argument('--denoise', default=True)
    args = parser.parse_args()

    get_aograsp_ply_and_config(args.environment_name, args.object_name, args.camera_pos, 
                           args.camera_quat, args.scale_factor, args.cfg_path, args.pcd_path, denoise=args.denoise)

