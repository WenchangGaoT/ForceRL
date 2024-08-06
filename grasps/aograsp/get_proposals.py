"""
Run pointscore inference on partial pointcloud
"""

import os
from argparse import ArgumentParser
import numpy as np
# import torch
import open3d as o3d

# from train_model.test import load_conf
# import aograsp.viz_utils as v_utils
# import aograsp.model_utils as m_utils 
import utils.aograsp_utils.dataset_utils as d_utils 
import utils.aograsp_utils.rotation_utils as r_utils 
from scipy.spatial.transform import Rotation
# from robosuite.utils.transform_utils import * 
import matplotlib 
import utils.mesh_utils as mesh_utils 

import utils.cgn_utils.vis_utils as cgn_v_utils


def get_aograsp_pts_in_cam_frame_z_front_with_info(pts_wf, camera_info_path):
    """
    Given a path to a partial point cloud render/000*/point_cloud_seg.npz 
    and a path to the corresponding infomation npz.
    from the AO-Grasp dataset, get points in camera frame with z-front, y-down. 

    camera_info_path: 
        must be a path to a .npz file. The npz is supposed to be dumped from a dictionary with key 'data'.
            {
                'data': {
                    'camera_config': {
                        'trans': [x, y, z],
                        'quat': [x, y, z, w]
                    }
                }
            }
    """
    if not os.path.exists(camera_info_path):
        raise ValueError(
            "info.npz cannot be found. Please ensure that you are passing in a path to a point_cloud_seg.npz file"
        )
    camera_info_dict = np.load(camera_info_path, allow_pickle=True)["data"] 
    if not isinstance(camera_info_dict, dict):
        camera_info_dict = camera_info_dict.item()
    cam_pos = camera_info_dict["camera_config"]["trans"]
    cam_quat = camera_info_dict["camera_config"]["quat"]

    # Transform points from world to camera frame
    H_cam2world_xfront = r_utils.get_H(r_utils.get_matrix_from_quat(cam_quat), cam_pos)
    H_world2cam_xfront = r_utils.get_H_inv(H_cam2world_xfront)
    H_world2cam_zfront = d_utils.get_H_world_to_cam_z_front_from_x_front(H_world2cam_xfront)
    pts_cf = r_utils.transform_pts(pts_wf, H_world2cam_zfront)

    return pts_cf 

def get_pts_from_zfront_to_wf(pts_cf_arr, camera_info_path): 
    '''
    Get the world frame point of a cf point. 
    info_path: the path to the config of camera to calculate cf.
    pts_cf_arr:
        an numpy array of the point clouds in the camera frams.
    
    camera_info_path:
        must be a path to a .npz file. The npz is supposed to be dumped from a dictionary with key 'data'.
    '''
    if not os.path.exists(camera_info_path):
        raise ValueError(
            "info.npz cannot be found. Please ensure that you are passing in a path to a point_cloud_seg.npz file"
        )
    camera_info_dict = np.load(camera_info_path, allow_pickle=True)["data"] 
    if not isinstance(camera_info_dict, dict):
        camera_info_dict = camera_info_dict.item()
    cam_pos = camera_info_dict["camera_config"]["trans"]
    cam_quat = camera_info_dict["camera_config"]["quat"] 

    # Transform points from world to camera frame
    H_cam2world_xfront = r_utils.get_H(r_utils.get_matrix_from_quat(cam_quat), cam_pos)
    H_world2cam_xfront = r_utils.get_H_inv(H_cam2world_xfront)
    H_world2cam_zfront = d_utils.get_H_world_to_cam_z_front_from_x_front(H_world2cam_xfront) 
    pts_cf_homo = np.concatenate([pts_cf_arr, np.ones_like(pts_cf_arr[:, :1])], axis=-1) 
    # pts_wf_arr = np.matmul(pts_cf_homo, np.linalg.inv(H_world2cam_zfront.T))[:, :3]
    pts_wf_arr = np.matmul(np.linalg.inv(H_world2cam_zfront),pts_cf_homo.T).T[:, :3]
    return pts_wf_arr 

def get_quat_from_zfront_to_wf(pts_quat_cf, camera_info_path): 
    '''
    Get the world frame point of a cf point. 
    camera_info_path: the path to the config of camera to calculate cf. 
        {
            'data': {
                'camera_config': {
                    'trans': np.array(3,), 
                    'quat': np.array(4,)
                }
            }
        }
    '''
    if not os.path.exists(camera_info_path):
        raise ValueError(
            "info.npz cannot be found. Please ensure that you are passing in a path to a point_cloud_seg.npz file"
        )
    camera_info_dict = np.load(camera_info_path, allow_pickle=True)["data"] 
    if not isinstance(camera_info_dict, dict):
        camera_info_dict = camera_info_dict.item()
    cam_pos = camera_info_dict["camera_config"]["trans"]
    cam_quat = camera_info_dict["camera_config"]["quat"] 

    # Transform points from world to camera frame
    H_cam2world_xfront = r_utils.get_H(r_utils.get_matrix_from_quat(cam_quat), cam_pos)
    H_world2cam_xfront = r_utils.get_H_inv(H_cam2world_xfront)
    H_world2cam_zfront = d_utils.get_H_world_to_cam_z_front_from_x_front(H_world2cam_xfront) 
    H_cam2world_zfront = np.linalg.inv(H_world2cam_zfront)
    # get rotation part of H_cam2world_zfront
    R_rotation_cam2world_zfront = H_cam2world_zfront[:3, :3]
    
    # transform the quaternion from cf to wf
    # use rotation matrix to rotate the quaternion
    R_rotation_cam2world_zfront_mat = Rotation.from_matrix(R_rotation_cam2world_zfront)
    pts_rotation_cf = [Rotation.from_quat(q) for q in pts_quat_cf]
    pts_rotation_wf = [R_rotation_cam2world_zfront_mat * q for q in pts_rotation_cf]

    # back to quaternion
    pts_quat_wf = [q.as_quat() for q in pts_rotation_wf]

    return pts_quat_wf 

def recover_pts_from_cam_to_wf(pts_cf_info_path, camera_info_path): 
    '''
    Recover the camera frame point cloud to world frame point cloud. 
    pts_cf_path:
        the path to the camera frame point cloud information npz file dumped from a dictionary with pickle.
            {
                'pts_arr': np.array([...]),
                'mean': np.array([...])
            }
    '''
    pts_cf_info_dict = np.load(pts_cf_info_path, allow_pickle=True) 
    pts_cf_arr = pts_cf_info_dict['pts_arr']
    pts_wf_arr = get_pts_from_zfront_to_wf(pts_cf_arr, camera_info_path) 
    return pts_wf_arr 

def convert_proposal_to_wf(proposal_path, camera_info_path, pts_cf_info_path): 
    '''
    Convert the grasp proposals from normalized camera frame to world frame. 
    proposal_path:
        the path to the proposal npz file dumped from a dictionary with pickle.
            {
                'data': {
                    'proposals': [
                        (pts, quats, score),
                        ...
                    ],
                    'cgn_grasps': [
                        (pts, quats, score),
                        ...
                    ]
            }
    '''
    proposal_cf_dict = np.load(proposal_path, allow_pickle=True)['data'].item()  
    pts_cf_info_dict = np.load(pts_cf_info_path, allow_pickle=True) 
    proposals = proposal_cf_dict['proposals'] 
    cgn_grasps = proposal_cf_dict['cgn_grasps'] 
    proposal_points_cf = np.array([p[0] for p in proposals]) 
    proposal_quats_cf = np.array([p[1] for p in proposals]) 
    proposal_scores = np.array([p[2] for p in proposals]) 
    
    proposal_points_cf += pts_cf_info_dict['mean']
    proposal_points_wf = get_pts_from_zfront_to_wf(proposal_points_cf, camera_info_path) 
    proposal_quats_wf = get_quat_from_zfront_to_wf(proposal_quats_cf, camera_info_path) 

    return proposal_points_wf, proposal_quats_wf, proposal_scores, cgn_grasps

def read_grasp_proposals(proposals, camera_info, top_k): 
    '''
    read top K proposals and convert the proposals to world frame
    '''
    # Recover 
    pass 

def get_grasp_proposals(score_path): 
    '''
    get the grasp proposals and return the tuple list
    '''
    pass


if __name__ == '__main__': 
    parser = ArgumentParser() 
    # parser.add_argument('--pcd_path', type=str) 
    parser.add_argument('--pcd_cf_info_path', type=str)
    parser.add_argument('--camera_info_path', type=str)  
    parser.add_argument('--proposal_path', type=str) 
    parser.add_argument('--top_k', type=int, default=10)

    args = parser.parse_args()  
    pts_wf_arr = recover_pts_from_cam_to_wf(args.pcd_cf_info_path, args.camera_info_path)
    g_pos_wf, g_quat_wf, scores, cgn_gs = convert_proposal_to_wf(args.proposal_path, args.camera_info_path, args.pcd_cf_info_path) 
    eef_pos_list = [g for g in g_pos_wf] 
    eef_quat_list = [g for g in g_quat_wf] 
    sorted_grasp_tuples = [(g_pos_wf[i], g_quat_wf[i], scores[i]) for i in range(len(g_pos_wf))] 
    sorted_grasp_tuples.sort(key=lambda x: x[2], reverse=True) 
    eef_pos_list = [g[0] for g in sorted_grasp_tuples][:args.top_k]
    eef_quat_list = [g[1] for g in sorted_grasp_tuples][:args.top_k]
    cgn_v_utils.viz_pts_and_eef_o3d(
        pts_pcd=pts_wf_arr, 
        eef_pos_list=eef_pos_list, 
        eef_quat_list=eef_quat_list
    )
    # # pts_cf = o3d.io.read_point_cloud(args.pcd_path) 
    # # pts_cf_arr = np.array(pts_cf.points) 
    # # pts_wf_arr = get_pts_from_zfront_to_wf(pts_cf_arr, args.info_path) 
    # # pts_wf = o3d.geometry.PointCloud() 
    # # pts_wf.points = o3d.utility.Vector3dVector(pts_wf_arr) 
    # # o3d.io.write_point_cloud('point_clouds/world_frame_temp_door.ply', pts_wf) 
    # pts_wf_arr = recover_pts_from_cam_to_wf(args.pcd_info_path, args.camera_info_path) 
    # pts_wf = o3d.geometry.PointCloud() 
    # pts_wf.points = o3d.utility.Vector3dVector(pts_wf_arr) 
    # o3d.io.write_point_cloud('point_clouds/world_frame_temp_door.ply', pts_wf)
    # pass