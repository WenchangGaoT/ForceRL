"""
Run pointscore inference on partial pointcloud
"""

import os
from argparse import ArgumentParser
import numpy as np
import torch
import open3d as o3d

# from train_model.test import load_conf
import aograsp.viz_utils as v_utils
import aograsp.model_utils as m_utils 
import aograsp.data_utils.dataset_utils as d_utils 
import aograsp.rotation_utils as r_utils


def get_aograsp_pts_in_cam_frame_z_front_with_info(pts_wf, info_path):
    """
    Given a path to a partial point cloud render/000*/point_cloud_seg.npz 
    and a path to the corresponding infomation npz.
    from the AO-Grasp dataset, get points in camera frame with z-front, y-down
    """

    # Load pts in world frame
    # pcd_dict = np.load(pc_path, allow_pickle=True)["data"].item()
    # pts_wf = pcd_dict["pts"]

    # Load camera pose information
    # render_dir = os.path.dirname(pc_path)
    # info_path = os.path.join(render_dir, "info.npz")
    if not os.path.exists(info_path):
        raise ValueError(
            "info.npz cannot be found. Please ensure that you are passing in a path to a point_cloud_seg.npz file"
        )
    info_dict = np.load(info_path, allow_pickle=True)["data"] 
    if not isinstance(info_dict, dict):
        info_dict = info_dict.item()
    cam_pos = info_dict["camera_config"]["trans"]
    cam_quat = info_dict["camera_config"]["quat"]

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
    '''
    if not os.path.exists(camera_info_path):
        raise ValueError(
            "info.npz cannot be found. Please ensure that you are passing in a path to a point_cloud_seg.npz file"
        )
    info_dict = np.load(camera_info_path, allow_pickle=True)["data"] 
    if not isinstance(info_dict, dict):
        info_dict = info_dict.item()
    cam_pos = info_dict["camera_config"]["trans"]
    cam_quat = info_dict["camera_config"]["quat"] 

    # Transform points from world to camera frame
    H_cam2world_xfront = r_utils.get_H(r_utils.get_matrix_from_quat(cam_quat), cam_pos)
    H_world2cam_xfront = r_utils.get_H_inv(H_cam2world_xfront)
    H_world2cam_zfront = d_utils.get_H_world_to_cam_z_front_from_x_front(H_world2cam_xfront) 
    pts_cf_homo = np.concatenate([pts_cf_arr, np.ones_like(pts_cf_arr[:, :1])], axis=-1) 
    # pts_wf_arr = np.matmul(pts_cf_homo, np.linalg.inv(H_world2cam_zfront.T))[:, :3]
    pts_wf_arr = np.matmul(np.linalg.inv(H_world2cam_zfront),pts_cf_homo.T).T[:, :3]
    return pts_wf_arr 

def recover_pts_from_cam_to_wf(pts_cf_path, camera_info_path):
    pts_cf_dict = np.load(pts_cf_path, allow_pickle=True) 
    # print(pts_cf_dict)
    # pts_cf_arr = pts_cf_dict['pts_arr'] + pts_cf_dict['mean']
    pts_cf_arr = pts_cf_dict['pts_arr']
    pts_wf_arr = get_pts_from_zfront_to_wf(pts_cf_arr, camera_info_path) 
    return pts_wf_arr 

def convert_proposal_to_wf(proposal_path, camera_info_path, pts_cf_info_path):
    pass

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
    parser.add_argument('--pcd_info_path', type=str)
    parser.add_argument('--camera_info_path', type=str) 
    args = parser.parse_args() 
    # pts_cf = o3d.io.read_point_cloud(args.pcd_path) 
    # pts_cf_arr = np.array(pts_cf.points) 
    # pts_wf_arr = get_pts_from_zfront_to_wf(pts_cf_arr, args.info_path) 
    # pts_wf = o3d.geometry.PointCloud() 
    # pts_wf.points = o3d.utility.Vector3dVector(pts_wf_arr) 
    # o3d.io.write_point_cloud('point_clouds/world_frame_temp_door.ply', pts_wf) 
    pts_wf_arr = recover_pts_from_cam_to_wf(args.pcd_info_path, args.camera_info_path) 
    pts_wf = o3d.geometry.PointCloud() 
    pts_wf.points = o3d.utility.Vector3dVector(pts_wf_arr) 
    o3d.io.write_point_cloud('point_clouds/world_frame_temp_door.ply', pts_wf)
    # pass