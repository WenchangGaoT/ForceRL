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
 

def scale_to_0_1(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data)) 

def get_o3d_pts(pts, normals=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd

def get_eef_line_set_for_o3d_viz(eef_pos_list, eef_quat_list, highlight_top_k=None):
    # Get base gripper points
    g_opening = 0.07
    gripper = mesh_utils.create_gripper("panda")
    gripper_control_points = gripper.get_control_point_tensor(
        1, False, convex_hull=False
    ).squeeze()
    mid_point = 0.5 * (gripper_control_points[1, :] + gripper_control_points[2, :])
    grasp_line_plot = np.array(
        [
            np.zeros((3,)),
            mid_point,
            gripper_control_points[1],
            gripper_control_points[3],
            gripper_control_points[1],
            gripper_control_points[2],
            gripper_control_points[4],
        ]
    )
    gripper_control_points_base = grasp_line_plot.copy()
    gripper_control_points_base[2:, 0] = np.sign(grasp_line_plot[2:, 0]) * g_opening / 2
    # Need to rotate base points, our gripper frame is different
    # ContactGraspNet
    r = Rotation.from_euler("z", 90, degrees=True)
    gripper_control_points_base = r.apply(gripper_control_points_base)

    # Compute gripper viz pts based on eef_pos and eef_quat
    line_set_list = []
    for i in range(len(eef_pos_list)):
        eef_pos = eef_pos_list[i]
        eef_quat = eef_quat_list[i]

        gripper_control_points = gripper_control_points_base.copy()
        g = np.zeros((4, 4))
        rot = Rotation.from_quat(eef_quat).as_matrix()
        g[:3, :3] = rot
        g[:3, 3] = eef_pos.T
        g[3, 3] = 1
        z = gripper_control_points[-1, -1]
        gripper_control_points[:, -1] -= z
        gripper_control_points[[1], -1] -= 0.02
        pts = np.matmul(gripper_control_points, g[:3, :3].T)
        pts += np.expand_dims(g[:3, 3], 0)

        lines = [[0, 1], [2, 3], [1, 4], [1, 5], [5, 6]]
        if highlight_top_k is not None:
            if i < highlight_top_k:
                # Draw grasp in green
                colors = [[0,1,0] for i in range(len(lines))]
            else:
                colors = [[0,0,0] for i in range(len(lines))]
        else:
            colors = [[0,0,0] for i in range(len(lines))]

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        line_set_list.append(line_set)

    return line_set_list

def viz_pts_and_eef_o3d(
    pts_pcd,
    eef_pos_list,
    eef_quat_list,
    heatmap_labels=None,
    save_path=None,
    frame="world",
    draw_frame=False,
    cam_frame_x_front=False,
    highlight_top_k=None,
    pcd_rgb=None
):
    """
    Plot eef in o3d visualization, with point cloud, at positions and
    orientations specified in eef_pos_list and eef_quat_list
    pts_pcd, eef_pos_list, and eef_quat_list need to be in same frame
    """
    print('o3ding...')
    pcd = get_o3d_pts(pts_pcd)
    if pcd_rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(pcd_rgb)
    else:
        if heatmap_labels is not None:
            # Scale heatmap for visualization
            heatmap_labels = scale_to_0_1(heatmap_labels)

            cmap = matplotlib.cm.get_cmap("RdYlGn")
            colors = cmap(np.squeeze(heatmap_labels))[:, :3]
            pcd.colors = o3d.utility.Vector3dVector(colors)

    # Get line_set for drawing eef in o3d
    line_set_list = get_eef_line_set_for_o3d_viz(
        eef_pos_list, eef_quat_list, highlight_top_k=highlight_top_k,
    )

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.add_geometry(pcd)

    for line_set in line_set_list:
        vis.add_geometry(line_set)

    # Draw ref frame
    if draw_frame:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        vis.add_geometry(mesh_frame)

    # Move camera
    if frame == "camera":
        # If visualizing in camera frame, view pcd from scene view
        ctr = vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        fov = ctr.get_field_of_view()
        H = np.eye(4)
        if cam_frame_x_front:
            R = Rotation.from_euler("XYZ", [90, 0, 90], degrees=True).as_matrix()
            H[:3, :3] = R
        param.extrinsic = H
        ctr.convert_from_pinhole_camera_parameters(param)
    else:
        # If world frame, place camera accordingly to face object front
        ctr = vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        fov = ctr.get_field_of_view()
        H = np.eye(4)
        H[2, -1] = 1
        R = Rotation.from_euler("XYZ", [90, 0, 90], degrees=True).as_matrix()
        H[:3, :3] = R
        param.extrinsic = H
        ctr.convert_from_pinhole_camera_parameters(param)

    vis.poll_events()
    vis.update_renderer()

    if save_path is None:
        vis.run()
    else:
        vis.capture_screen_image(
            save_path,
            do_render=True,
        )
    vis.destroy_window()


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

def get_quat_from_zfront_to_wf(pts_quat_cf, camera_info_path): 
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

def recover_pts_from_cam_to_wf(pts_cf_path, camera_info_path):
    pts_cf_dict = np.load(pts_cf_path, allow_pickle=True) 
    # print(pts_cf_dict)
    # pts_cf_arr = pts_cf_dict['pts_arr'] + pts_cf_dict['mean']
    pts_cf_arr = pts_cf_dict['pts_arr']
    pts_wf_arr = get_pts_from_zfront_to_wf(pts_cf_arr, camera_info_path) 
    return pts_wf_arr 

def convert_proposal_to_wf(proposal_path, camera_info_path, pts_cf_info_path): 
    proposal_cf_dict = np.load(proposal_path, allow_pickle=True)['data'].item()  
    pts_cf_info_dict = np.load(pts_cf_info_path, allow_pickle=True) 
    # camera_info_dict = np.load(camera_info_path)
    # print(proposal_cf_dict) 
    proposals = proposal_cf_dict['proposals'] 
    # print()
    cgn_grasps = proposal_cf_dict['cgn_grasps'] 
    proposal_points_cf = np.array([p[0] for p in proposals]) 
    proposal_quats_cf = np.array([p[1] for p in proposals]) 
    proposal_scores = np.array([p[2] for p in proposals]) 
    
    proposal_points_cf += pts_cf_info_dict['mean']
    proposal_points_wf = get_pts_from_zfront_to_wf(proposal_points_cf, camera_info_path) 
    proposal_quats_wf = get_quat_from_zfront_to_wf(proposal_quats_cf, camera_info_path) 

    return proposal_points_wf, proposal_quats_wf, proposal_scores, cgn_grasps

    # proposal_ori_cf = np.array([quat2mat(p[1]) for p in proposals]) 
    # print(proposal_ori_cf)
    # pass

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
    viz_pts_and_eef_o3d(
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