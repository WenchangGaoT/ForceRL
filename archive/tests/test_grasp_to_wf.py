import open3d as o3d
import numpy as np
import os
from scipy.spatial.transform import Rotation as R 

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

def create_arrow(cylinder_radius=0.01, cylinder_height=0.1, cone_radius=0.02, cone_height=0.05):
    # Create cylinder for the shaft of the arrow
    mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=cylinder_radius, height=cylinder_height)
    mesh_cylinder.compute_vertex_normals()
    mesh_cylinder.paint_uniform_color([1, 0, 0])  # Red color for the shaft

    # Create cone for the head of the arrow
    mesh_cone = o3d.geometry.TriangleMesh.create_cone(radius=cone_radius, height=cone_height)
    mesh_cone.compute_vertex_normals()
    mesh_cone.paint_uniform_color([0, 1, 0])  # Green color for the head

    # Translate the cone to the top of the cylinder
    mesh_cone.translate([0, 0, cylinder_height])

    # Combine cylinder and cone into a single arrow mesh
    mesh_arrow = mesh_cylinder + mesh_cone
    return mesh_arrow

def transform_arrow(arrow, position, quaternion):
    # Create transformation matrix from position and quaternion
    rotation_matrix = R.from_quat(quaternion).as_matrix()
    transformation = np.eye(4)
    transformation[:3, :3] = rotation_matrix
    transformation[:3, 3] = position

    # Apply transformation to the arrow mesh
    arrow.transform(transformation)
    return arrow

# Create an arrow
arrow = create_arrow()

# Define the position (x, y, z) and orientation (quaternion)
position = np.array([1.0, 1.0, 1.0])
quaternion = np.array([0, 0, 0, 1])  # Identity quaternion for no rotation

# Transform the arrow to the specified position and orientation
transformed_arrow = transform_arrow(arrow, position, quaternion)

# Create a point cloud for visualization
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))  # Example random points 

# Visualize the point cloud with the arrow
o3d.visualization.draw_geometries([pcd, transformed_arrow])
