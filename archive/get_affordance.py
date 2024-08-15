
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