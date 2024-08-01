import numpy as np
import open3d as o3d
import robosuite.utils.camera_utils as camera_utils
import os
import copy

def flip_image(image):
    return np.flip(image, 0)

def get_pointcloud(env, obs, camera_names, 
                   cam_heights, cam_widths, 
                   geom_name:list):
    """
    Create poincloud from depth image and segmentation image, can have multiple cameras
    """
    assert isinstance(camera_names, list) or isinstance(camera_names, str), "camera_names should be a list or a string"
    if isinstance(camera_names, str):
        camera_names = [camera_names]
        cam_heights = [cam_heights]
        cam_widths = [cam_widths]
    # get depth image, rgb image, segmentation image
    print(obs.keys())
    masked_pcd_list = []

    for camera_name in camera_names:
        cam_width = cam_widths[camera_names.index(camera_name)]
        cam_height = cam_heights[camera_names.index(camera_name)]
        seg_image = obs["{}_segmentation_element".format(camera_name)]
        depth_image = obs['{}_depth'.format(camera_name)]
        depth_image = flip_image(depth_image) 
        # depth_image = np.flip(depth_image, 2)
        depth_image = camera_utils.get_real_depth_map(env.sim, depth_image)
        
        # get geom id corresponding to geom_name
        # pc_geom_id = [env.sim.model.geom_name2id(pc_geo) for pc_geo in env.sim.model.geom_names if 'cube' in pc_geo or 'table' in pc_geo]
        pc_geom_id = [env.sim.model.geom_name2id(pc_geo) for pc_geo in env.sim.model.geom_names if any(name in pc_geo for name in geom_name)]

        # create mask using geom_id and segmentation image
        masked_segmentation = np.isin(seg_image, pc_geom_id)
        masked_segmentation = flip_image(masked_segmentation) 
        # masked_segmentation = np.flip(masked_segmentation, 2)

        # mask the depth image
        masked_depth = np.multiply(depth_image, masked_segmentation).astype(np.float32)

        # create open3d image object
        masked_depth_image = o3d.geometry.Image(masked_depth)

        # get camera intrinsic and extrinsic parameters
        intrinisc_cam_parameters_numpy = camera_utils.get_camera_intrinsic_matrix(env.sim, camera_name, cam_height, cam_width)
        extrinsic_cam_parameters= camera_utils.get_camera_extrinsic_matrix(env.sim, camera_name)

        cx = intrinisc_cam_parameters_numpy[0][2]
        cy = intrinisc_cam_parameters_numpy[1][2]
        fx = intrinisc_cam_parameters_numpy[0][0]
        fy = intrinisc_cam_parameters_numpy[1][1]

        intrinisc_cam_parameters = o3d.camera.PinholeCameraIntrinsic(cam_width, #width 
                                                            cam_height, #height
                                                            fx,
                                                            fy,
                                                            cx,
                                                            cy)
        
        # create open3d pointcloud object
        masked_pcd = o3d.geometry.PointCloud.create_from_depth_image(masked_depth_image,                                                       
                                                intrinisc_cam_parameters
                                                )
        masked_pcd.transform(extrinsic_cam_parameters)

        #estimate normals
        masked_pcd.estimate_normals()
        #orientation normals to camera
        masked_pcd.orient_normals_towards_camera_location(extrinsic_cam_parameters[:3,3])
        masked_pcd_list.append(copy.deepcopy(masked_pcd))
    
    # for i in range(len(masked_pcd_list) - 1):
    #     complete_masked_pcd = masked_pcd_list[i] + masked_pcd_list[i+1]
    complete_masked_pcd = masked_pcd_list[0]
    for i in range(1, len(masked_pcd_list)):
        complete_masked_pcd += masked_pcd_list[i]
    if len(masked_pcd_list) == 1:
        complete_masked_pcd = masked_pcd_list[0]
    return complete_masked_pcd

def initialize_robot_position(env, desired_position):
    obs, reward, done, _ = env.step(np.zeros(env.action_dim))
    # print(obs["gripper_pos"])
    while np.square(obs["gripper_pos"]*50 - desired_position*50).sum() > 0.01:
        action = -(obs["gripper_pos"] - desired_position)
        # normalize the action scale
        action = action / np.linalg.norm(action) * 0.5
        action = np.concatenate([action, np.zeros(4)])
        obs, reward, done, _ = env.step(action)
        # print(obs["gripper_pos"])
        env.render()
    return obs

def open_gripper(env):
    for i in range(5):
        obs, reward, done, _ = env.step(np.concatenate([np.zeros(env.action_dim - 1),[-1]]))
        env.render()
    return obs

def close_gripper(env):
    for i in range(15):
        obs, reward, done, _ = env.step(np.concatenate([np.zeros(env.action_dim - 1),[1]]))
        env.render()
    return obs 

# object utils
def fill_custom_xml_path(xml_path):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abs_xml_path = os.path.join(base_dir, "sim_models", xml_path)
    return abs_xml_path