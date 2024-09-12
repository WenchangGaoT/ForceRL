import glob
import os
from gamma.visual_model.object_articulation_part import gamma_model_net
import argparse
import torch
import numpy as np
from gamma.datasets.data_utilts import translate_pc_world_to_camera  
from robosuite.utils.transform_utils import quat2mat
import open3d as o3d
from scipy.spatial.transform import Rotation 
import termcolor


def get_joint_param_main(
        pcd_wf_path, 
        camera_info_path,
        model_path = './checkpoints/perception_models/gamma_best.pth',
        in_channels=3,
        num_point = 10000,
        num_classes = 3,
        ignore_label = 2,
        device = 'cuda',
        viz = False, 
        return_results_list = False
):
    '''
    Get the joint parameters from the point cloud and camera info

    args:
        return_results_list: whether to return the results dict (used to filter joint types)
    '''
    
    
    model = gamma_model_net(in_channel=in_channels, num_point=int(num_point), num_classes=int(num_classes), device=device).to(device)
    assert os.path.exists(model_path)
    print(termcolor.colored(f"load model from path: {model_path}", 'yellow'))
    model.load_state_dict(torch.load(model_path)) 
    pcd = o3d.io.read_point_cloud(pcd_wf_path)
    pcd_arr = np.array(pcd.points)
    camera_info = np.load(camera_info_path, allow_pickle=True)['data']['camera_config']
    # print(camera_info) 
    R = quat2mat(camera_info['quat_for_gamma']) 
    # to euler
    euler = Rotation.from_matrix(R).as_euler('xyz', degrees=True)
    # euler = [euler[2], euler[1], -euler[0]]
    # R = Rotation.from_euler('xyz', euler, degrees=True).as_matrix()
    t = camera_info['trans_absolute']

    # print("camera translation: ", t)
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t
    pcd_arr_cam = translate_pc_world_to_camera(pcd_arr, extrinsic) 
    pcd_cam = o3d.geometry.PointCloud()
    pcd_cam.points = o3d.utility.Vector3dVector(pcd_arr_cam) 
    o3d.io.write_point_cloud("microwave_cam.ply", pcd_cam)
    with torch.no_grad():
        model.eval()
        results, instance_labels, camera_pcd = model.online_inference(camera_pcd=pcd_arr_cam, view_res=viz, ignore_label=ignore_label)

    joint_translations = [res["joint_translation"] for res in results]
    joint_directions = [res["joint_direction"] for res in results]
    joint_types = [res["joint_type"] for res in results] 

    print("-----------------Computing Joint Parameters------------------")
    print("camera quat gamma got: ", camera_info['quat_for_gamma'])
    print("camera euler gamma got: ", euler)
    print('joint parameter results: ', results)
    print('--------------------------------------------------------------')

    # translate the joint parameters from camera frame to world frame
    result_trans, result_dir = translate_joint_param_camera_to_world(joint_translations, joint_directions, extrinsic)


    if return_results_list:
        return result_trans, result_dir, joint_types, results
    else:
        return result_trans, result_dir, joint_types

def translate_joint_param_camera_to_world(joint_translations, joint_directions, extrinsic):
    joint_translations_world = []
    joint_directions_world = []
    for joint_translation, joint_direction in zip(joint_translations, joint_directions):
        # joint_translation_world = np.dot(extrinsic[:3, :3], joint_translation)
        joint_translation_world = extrinsic[:3, :3].T @ joint_translation + extrinsic[:3, 3]
        # joint_direction_world = np.dot(extrinsic[:3, :3], joint_direction)
        joint_direction_world = extrinsic[:3, :3].T @ joint_direction
        joint_translations_world.append(joint_translation_world)
        joint_directions_world.append(joint_direction_world)
    return joint_translations_world, joint_directions_world