"""
Run pointscore inference on partial pointcloud
"""

import os
import pickle
from argparse import ArgumentParser
import numpy as np
import torch
import open3d as o3d

# from train_model.test import load_conf
import aograsp.viz_utils as v_utils
import aograsp.model_utils as m_utils 
import aograsp.data_utils.dataset_utils as d_utils 
import aograsp.rotation_utils as r_utils
import copy


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

def inference_affordance(model, pcd_path, info_path, device, args):
    # Load model and weights
    # Directory to save output
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Create directories to save heatmaps and heatmap images
    point_score_dir = os.path.join(args.output_dir, "point_score")
    point_score_img_dir = os.path.join(args.output_dir, "point_score_img")
    os.makedirs(point_score_dir, exist_ok=True)
    os.makedirs(point_score_img_dir, exist_ok=True)

    # Read from args.pcd_path and put into input_dict for model
    if pcd_path is None:
        raise ValueError("Missing path to input point cloud (--pcd_path)")

    pcd_ext = os.path.splitext(pcd_path)[-1]
    if pcd_ext == ".ply":
        # Open3d pointcloud
        pcd = o3d.io.read_point_cloud(pcd_path)
        pcd = pcd.farthest_point_down_sample(4096) 
        ########################################################
        # Convert loaded point cloud into c frame
        pts_arr = np.array(pcd.points) 
        # print(pts_arr)
        pts_arr = d_utils.get_aograsp_pts_in_cam_frame_z_front_with_info(pts_arr, info_path) 
        # print(pts_arr)
    else:
        raise ValueError(f"{pcd_ext} filetype not supported")

    # Recenter pcd to origin 
    # pcd = pcd.farthest_point_down_sample(4096)
    mean = np.mean(pts_arr, axis=0)
    pts_arr_backup = copy.deepcopy(pts_arr)

    pts_arr -= mean 
    # pts_arr_backup = pts_arr

    # # Save the converted points 
    # pcd_cf = o3d.geometry.PointCloud() 
    # pcd_cf.points = o3d.utility.Vector3dVector(pts_arr)
    # o3d.io.write_point_cloud('point_clouds/camera_frame_temp_door.ply', pcd_cf)

    # Randomly shuffle points in pts
    np.random.shuffle(pts_arr)

    # Get pts as tensor and create input dict for model
    pts = torch.from_numpy(pts_arr).float().to(device) 

    print('Number of points: ', pts.shape[0])
    pts = torch.unsqueeze(pts, dim=0)
    input_dict = {"pcs": pts} 
    # print(pts) 

    # Run inference
    print("Computing heatmap")
    model.eval()
    with torch.no_grad():
        test_dict = model.test(input_dict, None)

    # Save heatmap point cloud 
    print(f'inference result: {test_dict.keys()}')
    scores = test_dict["point_score_heatmap"][0].cpu().numpy() 

    # Save the converted points and the mean into a npz.
    data_name = os.path.splitext(os.path.basename(pcd_path))[0] 
    points_cf_dict = {
        'pts_arr': pts_arr_backup, 
        'mean': mean
    } 
    with open('outputs/'+data_name+'_cf_info.npz', 'wb') as f:
        pickle.dump(points_cf_dict, f)
    # np.savez('outputs/'+data_name+'_cf_info.npz', points_cf_dict)

    return test_dict, pts_arr

def visualize_heatmap(test_dict, pts, point_score_dir, point_score_img_dir, data_name):
    scores = test_dict["point_score_heatmap"][0].cpu().numpy()
    pcd_path = os.path.join(point_score_dir, f"{data_name}.npz")

    pts_arr = pts
    mean = np.mean(pts_arr, axis=0) 
    pts_arr -= mean
    heatmap_dict = {
        "pts": pts_arr,  # Save original un-centered data
        "labels": scores,
    }
    np.savez_compressed(pcd_path, data=heatmap_dict) 
    print(pcd_path)

    # Save image of heatmap
    fig_path = os.path.join(point_score_img_dir, f"heatmap_{data_name}.png")
    hist_path = os.path.join(point_score_img_dir, f"heatmap_{data_name}_hist.png") 
    try:
        v_utils.viz_heatmap(
            heatmap_dict["pts"],
            scores,
            save_path=fig_path,
            frame="camera",
            scale_cmap_to_heatmap_range=True,
        )
    except Exception as e:
        print(e)

    v_utils.viz_histogram(
        scores,
        save_path=hist_path,
        scale_cmap_to_heatmap_range=True,
    )
    print(f"Heatmap saved to: {pcd_path}")
    print(f"Visualization saved to: {fig_path}")


def parse_args():
    parser = ArgumentParser()

    # main parameters (required)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Path to directory to write output files",
    )
    parser.add_argument("--pcd_path", type=str, help="Path to seg_pcd_clean.ply") 

    parser.add_argument('--info_path', type=str, help='Path to information.npz')

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="cpu or cuda:x for using cuda on GPU x",
    )

    parser.add_argument(
        "--display",
        type=str,
        choices=[":1", ":2", ":3"],
        help="Display number; for accomdating remote desktop setups",
    )

    # parse args
    conf = parser.parse_args()

    conf.data_path = None

    return conf



if __name__ == "__main__":
    args = parse_args() 
    model = m_utils.load_model(
        model_conf_path='/home/wgao22/ref-codes/ao-grasp/aograsp/aograsp_model/conf.pth', 
        ckpt_path='/home/wgao22/ref-codes/ao-grasp/aograsp/aograsp_model/770-network.pth'
    )
    model.to(args.device)
    pcd_path = args.pcd_path 
    info_path = args.info_path
    device = args.device
    t_dict, pts = inference_affordance(model, pcd_path, info_path, device, args) 
    # pts = o3d.io.read_point_cloud(pcd_path) 
    visualize_heatmap(t_dict, pts, '/home/wgao22/projects/ForceRL/outputs/point_score', '/home/wgao22/projects/ForceRL/outputs/point_score_img', 'microwave')
