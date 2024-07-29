import torch 
import argparse
import os

import robosuite as suite
import aograsp.model_utils as m_utils 
import aograsp.viz_utils as v_utils 
import aograsp.data_utils as d_utils 
import open3d as o3d 
import numpy as np

from utils.sim_utils import get_pointcloud 
from env.curri_door_env import CurriculumDoorEnv
from env.robot_revolute_env import RobotRevoluteOpening 
from env.robot_drawer_opening import RobotDrawerOpening

import aograsp.rotation_utils as r_utils
# from env.robo
# from aograsp.run_pointscore_inference import get_heatmap


def get_aograsp_pts_in_cam_frame_z_front(pcd):
    """
    Given a path to a partial point cloud render/000*/point_cloud_seg.npz
    from the AO-Grasp dataset, get points in camera frame with z-front, y-down
    """

    # Load pts in world frame
    # pcd_dict = np.load(pc_path, allow_pickle=True)["data"].item()
    # pts_wf = pcd_dict["pts"]

    # Load camera pose information
    render_dir = os.path.dirname(pc_path)
    info_path = os.path.join(render_dir, "info.npz")
    if not os.path.exists(info_path):
        raise ValueError(
            "info.npz cannot be found. Please ensure that you are passing in a path to a point_cloud_seg.npz file"
        )
    info_dict = np.load(info_path, allow_pickle=True)["data"].item()
    cam_pos = info_dict["camera_config"]["trans"]
    cam_quat = info_dict["camera_config"]["quat"]

    # Transform points from world to camera frame
    H_cam2world_xfront = r_utils.get_H(r_utils.get_matrix_from_quat(cam_quat), cam_pos)
    H_world2cam_xfront = r_utils.get_H_inv(H_cam2world_xfront)
    H_world2cam_zfront = get_H_world_to_cam_z_front_from_x_front(H_world2cam_xfront)
    pts_cf = r_utils.transform_pts(pts_wf, H_world2cam_zfront)

    return pts_cf


parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default='outputs/')
parser.add_argument('--pcd_path', type=str, default='point_clouds/temp_door.pcd') 
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()

aograsp_model = m_utils.load_model(
    model_conf_path='models/grasp_models/ao-grasp/conf.pth', 
    ckpt_path='models/grasp_models/ao-grasp/770-network.pth'
) 
aograsp_model.to(torch.device(args.device))

controller_name = "OSC_POSE"
controller_configs = suite.load_controller_config(default_controller=controller_name)
env = suite.make(
    "RobotRevoluteOpening",
    robots="Panda",
    has_renderer=True,
    use_camera_obs=True,
    has_offscreen_renderer=True,
    camera_depths = True,
    camera_segmentations = "element",
    controller_configs=controller_configs,
    control_freq = 20,
    horizon=10000,
    camera_names = ["birdview", "agentview", "frontview", "sideview"],
    # camera_heights = 256,
    # camera_widths = 256
    camera_heights = [256,512, 256,1024],
    camera_widths = [1024,1024,1024,1024]
)

obs = env.reset()
env.render()
# get pointcloud
pointcloud = get_pointcloud(env, obs, ['birdview', "agentview", "frontview", "sideview"], [256, 512, 256, 1024], [1024, 1024,1024,1024], ['table', 'drawer']) 
pointcloud = pointcloud.farthest_point_down_sample(4096) 


obs_arr = np.asarray(pointcloud.points) 
# print(obs_arr)
o3d.io.write_point_cloud("point_clouds/temp_door.ply", pointcloud) 

# pointcloud = d_utils.get_aograsp_pts_in_cam_frame_z_front('point_clouds/temp_door.ply')

point_score_dir = os.path.join(args.output_dir, "point_score")
point_score_img_dir = os.path.join(args.output_dir, "point_score_img")
os.makedirs(point_score_dir, exist_ok=True)
os.makedirs(point_score_img_dir, exist_ok=True) 

mean = np.mean(obs_arr, axis=0)
obs_arr -= mean

# Randomly shuffle points in pts
np.random.shuffle(obs_arr)

# Get pts as tensor and create input dict for model
pts = torch.from_numpy(obs_arr).float().to(args.device)
pts = torch.unsqueeze(pts, dim=0)
input_dict = {"pcs": pts}

# Run inference
print("Computing heatmap")

aograsp_model.eval()
with torch.no_grad():
    test_dict = aograsp_model.test(input_dict, None)

# Save heatmap point cloud
scores = test_dict["point_score_heatmap"][0].cpu().numpy()

# pcd_path = os.path.join(point_score_dir, f"{data_name}.npz")
heatmap_dict = {
    "pts": obs_arr + mean,  # Save original un-centered data
    "labels": scores,
}
# np.savez_compressed(pcd_path, data=heatmap_dict) 
# print(pcd_path)

# Save image of heatmap
fig_path = os.path.join(point_score_img_dir, f"heatmap.png")
hist_path = os.path.join(point_score_img_dir, f"heatmap_hist.png") 
print(point_score_img_dir) 
print(scores)
try:
    v_utils.viz_heatmap(
        heatmap_dict["pts"],
        scores,
        save_path=fig_path,
        frame="camera",
        scale_cmap_to_heatmap_range=True,
    ) 
    print('heatmapped')
except Exception as e:
    print(e)

v_utils.viz_histogram(
    scores,
    save_path=hist_path,
    scale_cmap_to_heatmap_range=True,
)
# print(f"Heatmap saved to: {pcd_path}")
# print(f"Visualization saved to: {fig_path}")