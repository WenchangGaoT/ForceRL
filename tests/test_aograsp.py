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
from env.original_door_env import OriginalDoorEnv 
from env.camera_door_env import CameraDoorEnv

import aograsp.rotation_utils as r_utils 
import matplotlib.pyplot as plt
# from env.robo
# from aograsp.run_pointscore_inference import get_heatmap 
from scipy.spatial.transform import Rotation as R 
from robosuite.utils.transform_utils import quat2mat

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
    # use_camera_obs=True,
    controller_configs=controller_configs,
    control_freq = 20,
    horizon=10000,
    # camera_names = ["birdview", "agentview", "frontview", "sideview"], 
    # camera_heights = [256,512, 256,1024], 
    # camera_widths = [1024,1024,1024,1024]

    # camera_names = ['agentview'], 
    # camera_heights = 256,
    # camera_widths = 256, 

    # camera_names = ['sideview', 'birdview'], 
    # camera_heights = [512, 512], 
    # camera_widths = [1024, 1024] 

    camera_names = ['sideview'], 
    camera_heights = 256, 
    camera_widths = 256

)

obs = env.reset() 
# env.display_cameras() 
# print(obs['frontview_depth']) 
print('rotation matrix for [0.5, 0.5, 0.5, 0.5]: ') 
m1 = quat2mat(np.array([-0.5, -0.5, 0.5, 0.5])) 
print(m1)

print('rotation matrix for quat: ') 
m2 = quat2mat(np.array([-0.005696068282031459, 0.19181093598117868, 0.02913152799094475, 0.9809829120433564]))  
# m2 = quat2mat(np.array([0, 0, 0, 1]))
print(m2)

M = np.dot(m1, m2) 
quat = R.from_matrix(M).as_quat() 
print('Corresponding quaternion: ', quat)


obj_pos = env.obj_pos 
camera_trans = 2*np.array([-1.77542536, -0.02539806,  0.30146208])
# set_camera_pose(env, 'sideview', [-0.77542536, -0.02539806,  2.20146208], [-0.005696068282031459, 0.19181093598117868, 0.02913152799094475, 0.9809829120433564])
set_camera_pose(env, 'sideview', obj_pos + camera_trans, quat) 
# display_camera_pose(env, 'frontview')
display_camera_pose(env, 'sideview') 
low, high = env.action_spec
obs, reward, done, _ = env.step(np.zeros_like(low))
plt.imshow(obs['sideview_image']) 
plt.show()
plt.imshow(obs['sideview_depth'], cmap='gray')
plt.show() 

# obj_name = 'drawer' 
# obj_id = env.sim.model.body_name2id(obj_name) 
# print(obj_name)
# env.render()
# get pointcloud
# pointcloud = get_pointcloud(env, obs, ['birdview', "agentview", "frontview", "sideview"], [256, 512, 256, 1024], [1024, 1024,1024,1024], ['table', 'drawer']) 
# pointcloud = get_pointcloud(env, obs, ['sideview', 'birdview'], [512, 512], [1024, 1024], ['door'])
pointcloud = get_pointcloud(env, obs, ['sideview'], [256], [256], ['drawer'])
pointcloud = pointcloud.farthest_point_down_sample(4096) 


obs_arr = np.asarray(pointcloud.points) 
# obs_arr[:, 0:2] = -obs_arr[:, -0:2]
print(obs_arr.shape) 

pointcloud = o3d.geometry.PointCloud()
pointcloud.points = o3d.utility.Vector3dVector(obs_arr)


# print(obs_arr)
o3d.io.write_point_cloud("point_clouds/temp_door.ply", pointcloud) 
print('point cloud saved')

# pointcloud = d_utils.get_aograsp_pts_in_cam_frame_z_front('point_clouds/temp_door.ply')

# point_score_dir = os.path.join(args.output_dir, "point_score")
# point_score_img_dir = os.path.join(args.output_dir, "point_score_img")
# os.makedirs(point_score_dir, exist_ok=True)
# os.makedirs(point_score_img_dir, exist_ok=True) 

# mean = np.mean(obs_arr, axis=0)
# obs_arr -= mean

# # Randomly shuffle points in pts
# np.random.shuffle(obs_arr)

# # Get pts as tensor and create input dict for model
# pts = torch.from_numpy(obs_arr).float().to(args.device)
# pts = torch.unsqueeze(pts, dim=0)
# input_dict = {"pcs": pts}

# # Run inference
# print("Computing heatmap")

# aograsp_model.eval()
# with torch.no_grad():
#     test_dict = aograsp_model.test(input_dict, None)

# # Save heatmap point cloud
# scores = test_dict["point_score_heatmap"][0].cpu().numpy()

# # pcd_path = os.path.join(point_score_dir, f"{data_name}.npz")
# heatmap_dict = {
#     "pts": obs_arr + mean,  # Save original un-centered data
#     "labels": scores,
# }
# # np.savez_compressed(pcd_path, data=heatmap_dict) 
# # print(pcd_path)

# # Save image of heatmap
# fig_path = os.path.join(point_score_img_dir, f"heatmap.png")
# hist_path = os.path.join(point_score_img_dir, f"heatmap_hist.png") 
# print(point_score_img_dir) 
# print(scores)
# try:
#     v_utils.viz_heatmap(
#         heatmap_dict["pts"],
#         scores,
#         save_path=fig_path,
#         frame="camera",
#         scale_cmap_to_heatmap_range=True,
#     ) 
#     print('heatmapped')
# except Exception as e:
#     print(e)

# v_utils.viz_histogram(
#     scores,
#     save_path=hist_path,
#     scale_cmap_to_heatmap_range=True,
# )

# for _ in range(10000):
#     env.step(np.zeros(3))
# # print(f"Heatmap saved to: {pcd_path}")
# # print(f"Visualization saved to: {fig_path}")