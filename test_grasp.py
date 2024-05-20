import pickle
from utils.gpd_utils import gpd_get_grasp_pose 

x = gpd_get_grasp_pose('test.pcd',cfg_file_name='test.cfg')
# print('-------------------------------\n\n\n\n\n\n')
# print(type(x.pos_x))

# grasp_pose_path = "/home/wenchang/projects/ForceRL/MjJacoDoorGrasps"

# with open(grasp_pose_path, 'rb') as f:
#     grasp_poses = pickle.load(f)
# print(grasp_poses)