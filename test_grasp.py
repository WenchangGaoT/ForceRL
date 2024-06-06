import pickle
from utils.gpd_utils import gpd_get_grasp_pose 

x = gpd_get_grasp_pose('krylon.pcd',cfg_file_name='experiment_grasp.cfg')
# print('-------------------------------\n\n\n\n\n\n')
# print(type(x.pos_x))

print(x.pos_x[0], x.pos_y[0], x.pos_z[0])
for j in range(9):
    print(x.rotation_matrix_list[0][j])
# grasp_pose_path = "/home/wenchang/projects/ForceRL/MjJacoDoorGrasps"

# with open(grasp_pose_path, 'rb') as f:
#     grasp_poses = pickle.load(f)
# print(grasp_poses)