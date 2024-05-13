from utils.gpd_utils import gpd_get_grasp_pose 

x = gpd_get_grasp_pose('test.pcd',cfg_file_name='test.cfg')
print('-------------------------------\n\n\n\n\n\n')
print(type(x.size))