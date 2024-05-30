# open drawer using robot
# TODO: use ikflow to get ik solution

import open3d as o3d
import robosuite as suite
from env.robot_drawer_opening import RobotDrawerOpening
from env.drawer_opening import DrawerOpeningEnv
from utils.sim_utils import get_pointcloud, flip_image
from utils.gpd_utils import gpd_get_grasp_pose
import numpy as np
import matplotlib.pyplot as plt

# initialize robot controller
controller_name = "OSC_POSE"
controller_configs = suite.load_controller_config(default_controller=controller_name)
env:RobotDrawerOpening = suite.make(
    "RobotDrawerOpening",
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
    camera_heights = [1024,256,512,1024],
    camera_widths = [1024,1024,1024,1024]
)

# env:RobotDrawerOpening = suite.make(
#     "DrawerOpeningEnv",
#     has_renderer=True,
#     use_camera_obs=True,
#     has_offscreen_renderer=True,
#     camera_depths = True,
#     camera_segmentations = "element",
#     control_freq = 20,
#     horizon=10000,
#     camera_names = ["birdview", "agentview", "frontview", "sideview"],
#     camera_heights = [1024,1024,512,1024],
#     camera_widths = [1024,1024,1024,1024]
# )

obs = env.reset()
# env.render()
camera_name = "frontview"
depth_image = obs['{}_depth'.format(camera_name)]
depth_image = flip_image(depth_image)
# plt.imshow(depth_image)
# plt.show()
pointcloud = get_pointcloud(env, obs, ["agentview", "frontview","sideview"], [256,512,1024], [1024,1024,1024], ["handle", "foo"])
o3d.visualization.draw_geometries([pointcloud])
o3d.io.write_point_cloud("point_clouds/drawer_pointcloud.pcd", pointcloud)
x = gpd_get_grasp_pose('drawer_pointcloud.pcd',cfg_file_name='experiment_grasp.cfg')


