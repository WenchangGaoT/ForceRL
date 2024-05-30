# open drawer using robot
# TODO: use ikflow to get ik solution

import open3d as o3d
import robosuite as suite
from env.robot_drawer_opening import RobotDrawerOpening
from utils.sim_utils import get_pointcloud
from utils.gpd_utils import gpd_get_grasp_pose
import numpy as np

# initialize robot controller
controller_name = "OSC_POSE"
controller_configs = suite.load_controller_config(default_controller=controller_name)

