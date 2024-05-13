from env.move_box_env import MoveBoxEnv
from env.robot_env_cube_move import RobotCubeMove
import robosuite as suite
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import robosuite.utils.camera_utils as camera_utils
import sys
np.set_printoptions(threshold=sys.maxsize)
env = suite.make(
    "RobotCubeMove",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_segmentations = "element",
    camera_depths = True,
    # reward_shaping=True,
    control_freq=20,
    horizon=100,
    camera_names = ["birdview", "agentview", "frontview", "sideview"],
    camera_heights = 256,
    camera_widths = 256
)

def filp_image(image):
    return np.flip(image, 0)

if __name__ == "__main__":
    obs = env.reset()
    print(obs.keys())
    # seg_image = obs["agentview_segmentation_element"]
    # depth_image = obs['agentview_depth']
    # rgb_image = obs['agentview_image']
    seg_image = obs["agentview_segmentation_element"]
    depth_image = obs['agentview_depth']
    depth_image = camera_utils.get_real_depth_map(env.sim, depth_image)
    rgb_image = obs['agentview_image']
    # element_id = env.sim.model.geom_name2id('table_visual')
    # element_id = env.sim.model.geom_name2id('Door_handle_visual')
    element_id = env.sim.model.geom_name2id('cube_g0_vis')
    masked_segmentation = np.where(seg_image == int(element_id), 1.0, 0.0)
    print("max depth",np.max(depth_image))
    print(depth_image.shape )

    masked_rgb = np.multiply(rgb_image, masked_segmentation)
    # to int image
    masked_rgb = masked_rgb.astype(np.uint8)
    # print(masked_rgb)
    # show the masked rgb image
   
    masked_depth = np.multiply(depth_image, masked_segmentation).astype(np.float32)
    # masked_depth = np.zeros_like(depth_image,dtype=np.float32)
    # masked_depth = masked_depth + 0.5
    # for i in range(len(depth_image)):
    #     for j in range(len(depth_image[0])):
    #         if masked_segmentation[i][j] == 0:
    #             depth_image[i][j][0] = 0
    #         else:
    #             masked_depth[i][j][0] = depth_image[i][j][0]
    # masked_depth[masked_depth > 0.9] = 0.0 
    print(np.max(masked_depth))
    print(np.min(masked_depth))
    # normalize to 0-1
    # masked_depth = masked_depth - 0.97
    # masked_depth = masked_depth / np.max(masked_depth)
    # masked_depth = np.clip(masked_depth, 0, 1)
    masked_depth_image = o3d.geometry.Image(masked_depth)
    intrinisc_cam_parameters_numpy = camera_utils.get_camera_intrinsic_matrix(env.sim, "agentview", 256, 256)
    extrinsic_cam_parameters= camera_utils.get_camera_extrinsic_matrix(env.sim, "agentview")

    cx = intrinisc_cam_parameters_numpy[0][2]
    cy = intrinisc_cam_parameters_numpy[1][2]
    fx = intrinisc_cam_parameters_numpy[0][0]
    fy = intrinisc_cam_parameters_numpy[1][1]

    intrinisc_cam_parameters = o3d.camera.PinholeCameraIntrinsic(256, #width 
                                                        256, #height
                                                        fx,
                                                        fy,
                                                        cx,
                                                        cy)
    masked_pcd = o3d.geometry.PointCloud.create_from_depth_image(masked_depth_image,                                                       
                                              intrinisc_cam_parameters
                                             )
    masked_pcd.transform(extrinsic_cam_parameters)

        #estimate normals
    masked_pcd.estimate_normals()
        #orientation normals to camera
    masked_pcd.orient_normals_towards_camera_location(extrinsic_cam_parameters[:3,3])
    o3d.visualization.draw_geometries([masked_pcd]) 

    o3d.io.write_point_cloud('point_clouds/test.pcd', masked_pcd)

    plt.imshow(masked_depth)
    plt.show()
