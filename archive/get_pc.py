 
controller_name = "OSC_POSE"
    controller_configs = suite.load_controller_config(default_controller=controller_name)


# env = suite.make(
    #     environment_name,
    #     robots="Panda",
    #     has_renderer=True,
    #     use_camera_obs=True,
    #     has_offscreen_renderer=True,
    #     camera_depths = True,
    #     camera_segmentations = "element",
    #     controller_configs=controller_configs,
    #     control_freq = 20,
    #     horizon=10000,
    #     camera_names = ['sideview'], 
    #     camera_heights = 256, 
    #     camera_widths = 256, 
    #     obj_rotation=(np.pi/6, np.pi/6)
    # )    

# print('rotation matrix for [-0.5, -0.5, 0.5, 0.5]: ') 
    # m1 = quat2mat(np.array([-0.5, -0.5, 0.5, 0.5])) # Camera local frame to world frame front, set camera fram
    # # print(m1)

    # obj_quat = env.obj_quat 
    # obj_quat = convert_quat(obj_quat, to='xyzw')
    # rotation_mat_world = quat2mat(obj_quat)
    # rotation_euler_world = mat2euler(rotation_mat_world)
    # rotation_euler_cam = np.array([rotation_euler_world[2], 0,0])
    # m3_world = quat2mat(obj_quat)
    # # obj_quat = np.array([0.383, 0, 0, 0.924])

    # m3 = euler2mat(rotation_euler_cam)# Turn camera and microwave simultaneously
    # # m3 = np.eye(3)

    # print('rotation matrix for quat: ') 
    # # m2 = quat2mat(np.array([-0.005696068282031459, 0.19181093598117868, 0.02913152799094475, 0.9809829120433564])) # Turn camera to microwave

    # m2 = quat2mat(np.array(camera_quat)) # Turn camera to microwave
    # M = np.dot(m1,m2)
    # M = np.dot(M, m3.T) 
    # quat = R.from_matrix(M).as_quat() 
    # print('Corresponding quaternion: ', quat)

    # obj_pos = env.obj_pos 
    # camera_pos = np.array(camera_pos)
    # camera_trans = scale_factor*camera_pos 
    # camera_trans = np.dot(m3_world, camera_trans) 

    # set_camera_pose(env, 'sideview', obj_pos + camera_trans, quat) 


        # pts_npz = {
    #     'data': {
    #         'pts': pts_arr, 
    #         'obj_pos': np.array(env.obj_pos), 
    #         'obj_quat': np.array(env.obj_quat)
    #         }
    #     }
    
    # np.save(f"{pcd_path}.npz", pts_npz) 
    # with open(f'{pcd_path}.npz', 'wb') as f:
    #     pickle.dump(pts_npz, f)
    # print(f'point cloud npz saved to {pcd_path}.npz')
