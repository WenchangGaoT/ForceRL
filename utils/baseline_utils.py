import pickle
from utils.sim_utils import init_camera_pose
import robosuite as suite
import robosuite.utils.mjcf_utils as mjcf_utils
import robosuite.utils.transform_utils as transform_utils 
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import json
import copy
import h5py
import utils.aograsp_utils.dataset_utils as d_utils 
import utils.aograsp_utils.rotation_utils as r_utils
import open3d as o3d


def save_camera_info(env,
                     camera_info_path, 
                     camera_pos, 
                     camera_quat, 
                     scale_factor = 1):
    

    # print("object quat: ", env.obj_quat)
    object_euler = R.from_quat(env.obj_quat).as_euler('xyz', degrees=False)
    print("object euler: ", R.from_quat(env.obj_quat).as_euler('xyz', degrees=True))
    cam_pos_actual = init_camera_pose(env, camera_pos, scale_factor, camera_quat=camera_quat)
    camera_rot_90 = transform_utils.euler2mat(np.array([0, 0, -np.pi-object_euler[0]])) @ transform_utils.quat2mat(camera_quat) 
    camera_quat_rot_90 = transform_utils.mat2quat(camera_rot_90)
    
    camera_euler = R.from_quat(camera_quat).as_euler('xyz', degrees=True)
    camera_euler_for_gamma = np.array([camera_euler[-1], camera_euler[1], -camera_euler[0]]) 
    camera_rot_for_gamma = R.from_euler('xyz', camera_euler_for_gamma, degrees=True).as_matrix()
    camera_rot_for_gamma = transform_utils.euler2mat(np.array([0, 0, -np.pi-object_euler[0]])) @ camera_rot_for_gamma
    camera_quat_for_gamma = R.from_matrix(camera_rot_for_gamma).as_quat()

    camera_config = {'data': {
                        'camera_config': {
                            # 'trans': camera_pos*scale_factor, 
                            'trans': camera_pos, 
                            'quat': camera_quat_rot_90,
                            'trans_absolute': cam_pos_actual,
                            'quat_for_gamma': camera_quat_for_gamma,
                        }
                    }
                } 
    
    with open(camera_info_path, 'wb') as f:
        pickle.dump(camera_config, f) 

def get_camera_info(env,
                    camera_pos, 
                     camera_quat, 
                     scale_factor = 1):
    '''
    get camera information
    '''
    object_euler = R.from_quat(env.obj_quat).as_euler('xyz', degrees=False)
    print("object euler: ", R.from_quat(env.obj_quat).as_euler('xyz', degrees=True))
    cam_pos_actual = init_camera_pose(env, camera_pos, scale_factor, camera_quat=camera_quat)
    camera_rot_90 = transform_utils.euler2mat(np.array([0, 0, -np.pi-object_euler[0]])) @ transform_utils.quat2mat(camera_quat) 
    camera_quat_rot_90 = transform_utils.mat2quat(camera_rot_90)
    
    camera_euler = R.from_quat(camera_quat).as_euler('xyz', degrees=True)
    camera_euler_for_gamma = np.array([camera_euler[-1], camera_euler[1], -camera_euler[0]]) 
    camera_rot_for_gamma = R.from_euler('xyz', camera_euler_for_gamma, degrees=True).as_matrix()
    camera_rot_for_gamma = transform_utils.euler2mat(np.array([0, 0, -np.pi-object_euler[0]])) @ camera_rot_for_gamma
    camera_quat_for_gamma = R.from_matrix(camera_rot_for_gamma).as_quat()

    camera_config = { 'camera_config': {
                            # 'trans': camera_pos*scale_factor, 
                            'trans': camera_pos, 
                            'quat': camera_quat_rot_90,
                            'trans_absolute': cam_pos_actual,
                            'quat_for_gamma': camera_quat_for_gamma,
                        }
                } 
    return camera_config


def load_wf_grasp_proposals(proposal_path, top_k=10):
    with open(proposal_path, 'rb') as f:
        proposals = np.load(f, allow_pickle=True)
        proposals = proposals['data'].item()
    
    g_pos_wf = proposals['pos']
    g_quat_wf = proposals['quat']
    scores = proposals['scores']

    sorted_grasp_tuples = [(g_pos_wf[i], g_quat_wf[i], scores[i]) for i in range(len(g_pos_wf))]
    sorted_grasp_tuples.sort(key=lambda x: x[2], reverse=True)
    top_k_pos_wf = [g[0] for g in sorted_grasp_tuples][:top_k]
    top_k_quat_wf = [g[1] for g in sorted_grasp_tuples][:top_k]

    return top_k_pos_wf, top_k_quat_wf


def  get_grasp_env_states(
        env_name, 
        env_kwargs,
        grasp_pos,
        grasp_rot_vec,
        env_model_dir,
        object_rotation_range = (-np.pi / 2, 0.),
        object_robot_distance_range = (0.65, 0.75),
        number_of_grasp_states = 4,
):
    '''
    create env, set object position and rotation, drive the robto to grasp the object, save the robot states
    '''
    env_kwargs["rotate_around_robot"] = True

    # rendering config for debugging
    env_kwargs["has_renderer"] = True
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = True
    env_kwargs["camera_depths"] = True
    env_kwargs["camera_names"] = "sideview"
    env_kwargs["camera_heights"] = 256
    env_kwargs["camera_widths"] = 256
    env_kwargs["cache_video"] = False
    env_kwargs["get_grasp_proposals_flag"] = True
    # env_kwargs["use_grasp_states"] = False
    env_kwargs["move_robot_away"] = False
    # env_kwargs["open_percentage"] = 1.0
    # env_kwargs["render_camera"] = "birdview"

    object_name = env_kwargs["object_name"]
    object_scale = env_kwargs["object_scale"]

    print(env_kwargs["get_grasp_proposals_flag"])

    controller_cfg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"controller_configs")
    controller_cfg_path = os.path.join(controller_cfg_dir, "osc_pose_absolute_kp_20.json")

    with open(controller_cfg_path, 'r') as f:
        controller_configs = json.load(f)
    
    # need to load absolute pose controller config
    env_kwargs["controller_configs"] = controller_configs
    
    # load a dummy env, just to get the available objects
    dummy_env = suite.make(env_name, **env_kwargs)
    available_objects = dummy_env.available_objects()
    dummy_env.close()
    available_objects_list = []
    for obj_list in available_objects.values():
        available_objects_list.extend(obj_list)
    # available_objects = ["train-dishwasher-1"]
    print("available objects: ", available_objects)

    env_states_dict = dict()

    for object_name in available_objects_list:

        env_kwargs["object_name"] = object_name

        for i in range(number_of_grasp_states):

            current_object_rotation = object_rotation_range[0] + (object_rotation_range[1] - object_rotation_range[0]) * i / (number_of_grasp_states - 1)
            print("current object rotation: ", current_object_rotation * 180 / np.pi)
            current_object_distance = np.random.uniform(object_robot_distance_range[0], object_robot_distance_range[1])
            env_kwargs["obj_rotation"] = (current_object_rotation, current_object_rotation)
            # env_kwargs["obj_rotation"] = (-np.pi/2, -np.pi/2)
            env_kwargs["object_robot_distance"] = (current_object_distance, current_object_distance)

            env = suite.make(env_name, **env_kwargs)

            # drive the robot to grasp the object
            drive_to_grasp(env, grasp_pos, grasp_rot_vec)

            # get the env states
            env_states_flattened = copy.deepcopy(env.sim.get_state().flatten())
            print("env states shape: ", env_states_flattened.shape)
            model = env.sim.model.get_xml()

            env_states_dict[f"{object_name}_{i}"] = (env_states_flattened, model)
            
            env.close()
    
    # save the env states
    grasp_states_save_path = os.path.join(env_model_dir, f"{env_name}.h5")
    with h5py.File(grasp_states_save_path, 'w') as hf:
        for name, (env_states_flattened, model) in env_states_dict.items():
            group = hf.create_group(name)
            group.create_dataset("state", data=env_states_flattened)
            group.create_dataset("model", data=np.string_(model))
        

def drive_to_grasp(env
                   , grasp_pos, grasp_rot_vec):
    '''
    drive the robot to grasp the object
    '''
    obs = env.reset()

    final_grasp_pos = obs["grasp_pos"]
    print("final grasp pos: ", final_grasp_pos)
    robot_gripper_pos = obs["gripper_pos"]
    rotation_vector = obs["grasp_rot"]

    prepaer_grasp_pos = final_grasp_pos + np.array([-0., 0, 0.15])
    
    action = np.concatenate([robot_gripper_pos, rotation_vector, [-1]])

    for i in range(150):
        # action = np.zeros_like(env.action_spec[0])
        action = np.concatenate([prepaer_grasp_pos, rotation_vector, [-1]])
        env.step(action)
        
        env.render()


    for i in range(50):
        # action = np.zeros_like(env.action_spec[0])
        action = np.concatenate([final_grasp_pos, rotation_vector, [-1]])
        env.step(action)
        env.render()

    # close the gripper
    for i in range(50):
        action = np.concatenate([final_grasp_pos, rotation_vector, [1]])
        env.step(action)
        env.render()


def baseline_read_cf_grasp_proposals(pts_cf_path,
                                     proposal_path):
    '''
    read the grasp proposals in camera frame
    '''
    proposal_cf_dict = np.load(proposal_path, allow_pickle=True)['data'].item()  

    pts_cf = o3d.io.read_point_cloud(pts_cf_path)
    pts_cf_arr = np.array(pts_cf.points)

    proposals = proposal_cf_dict['proposals'] 
    proposal_points_cf = np.array([p[0] for p in proposals]) 
    proposal_quats_cf = np.array([p[1] for p in proposals]) 
    proposal_scores = np.array([p[2] for p in proposals])
    proposal_points_cf += np.mean(pts_cf_arr, axis=0) 
    return proposal_points_cf, proposal_quats_cf, proposal_scores


def baseline_pts_zfront_to_wf(camera_info_dict, proposal_points_cf):
    '''
    convert the proposals from camera frame to world frame
    '''
    print("camera_info_dict: ", camera_info_dict)
    cam_pos = camera_info_dict["camera_config"]["trans_absolute"]
    cam_quat = camera_info_dict["camera_config"]["quat_for_gamma"]
    H_cam2world_xfront = r_utils.get_H(r_utils.get_matrix_from_quat(cam_quat), cam_pos)
    H_world2cam_xfront = r_utils.get_H_inv(H_cam2world_xfront)
    H_world2cam_zfront = d_utils.get_H_world_to_cam_z_front_from_x_front(H_world2cam_xfront) 
    pts_cf_homo = np.concatenate([proposal_points_cf, np.ones_like(proposal_points_cf[:, :1])], axis=-1) 
    # pts_wf_arr = np.matmul(pts_cf_homo, np.linalg.inv(H_world2cam_zfront.T))[:, :3]
    pts_wf_arr = np.matmul(np.linalg.inv(H_world2cam_zfront),pts_cf_homo.T).T[:, :3]
    return pts_wf_arr 

def baseline_quat_zfront_to_wf(proposal_quats_cf,camera_info_dict):
    '''
    convert proposal quat from cam frame to world frame
    '''
    cam_pos = camera_info_dict["camera_config"]["trans_absolute"]
    cam_quat = camera_info_dict["camera_config"]["quat_for_gamma"]
    H_cam2world_xfront = r_utils.get_H(r_utils.get_matrix_from_quat(cam_quat), cam_pos)
    H_world2cam_xfront = r_utils.get_H_inv(H_cam2world_xfront)
    H_world2cam_zfront = d_utils.get_H_world_to_cam_z_front_from_x_front(H_world2cam_xfront) 
    H_cam2world_zfront = np.linalg.inv(H_world2cam_zfront)
    # get rotation part of H_cam2world_zfront
    R_rotation_cam2world_zfront = H_cam2world_zfront[:3, :3]
    
    # transform the quaternion from cf to wf
    # use rotation matrix to rotate the quaternion
    R_rotation_cam2world_zfront_mat = R.from_matrix(R_rotation_cam2world_zfront)
    pts_rotation_cf = [R.from_quat(q) for q in proposal_quats_cf]
    pts_rotation_wf = [R_rotation_cam2world_zfront_mat * q for q in pts_rotation_cf]

    # back to quaternion
    pts_quat_wf = [q.as_quat() for q in pts_rotation_wf]
    return pts_quat_wf



def baseline_get_grasp_proposals(camera_info_dict,
                                 proposal_points_cf,
                                    proposal_quats_cf,
                                    proposal_scores,
                                    top_k=10):
    '''
    main function for getting grasp proposals

    args:
        run_cgn: bool, whether to run the CGN model to get a new set of grasp proposals since CGN is slow.
    '''


    # convert the proposals from camera frame to world frame
    g_pos_wf = baseline_pts_zfront_to_wf(camera_info_dict, proposal_points_cf)
    g_quat_wf = baseline_quat_zfront_to_wf(proposal_quats_cf, camera_info_dict)
    scores = proposal_scores
    sorted_grasp_tuples = [(g_pos_wf[i], g_quat_wf[i], scores[i]) for i in range(len(g_pos_wf))]
    sorted_grasp_tuples.sort(key=lambda x: x[2], reverse=True)
    top_k_pos_wf = [g[0] for g in sorted_grasp_tuples][:top_k]
    top_k_quat_wf = [g[1] for g in sorted_grasp_tuples][:top_k]

    
    top_k_pos_wf = [g[0] for g in sorted_grasp_tuples][:top_k]
    top_k_quat_wf = [g[1] for g in sorted_grasp_tuples][:top_k]

    return  top_k_pos_wf, top_k_quat_wf


