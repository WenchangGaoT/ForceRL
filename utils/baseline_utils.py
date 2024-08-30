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
        grasp_states_save_path,
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
    env_kwargs["use_grasp_states"] = False
    env_kwargs["move_robot_away"] = False
    # env_kwargs["open_percentage"] = 1.0
    # env_kwargs["render_camera"] = "birdview"
    print(env_kwargs["get_grasp_proposals_flag"])

    controller_cfg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"controller_configs")
    controller_cfg_path = os.path.join(controller_cfg_dir, "osc_pose_absolute_kp_20.json")

    with open(controller_cfg_path, 'r') as f:
        controller_configs = json.load(f)
    
    # need to load absolute pose controller config
    env_kwargs["controller_configs"] = controller_configs

    env_states_list = []

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
        print("env states shape: ", env_states_flattened)
        env_states_list.append(env_states_flattened)

        # model_save_path = os.path.join(env_model_dir, "{}_{}.xml".format(env_name, i))
        # mjcf_utils.save_sim_model(env.sim, model_save_path)

        env.close()
    
    # save the env states
    env_states_array = np.array(env_states_list)
    # test_array = np.array([1,2,3,4,5,6,7,8,9,10])
    np.save(grasp_states_save_path, env_states_array)
    return grasp_states_save_path


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

    for i in range(100):
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
    for i in range(20):
        action = np.concatenate([final_grasp_pos, rotation_vector, [1]])
        env.step(action)
        env.render()
