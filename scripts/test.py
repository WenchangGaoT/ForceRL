from env.robot_prismatic_env import RobotPrismaticEnv 
from env.robot_revolute_env import RobotRevoluteOpening
import open3d as o3d
import robosuite as suite
from utils.sim_utils import get_pointcloud, flip_image
from utils.gpd_utils import gpd_get_grasp_pose
import numpy as np
import matplotlib.pyplot as plt
from utils.sim_utils import initialize_robot_position, get_close_to_grasp_point, interact_estimate_params
import robosuite.utils.transform_utils as transform
from utils.control_utils import PathPlanner, Linear, Gaussian
import os 
import json
from scipy.spatial.transform import Rotation as R 
import utils.sim_utils as sim_utils
from grasps.aograsp.get_pointclouds import get_aograsp_ply_and_config
from grasps.aograsp.get_affordance import get_affordance_main
from grasps.aograsp.get_proposals import get_grasp_proposals_main
from gamma.get_joint_param import get_joint_param_main
from agent.td3 import TD3
from termcolor import colored, cprint
import termcolor
import warnings
import yaml 
from perceptions.interactive_perception import InteractivePerception 
from env.wrappers import make_env
warnings.filterwarnings("ignore", category=FutureWarning) 

revolute_state = lambda state: np.concatenate([
    state['hinge_direction'], 
    state['force_point']-state['hinge_position']-np.dot(state['force_point']-state['hinge_position'], state['hinge_direction'])*state['hinge_direction']
    ]) 

prismatic_state = lambda obs: np.concatenate([obs["joint_direction"], obs["force_point_relative_to_start"]]) 


def get_state(obs, params, final_grasp_pos):
    '''
    Get the state from observation and estimated parameters
    '''
    if params['joint_type'] == 'prismatic':
        direction = params['joint_direction'] 
        progress = obs['robot0_eef_pos'] - final_grasp_pos
        return np.concatenate([direction, progress])
    else:
        hinge_direction = params['joint_direction'] 
        hinge_position = params['joint_position']
        force_point = obs['robot0_eef_pos'] 
        projection = force_point - hinge_position - np.dot(force_point - hinge_position, hinge_direction) * hinge_direction 
        return np.concatenate([hinge_direction, projection])
        # return revolute_state(obs)

def test(run_id):

    controller_name = "OSC_POSE"
    controller_cfg_dir = os.path.join('cfg',"controller_configs")
    controller_cfg_path = os.path.join(controller_cfg_dir, "osc_pose_absolute.json")
    controller_configs = suite.load_controller_config(custom_fpath = controller_cfg_path,default_controller=controller_name)

    with open(controller_cfg_path, 'r') as f:
        controller_configs = json.load(f)

    with open('cfg/test_config.yaml') as f:
        config = yaml.safe_load(f) 

    experiment_name = config['experiment_name'] 
    test_objects = config['test_objects'] 
    run_id = config['run_id']

    force_policy_config = config['force_policy'] 
    environment_config = config['environment'] 
    robot_config = config['robot'] 

    # RL environment configs
    max_timesteps = environment_config['max_timesteps'] 
    action_repeat = environment_config['action_repeat']

    # Robot configs
    action_scale = robot_config['action_scale']
    kp_interaction = robot_config['kp_interaction']
    kp_execution = robot_config['kp_execution']
    interaction_timesteps = robot_config['interaction_timesteps'] 
    interaction_directions = robot_config['interaction_directions']

    env_kwargs = dict(
        robots=robot_config['name'],
        object_name = "cabinet-1",
        # object_name='dishwasher-2', 
        # object_type='dishwasher', 
        obj_rotation=(-np.pi/2, -np.pi/3),
        scale_object = True,
        object_scale = 0.5,
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
        camera_widths = [1024,1024,1024,1024],
        move_robot_away = False,
        x_range = (0.5,0.5),
        y_range = (0.5, 0.5), 
        cache_video = True
    )
    env_name = "RobotPrismaticEnv"
    # env_name = 'RobotRevoluteOpening'

    env = suite.make(
        env_name,
        **env_kwargs
    )

    obs = env.reset()


    # set the open percentage
    env.set_open_percentage(0.4)


    robot_gripper_pos = obs["robot0_eef_pos"]

    # get the joint qpos
    reset_joint_qpos = env.sim.data.qpos[env.slider_qpos_addr]

    # sim_utils.init_camera_pose(env, camera_pos=np.array([-0.77542536, -0.02539806,  0.30146208]), scale_factor=3) 

    # if want to visualize the point cloud in open3d, the environment later will not work
    viz_imgs = False
    need_o3d_viz = False
    run_cgn = True

    pcd_wf_path = f'point_clouds/world_frame_pointclouds/world_frame_{env_kwargs["object_name"]}.ply'
    pcd_wf_no_downsample_path = f'point_clouds/world_frame_pointclouds/world_frame_{env_kwargs["object_name"]}_no_downsample.ply'
    camera_info_path = f'infos/camera_info_{env_kwargs["object_name"]}.npz'


    camera_euler = np.array([ 45., 22., 3.])
    camera_euler_for_pos = np.array([camera_euler[-1], camera_euler[1], -camera_euler[0]])
    camera_rotation = R.from_euler("xyz", camera_euler_for_pos, degrees=True)
    camera_quat = R.from_euler("xyz", camera_euler, degrees=True).as_quat()
    camera_forward = np.array([1, 0, 0])
    world_forward = camera_rotation.apply(camera_forward)
    distance = 1.5
    camera_pos = -distance * world_forward

    reset_x_range = (1,1)
    reset_y_range = (1,1)

    pcd_wf_path, pcd_wf_no_downsample_path,camera_info_path = get_aograsp_ply_and_config(env_name = env_name, 
                            env_kwargs=env_kwargs,
                            object_name=env_kwargs["object_name"], 
                            # camera_pos=np.array([-0.77542536, -0.02539806,  0.30146208]),
                            camera_pos=camera_pos,
                            camera_quat=camera_quat,
                            scale_factor=1,
                            pcd_wf_path=pcd_wf_path,
                            pcd_wf_no_downsample_path=pcd_wf_no_downsample_path,
                            camera_info_path=camera_info_path,
                            viz=viz_imgs, 
                            need_o3d_viz=need_o3d_viz,
                            reset_joint_qpos=reset_joint_qpos,
                            reset_x_range = reset_x_range, 
                            reset_y_range = reset_y_range,
                            )

    # get the affordance heatmap
    pcd_cf_path, affordance_path = get_affordance_main(pcd_wf_path, camera_info_path, 
                                                    viz=False)


    # get the grasp proposals
    world_frame_proposal_path, top_k_pos_wf, top_k_quat_wf = get_grasp_proposals_main(
        pcd_cf_path, 
        affordance_path, 
        camera_info_path, 
        run_cgn=run_cgn, 
        viz=need_o3d_viz, 
        save_wf_pointcloud=True,
        object_name=env_kwargs["object_name"],
        top_k=-1,
    )

    object_pos = np.array(env.obj_pos)

    # GAMMA STUFF
    # TODO: REMOVE IT
    # get the joint parameters
    # joint_poses, joint_directions, joint_types, resuts_list = get_joint_param_main(
    #     pcd_wf_no_downsample_path, 
    #     camera_info_path, 
    #     viz=need_o3d_viz, 
    #     return_results_list=True)
    # object_pos = np.array(env.obj_pos)
    # # find the first joint in the list with joint_type = 1
    # joint_pose_selected = None
    # joint_direction_selected = None
    # for i in range(len(resuts_list)):
    #     if joint_types[i] == 1:
    #         joint_pose_selected = joint_poses[i]
    #         joint_direction_selected = joint_directions[i]
    #         break
    # # if not found, use the first joint
    # if joint_pose_selected is None:
    #     joint_pose_selected = joint_poses[0]
    #     joint_direction_selected = joint_directions[0]
    # joint_direction_selected = joint_direction_selected / np.linalg.norm(joint_direction_selected)
    # joint_direction_selected = -joint_direction_selected 

    # print('GAMMA joint direction selected: ', joint_direction_selected)
    ## GAMMA stuff ends

    obs_dim = 6
    action_dim = 3
    lr = 0.001
    max_action = float(5)


    # load the trained policy
    # policy_dir = 'checkpoints/force_policies/prismatic_td3/run_0/curriculum_5'
    # policy_name = 'prismatic_td3_0_curriculum_5' 

    prismatic_policy_dir = f'checkpoints/force_policies/{experiment_name}/prismatic/{run_id}' 
    prismatic_policy_name = f'curriculum_0_ep_1000' 
    revolute_policy_dir = f'checkpoints/force_policies/{experiment_name}/revolute/{run_id}' 
    revolute_policy_name = f'curriculum_5_ep_1000'
    prismatic_policy = TD3(lr, obs_dim, action_dim, max_action)
    revolute_policy = TD3(lr, obs_dim, action_dim, max_action) 

    prismatic_policy.load(prismatic_policy_dir, prismatic_policy_name) 
    revolute_policy.load(revolute_policy_dir, revolute_policy_name)

    top_k_pos_wf = top_k_pos_wf + object_pos
    # top_k_pos_wf = top_k_pos_wf + np.array([0, 0.1, 0.])

    grasp_pos = top_k_pos_wf[1]
    grasp_quat = top_k_quat_wf[1] 

    final_grasp_pos, rotation_vector, obs = get_close_to_grasp_point(env, grasp_pos, grasp_quat, robot_gripper_pos)

    done = False
    last_grasp_pos = final_grasp_pos
    obs, estimation_dict = interact_estimate_params(
        env=env, 
        interaction_kp=kp_interaction, 
        interaction_action_scale=action_scale, 
        interaction_timesteps=interaction_timesteps, 
        rotation_vector=rotation_vector, 
        final_grasp_pos=final_grasp_pos, 
        obs=obs, 
        directions=interaction_directions
        )

    # obs = np.concatenate([joint_direction_selected, np.array(obs["robot0_eef_pos"]) - final_grasp_pos]) 
    state = get_state(obs, estimation_dict, final_grasp_pos) 
    print('ip estimated joint direction: ', estimation_dict['joint_direction']) 
    print('gt joint direction: ', obs['joint_direction'])
    # print('ground truth joint direction: ', env.hinge_direction if env.is_prismatic)

    # print(obs.keys()) 
    # obs = revolute_state(obs) if not env.is_prismatic else prismatic_state(obs)

    if estimation_dict['joint_type'] == 'prismatic' and env.is_prismatic: 
        classification_correct = True
        print('joint classfication correct') 
    elif estimation_dict['joint_type'] == 'revolute' and not env.is_prismatic:
        classification_correct = True
        print('joint classfication correct') 
    else:
        classification_correct = False
        env.save_video(f'videos/ip_failure_video_{run_id}.mp4') 
        env.close()
        return 0, 0
        # print('joint classfication incorrect')

    if estimation_dict['joint_type'] == 'prismatic':
        policy = prismatic_policy
    else:
        policy = revolute_policy

    for i in range(max_timesteps):
        # action = np.zeros_like(env.action_spec[0]) 
        action = policy.select_action(state)
        action = action.flatten()
        action = action * action_scale 
        # action = -action if estimation_dict['joint_type'] == 'revolute' else action
        # action = 

        # print('action: ', action)
        action = np.concatenate([last_grasp_pos + action,rotation_vector, [1]]) 
        next_obs, reward, done, _ = env.step(action)
        # if env._check_success():
        #     print('success!!')
        if done or env._check_success():
            break
        last_grasp_pos = next_obs['robot0_eef_pos']
        state = get_state(next_obs, estimation_dict, final_grasp_pos)
        # next_obs = np.concatenate([joint_direction_selected, np.array(next_obs["robot0_eef_pos"]) - final_grasp_pos]) 
        # next_obs = revolute_state(next_obs) if not env.is_prismatic else prismatic_state(next_obs)
        # obs = next_obs
        # print(obs)

    # print('success' if env._check_success() else 'failed')
    success = env._check_success()

    if not success:
        env.save_video(f'videos/failure_test_video_{run_id}.mp4')
    # env.save_video('videos/test_video.mp4')
    print('gt joint direction: ', obs['joint_direction'])
    env.close()
    return classification_correct, success

if __name__ == '__main__':
    ss = []
    for i in range(1):
        classification_correct, success = test(i)
        ss.append(success)
    print(np.mean(ss))

