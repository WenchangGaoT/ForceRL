import argparse
import json 

import numpy as np

import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_controller_config
from robosuite.renderers import load_renderer_config
from robosuite.utils.input_utils import * 
from robosuite.utils.transform_utils import quat2mat
# from env.robot_env_cube_move import RobotCubeMove
# from env.robot_drawer_opening import RobotDrawerOpening 
from env.curri_door_env import CurriculumDoorEnv
from env.robot_revolute_env import RobotRevoluteOpening
from env.drawer_opening import DrawerOpeningEnv 
from env.original_door_env import OriginalDoorEnv 
from env.camera_door_env import CameraDoorEnv 

import matplotlib.pyplot as plt 
from scipy.spatial.transform import Rotation as R
# from 


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.") 
    
def set_camera_pose(env, camera_name, position, quaternion):
    sim = env.sim
    cam_id = sim.model.camera_name2id(camera_name)
    sim.model.cam_pos[cam_id] = position
    sim.model.cam_quat[cam_id] = quaternion
    sim.forward() 

def display_camera_pose(env, camera_name):
    cam_id = env.sim.model.camera_name2id(camera_name)
    print(f'{camera_name} pose: {env.sim.model.cam_pos[cam_id]}') 
    print(f'{camera_name} quat: {env.sim.model.cam_quat[cam_id]}') 



if __name__ == "__main__":

    """
    Registered environments: Lift, Stack, NutAssembly, NutAssemblySingle, NutAssemblySquare, NutAssemblyRound,
                             PickPlace, PickPlaceSingle, PickPlaceMilk, PickPlaceBread, PickPlaceCereal,
                             PickPlaceCan, Door, Wipe, TwoArmLift, TwoArmPegInHole, TwoArmHandover

    Possible robots: Baxter, IIWA, Jaco, Kinova3, Panda, Sawyer, UR5e
    """

    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--renderer", type=str, default="mujoco", help="Valid options include mujoco, and nvisii")

    args = parser.parse_args()
    renderer = args.renderer

    # options["env_name"] = choose_environment()
    options['env_name'] = 'RobotRevoluteOpening'

    # If a multi-arm environment has been chosen, choose configuration and appropriate robot(s)
    if "TwoArm" in options["env_name"]:
        # Choose env config and add it to options
        options["env_configuration"] = choose_multi_arm_config()

        # If chosen configuration was bimanual, the corresponding robot must be Baxter. Else, have user choose robots
        if options["env_configuration"] == "bimanual":
            options["robots"] = "Baxter"
        else:
            options["robots"] = []

            # Have user choose two robots
            print("A multiple single-arm configuration was chosen.\n")

            for i in range(2):
                print("Please choose Robot {}...\n".format(i))
                options["robots"].append(choose_robots(exclude_bimanual=True))

    # Else, we simply choose a single (single-armed) robot to instantiate in the environment
    else:
        options["robots"] = choose_robots(exclude_bimanual=True)

    # Choose controller
    controller_name = choose_controller()

    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller=controller_name) 
    options['render_camera']='sideview'

    # del options['robots'] 
    # del options['controller_configs'] 
    env = suite.make(
        **options,
        has_renderer=False if renderer != "mujoco" else True,  # no on-screen renderer
        has_offscreen_renderer=True,  # no off-screen renderer
        ignore_done=True,
        use_camera_obs=True,  # no camera observations
        control_freq=20,
        renderer=renderer,
    ) 

    obs = env.reset() 
    print('camera width and height: ', env.camera_widths, env.camera_heights)

    print('rotation matrix for [0.5, 0.5, 0.5, 0.5]: ') 
    m1 = quat2mat(np.array([-0.5, -0.5, 0.5, 0.5])) 
    print(m1)

    print('rotation matrix for quat: ') 
    m2 = quat2mat(np.array([-0.005696068282031459, 0.19181093598117868, 0.02913152799094475, 0.9809829120433564]))  
    # m2 = quat2mat(np.array([0, 0, 0, 1]))
    print(m2)

    M = np.dot(m1, m2) 
    quat = R.from_matrix(M).as_quat() 
    print('Corresponding quaternion: ', quat)

    
    obj_pos = env.obj_pos 
    camera_trans = 3*np.array([-0.77542536, -0.02539806,  0.30146208])
    # set_camera_pose(env, 'sideview', [-0.77542536, -0.02539806,  2.20146208], [-0.005696068282031459, 0.19181093598117868, 0.02913152799094475, 0.9809829120433564])
    set_camera_pose(env, 'sideview', obj_pos + camera_trans, quat) 
    display_camera_pose(env, 'frontview')
    display_camera_pose(env, 'sideview') 
    print(obs.keys())
    # plt.imshow(obs['sideview_depth'], cmap='gray')
    # plt.show()
    config = env._get_camera_config('birdview')
    config = {
        'data': config
    } 
    print(config) 
    np.savez('temp_info.npz', **config) 
    camera_name = "birdview"
    camera_id = env.sim.model.camera_name2id(camera_name) 

    obj_name = 'drawer_main' 
    obj_id = env.sim.model.body_name2id(obj_name) 
    env.sim.data.body_xquat[obj_id] = np.array([0, 0, 0, 1])
    env.sim.forward()
    print('drawer body id: ', obj_id)
    print('object pos: ', env.sim.model.body_pos[obj_id]) 
    print('object quat: ', env.sim.data.get_body_xquat(obj_name))

    print(f"Camera ID for '{camera_name}':", camera_id) 
    # env.sim.data.cam_xpos[camera_id] = np.array([0, 0, 3]) 
    # env.sim.forward()
    camera_pos = env.sim.data.cam_xpos[camera_id]  
    camera_quat = env.sim.model.cam_quat[camera_id] 

    print('bird view camera pos: ', camera_pos) 
    print('bird view camera quat: ', camera_quat)


    # print('Bodies: ')
    # for b_id, b_name in enumerate(env.sim.model.body_names):
    #     print(f'body {b_id}: {b_name}') 

    # print('geoms: ')
    # for g_id, g_name in enumerate(env.sim.model.geom_names):
    #     print(f'geom {g_id}: {g_name}')


    low, high = env.action_spec 
    obs, reward, done, _ = env.step(np.zeros_like(low)) 
    # print(obs['sideview_image'].shape)

    if renderer == "nvisii":

        timesteps = 300
        for i in range(timesteps):
            action = np.random.uniform(low, high )
            # action = np.zeros(3)
            obs, reward, done, _ = env.step(action)

            if i % 100 == 0:
                env.render()

    else:

        # do visualization
        for i in range(10000):
            action = np.random.uniform(low, high)
            # action = np.zeros(3)
            obs, reward, done, _ = env.step(action)
            env.render()

    env.close_renderer()
    print("Done.")
