from robosuite.wrappers import Wrapper
from robosuite.controllers.base_controller import Controller
import os
from utils import baseline_utils as b_utils
import numpy as np
import h5py
from robosuite.models.tasks import ManipulationTask
from objects.baseline_objects import BaselineTrainRevoluteObjects, BaselineTrainPrismaticObjects
from robosuite.models.arenas import EmptyArena
import json
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common import monitor, policies
from gymnasium import spaces
import robosuite as suite

 

class ActionRepeatWrapperNew(Wrapper):
    def __init__(self, env, action_repeat):
        super().__init__(env)
        self.action_repeat = action_repeat

    def step(self, action): 
        rwd = 0
        for _ in range(self.action_repeat):
            n_state, n_rwd, done, info = self.env.step(action)
            rwd += n_rwd
            if done:
                break
        return n_state, rwd, done, info

class GraspStateWrapper(Wrapper):
    def __init__(self, env,
                number_of_grasp_states=4,
                use_wrapped_reward=False,
                reset_joint_friction = 3.0,
                reset_joint_damping = 0.1,
                ):

        assert env.skip_object_initialization, "GraspStateWrapper only works with skip_object_initialization=True"
        
        super().__init__(env)

        self.env = env
        self.env_name = env.env_name
        self.grasp_state = None
        self.project_dir = env.project_dir
        self.env_model_dir = os.path.join(self.project_dir, "baselines/states")
        self.states_file_path = os.path.join(self.env_model_dir, f"{self.env.env_name}.h5")

        self.number_of_grasp_states = number_of_grasp_states
        self.reset_joint_friction = reset_joint_friction
        self.reset_joint_damping = reset_joint_damping

        # env_dir = os.path.dirname(os.path.abspath(__file__))
        # cfg_path = os.path.join(env_dir, "controller_configs/osc_pose_small_kp.json")
        # with open(cfg_path, "r") as f:
        #     self.temp_controller_configs = json.load(f)

        self.temp_kp = Controller.nums2array(2, 6)
        # self.temp_kp = np.array([2,2,2,2,2,2])
        self.close_gripper_action = [0,0,0,0,0,0,1]
        self.use_wrapped_reward = use_wrapped_reward
        self.load_all_objects()
        self.get_grasp_states()
        # print("Grasp state wrapper initialized")


    def load_all_objects(self):
        '''
        load all objects during initialization to avoid loading objects multiple times
        '''
        revolute_object_list = BaselineTrainRevoluteObjects.available_objects()
        prismatic_object_list = BaselineTrainPrismaticObjects.available_objects()

        objects = {}
        for obj_name in revolute_object_list:
            objects[obj_name] = BaselineTrainRevoluteObjects(obj_name)
        for obj_name in prismatic_object_list:
            objects[obj_name] = BaselineTrainPrismaticObjects(obj_name)
        
        self.objects = objects
        self.arena = EmptyArena()



    def get_grasp_states(self):
        '''
        function for getting the grasp states
        '''
        # check existance of npy and xml files

        
        if not os.path.exists(self.states_file_path):
            print("Getting grasp states")
            print("obj rotation: ", self.obj_rotation)
            b_utils.get_grasp_env_states(
                env_name = self.env_name,
                env_kwargs = self.env.env_kwargs,
                grasp_pos = self.env.final_grasp_pose,
                grasp_rot_vec = self.env.grasp_rotation_vector,
                env_model_dir = self.env_model_dir,
                object_rotation_range = self.env.obj_rotation,
                # object_robot_distance_range=self.env.object_robot_distance,
                object_robot_distance_range = (0.7,0.7),
                number_of_grasp_states = self.number_of_grasp_states,
            )
        # print("loading grasp states")
        # load the grasp states
        data_dict = {}
        with h5py.File(self.states_file_path, 'r') as hf:
            for name in hf.keys():
                group = hf[name]
                # Load array and XML string
                array = group['state'][:]
                xml = group['model'][()].decode('utf-8')
                # print(type(xml))
                # Store in the dictionary
                data_dict[name] = (array, xml)
        self.data_dict = data_dict
          
    def sample_state(self,key_num = None):
        '''
        function for sampling a state
        '''
        if key_num is not None:
            key = list(self.data_dict.keys())[key_num]
        else:
            key = np.random.choice(list(self.data_dict.keys()))
        state, model = self.data_dict[key]
        object_name = key.split("_")[0]
        return object_name,state, model

    def reset(self, key_num = None):
        '''
        wrap the reset function to load a state
        '''
        # sample a state
        object_name, state, model = self.sample_state(key_num=key_num)
        # load the state

        object_model = self.objects[self.env.object_name]
        self.env.model = ManipulationTask(
            mujoco_arena=self.arena,
            mujoco_robots=[robot.robot_model for robot in self.env.robots],
            mujoco_objects=self.objects[object_name]
        )

        # load the temporary controller configs for closing the gripper
        # self.env.robots[0].controller_configs = self.temp_controller_configs
        # self.env.robots[0]._load_controller()
        
        if "Prismatic" in self.env_name:
            self.env.prismatic_object = self.objects[object_name]
        else:
            self.env.revolute_object = self.objects[object_name]

        self.env.reset_from_xml_string(model)
        self.env.sim.set_state_from_flattened(state)
        self.env.sim.forward()

        self.set_joint_friction_and_damping(self.reset_joint_friction, self.reset_joint_damping)

        # load a small kp for closing the gripper
        self.robots[0].controller.kp = self.temp_kp
        # close the gripper
        for _ in range(10):
            self.env.step(self.close_gripper_action)

        # set the original kp
        self.robots[0].controller.kp = Controller.nums2array(self.env.robots[0].controller_config["kp"],6)

        # need to calculate the relative grasp position because object pose is not properly set when resetting
        # if use wrapper

        self.env.final_grasp_pose = self.env._get_observations()["gripper_pos"]
        self.env.grasp_pos_relative = self.env.calculate_grasp_pos_relative()

        return self.env._get_observations()
    
    def set_joint_friction_and_damping(self, friction = 3.0, damping = 0.1):
        '''
        set the joint friction and damping
        '''
        if "Prismatic" in self.env_name:
            self.env._set_drawer_friction(friction)
            self.env._set_drawer_damping(damping)
        else:
            # self.env.modder.update_sim(self.env.sim)
            self.env._set_door_friction(friction)
            self.env._set_door_damping(damping)

    def step(self, action):
        '''
        wrap the step function to add the grasp state
        '''
        obs, rwd, done, info = self.env.step(action)
        # self.wrap_observation(obs)
        if self.use_wrapped_reward:
            rwd = self.staged_reward_wrapper(self.get_stage_wrapper(), obs["gripper_pos"])
        return obs, rwd, done, info

    def get_stage_wrapper(self):

        '''
        wrap the get_stage function to make it more strict
        now, stage2 can only be achived when the drawer is fully open and the grasp is successful
        '''

        task_percentage = (self.env.handle_current_progress - self.joint_range[0]) / (self.joint_range[1] - self.joint_range[0])
        if not self.check_grasp():
            return 0
        elif task_percentage >= 0.8:
            return 2  
        # elif task_percentage >= 0.8:
        #     return 2
        else:
            return 1
        # elif self.check_grasp() and task_percentage < 0.8:
        #     return 1 

    
    def staged_reward_wrapper(self, stage, gripper_pos):
        '''
        need to wrap the reward function since the grasp pos changes in wrapper
        '''
        eef_mult = 1.0
        stage_mult = 1.0
        drawer_mult = 1.0

        # self.handle_current_progress = self.sim.data.qpos[self.slider_qpos_addr]
        # print("stage: ", stage)

        reward_stage_list = [0,1,2]
        reward_stage = reward_stage_list[stage] * stage_mult

        dist = np.linalg.norm(gripper_pos - self.env.calculate_grasp_pos_absolute())
        reward_end_effector = (1 - np.tanh(2.0 * dist)) * eef_mult
        # clip the reward so that it is not too strong
        reward_end_effector = np.clip(reward_end_effector, 0, 0.9)
        
        # print("handle current progress: ", self.env.handle_current_progress)
        reward_open = (self.env.handle_current_progress - self.env.joint_range[0]) / (self.env.joint_range[1] - self.env.joint_range[0])
        reward_open = reward_open * drawer_mult

        reward_stop = self.env._check_success() * 10

        if stage == 0:
            reward = reward_stage + reward_end_effector
        elif stage == 1:
            reward_end_effector = 0.9
            reward = reward_stage + reward_end_effector + reward_open
        else:
            reward_end_effector = 0.9
            reward = reward_stage + reward_end_effector + reward_open + reward_stop
        
        return reward
   

class GymWrapper(Wrapper, gym.Env):
    metadata = None
    render_mode = None
    def __init__(self, env, keys=None):
        super().__init__(env=env)
        # robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        # self.name = robots + "_" + type(self.env).__name__

        # Get reward range
        self.reward_range = (0, self.env.reward_scale)

        if keys is None:
            keys = []
            # Add object obs if requested
            if self.env.use_object_obs:
                keys += ["object-state"]
            # Add image obs if requested
            if self.env.use_camera_obs:
                keys += [f"{cam_name}_image" for cam_name in self.env.camera_names]
            # Iterate over all robots to add to state
            # for idx in range(len(self.env.robots)):
            #     keys += ["robot{}_proprio-state".format(idx)]
        self.keys = keys

        # Gym specific attributes
        self.env.spec = None

        # set up observation and action spaces
        obs = self.env.reset()
        self.modality_dims = {key: obs[key].shape for key in self.keys}
        flat_ob = self._flatten_obs(obs)
        self.obs_dim = flat_ob.size
        high = np.inf * np.ones(self.obs_dim)
        # high = np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high) 
        # print('action spec: ', self.env.action_spec)
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low, high)
        # print("action space: ", self.action_space)

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed

        Returns:
            np.array: observations flattened into a 1d array
        """
        ob_lst = []
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(ob_lst)

    def reset(self, seed=None, options=None):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict and optionally resets seed

        Returns:
            np.array: Flattened environment observation space after reset occurs
        """
        if seed is not None:
            if isinstance(seed, int):
                np.random.seed(seed)
            else:
                raise TypeError("Seed must be an integer type!")
        ob_dict = self.env.reset()
        return self._flatten_obs(ob_dict), {}

    def step(self, action):
        """
        Extends vanilla step() function call to return flattened observation instead of normal OrderedDict.

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (np.array) flattened observations from the environment
                - (float) reward from the environment
                - (bool) episode ending after reaching an env terminal state
                - (bool) episode ending after an externally defined condition
                - (dict) misc information
        """
        ob_dict, reward, terminated, info = self.env.step(action)
        return self._flatten_obs(ob_dict), reward, terminated, False, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()
    

class SubprocessEnv(gym.Env):
    def __init__(self, env_kwargs):
        self.env = suite.make(**env_kwargs)
        self.env = GymWrapper(self.env) 
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, seed=0):
        return self.env.reset(seed)

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        self.env.close()


def make_env(env_kwargs):
    def _init():
        # MuJoCo-related objects should be created here
        env = SubprocessEnv(env_kwargs)
        # env.seed(seed)
        return env
    return _init


def make_vec_env_baselines(env_name,
                           obs_keys, 
                           env_kwargs,
                           log_dir,
                           grasp_state_wrapper = False,
                           grasp_state_wrapper_kwargs = None,
                           n_envs = 10):
    '''
    make env for baselines
    '''
    # make a temp env for initializing all the point clouds and grasp states stuff
    tmp_env = suite.make(env_name,
                     **env_kwargs)
    
    if grasp_state_wrapper:
        tmp_env = GraspStateWrapper(tmp_env, **grasp_state_wrapper_kwargs)
    
    env = GymWrapper(tmp_env, keys=obs_keys)
    env.close()
    spec = tmp_env.spec

    def make_env_new():

        assert env_kwargs is not None
        env = suite.make(env_name,
                     **env_kwargs)
        
        if grasp_state_wrapper:
            env = GraspStateWrapper(env, **grasp_state_wrapper_kwargs)
        
        env = GymWrapper(env, keys=obs_keys)
        env.reset()
        env = monitor.Monitor(env=env, filename=log_dir, allow_early_resets=True)
        return env

    env_fns = [make_env_new for _ in range(n_envs)]
    return SubprocVecEnv(env_fns, start_method="forkserver")



    
    


class MultiProcessingParallelEnvsWrappeer:
    def __init__(self, env_kwargs, n_envs):
        self.env_kwargs = env_kwargs
        self.n_envs = n_envs
        # self.envs = [suite.make(**env_kwargs) for _ in range(n_envs)]
        self.envs = SubprocVecEnv([make_env(env_kwargs) for i in range(self.n_envs)])

    def reset(self, seed=0):
        return self.envs.reset() 
    
    def step(self, actions):
        return self.envs.step(actions) 
    
    def close(self):
        self.envs.close()
        # self.pool = Pool(n_envs)


        