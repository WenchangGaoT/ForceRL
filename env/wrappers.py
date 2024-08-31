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
                use_wrapped_reward=False):

        assert env.skip_object_initialization, "GraspStateWrapper only works with skip_object_initialization=True"
        
        super().__init__(env)

        self.env = env
        self.env_name = env.env_name
        self.grasp_state = None
        self.project_dir = env.project_dir
        self.env_model_dir = os.path.join(self.project_dir, "baselines/states")
        self.states_file_path = os.path.join(self.env_model_dir, f"{self.env.env_name}.h5")

        self.number_of_grasp_states = number_of_grasp_states

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
        print("loading grasp states")
        # load the grasp states
        data_dict = {}
        with h5py.File(self.states_file_path, 'r') as hf:
            for name in hf.keys():
                group = hf[name]
                # Load array and XML string
                array = group['state'][:]
                xml = group['model'][()].decode('utf-8')
                print(type(xml))
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
    
    def step(self, action):
        '''
        wrap the step function to add the grasp state
        '''
        obs, rwd, done, info = self.env.step(action)
        # self.wrap_observation(obs)
        if self.use_wrapped_reward:
            rwd = self.staged_reward_wrapper(self.env.get_stage(), obs["gripper_pos"])
        return obs, rwd, done, info

  

    
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

        dist = np.linalg.norm(gripper_pos - self.env.calc_absolute_grasp_pos())
        reward_end_effector = (1 - np.tanh(2.0 * dist)) * eef_mult
        # clip the reward so that it is not too strong
        reward_end_effector = np.clip(reward_end_effector, 0, 0.9)
        
        print("handle current progress: ", self.env.handle_current_progress)
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
   

