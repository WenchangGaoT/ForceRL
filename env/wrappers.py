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
                number_of_grasp_states=4,):

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

        self.robots[0].controller.kp = Controller.nums2array(self.env.robots[0].controller_config["kp"],6)

        self.calc_relative_grasp_pos_wrapper()

        return self.wrap_observation(self.env._get_observations())
    
    def step(self, action):
        '''
        wrap the step function to add the grasp state
        '''
        obs, rwd, done, info = self.env.step(action)
        self.wrap_observation(obs)
        return obs, rwd, done, info

    def wrap_observation(self, obs):
        '''
        wrap the observation to add the grasp state
        '''
        obs["grasp_pos"] = self.calc_absolute_grasp_pos_wrapper()
        return obs

    def calc_relative_grasp_pos_wrapper(self):
        '''
        function for calculating the relative grasp pose after gripping the object
        '''
        grasp_pos = self.env._get_observations()["gripper_pos"]
        
        if "Prismatic" in self.env_name:
            object_pos = self.env.prismatic_body_pos
            object_quat = self.env.prismatic_body_quat
        else:
            object_pos = self.env.revolute_body_pos
            object_mat = self.env.sim.data.get_body_xmat(self.env.revolute_object.revolute_body)
        
        grasp_pos = grasp_pos - object_pos
        self.grasp_pos_relative = np.dot(object_mat.T, grasp_pos)
        
    def calc_absolute_grasp_pos_wrapper(self):
        '''
        function for calculating the absolute grasp pose after gripping the object
        '''
        grasp_pos_relative = self.grasp_pos_relative
        if "Prismatic" in self.env_name:
            object_pos = self.env.prismatic_body_pos
            object_quat = self.env.prismatic_body_quat
        else:
            object_pos = self.env.revolute_body_pos
            object_mat = self.env.sim.data.get_body_xmat(self.env.revolute_object.revolute_body)
        
        grasp_pos = np.dot(object_mat, grasp_pos_relative)
        grasp_pos = grasp_pos + object_pos
        return grasp_pos
        

