from robosuite.wrappers import Wrapper
import os
from utils import baseline_utils as b_utils
import xml.etree.ElementTree as ET
import numpy as np
import h5py
 

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
        
        self.get_grasp_states()




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
                grasp_states_save_path = self.env.grasp_states_path,
                env_model_dir = self.env_model_dir,
                object_rotation_range = self.env.obj_rotation,
                object_robot_distance_range=self.env.object_robot_distance,
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
          
    def sample_state(self):
        '''
        function for sampling a state
        '''
        key = np.random.choice(list(self.data_dict.keys()))
        state, model = self.data_dict[key]
        return state, model

    def reset(self):
        '''
        wrap the reset function to load a state
        '''
        # sample a state
        state, model = self.sample_state()
        # load the state
        self.env.reset_from_xml_string(model)
        self.env.sim.set_state_from_flattened(state)
        self.env.sim.forward()
        return self.env._get_observations()
        

