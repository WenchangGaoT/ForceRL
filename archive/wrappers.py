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

  def wrap_observation(self, obs):
        '''
        wrap the observation to add the grasp state
        '''
        obs["grasp_pos"] = self.calc_absolute_grasp_pos_wrapper()
        return obs
