

    # def relative_force_point_to_world(self, relative_force_point):
    #     '''
    #     convert the relative force point to world frame
    #     '''
    #     revolute_body_pos_abs = self.sim.data.get_body_xpos(self.revolute_object.revolute_body)
    #     # print("revolute body pos abs: ", revolute_body_pos_abs)
    #     revolute_body_rot_abs = self.sim.data.get_body_xmat(self.revolute_object.revolute_body)
    #     # print("revolute body rot abs: ", revolute_body_rot_abs)

    #     # calculate relative rotation with respect to relative zero
    #     revolute_body_rot = np.dot(revolute_body_rot_abs, self.revolute_body_rel_zero_rot.T) 
        
    #     # move and rotate relative to hinge position
    #     relative_force_position_to_hinge = np.dot(revolute_body_rot, relative_force_point)


    #     revolute_body_pos = revolute_body_pos_abs
    #     revolute_body_rot = revolute_body_rot_abs

    #     # compute the trainsition relative to 

    #     # print("np dot", np.dot(revolute_body_rot, relative_force_point))
        
    #     # return deepcopy(revolute_body_pos + np.dot(revolute_body_rot, relative_force_point))
    #     return deepcopy(self.hinge_position + relative_force_position_to_hinge)
