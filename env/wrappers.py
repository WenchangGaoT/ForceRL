from .drawer_opening import DrawerOpeningEnv


class ActionRepeatWrapper(DrawerOpeningEnv):
    def __init__(self, 
                 table_full_size=(0.8, 0.3, 0.05),
                 table_friction=(1.0, 5e-3, 1e-4),
                 use_camera_obs=False, 
                 placement_initializer=None,
                 has_renderer=True, 
                 has_offscreen_renderer=False,
                 render_camera="frontview",
                 render_collision_mesh=False,
                 render_visual_mesh=True,
                 render_gpu_device_id=-1,
                 control_freq=20, 
                 horizon=1000,
                 ignore_done=False,
                 hard_reset=True,
                 camera_names="agentview",
                 camera_heights=256,
                 camera_widths=256,
                 camera_depths=False,
                 camera_segmentations=None,
                 renderer="mujoco",
                 action_scale=2,
                 renderer_config=None, 
                 action_repeat=4): 
        
        super().__init__(
                 table_full_size=table_full_size,
                 table_friction=table_friction,
                 use_camera_obs=use_camera_obs, 
                 placement_initializer=placement_initializer,
                 has_renderer=has_renderer, 
                 has_offscreen_renderer=has_offscreen_renderer,
                 render_camera=render_camera,
                 render_collision_mesh=render_collision_mesh,
                 render_visual_mesh=render_visual_mesh,
                 render_gpu_device_id=render_gpu_device_id,
                 control_freq=control_freq, 
                 horizon=horizon,
                 ignore_done=ignore_done,
                 hard_reset=hard_reset,
                 camera_names=camera_names,
                 camera_heights=camera_heights,
                 camera_widths=camera_widths,
                 camera_depths=camera_depths,
                 camera_segmentations=camera_segmentations,
                 renderer=renderer,
                 action_scale=action_scale,
                 renderer_config=renderer_config)
        
        self.action_repeat = action_repeat
        # self.curr_action_ = 0 
        # self.last_action = None

    # def _reset_internal(self):
    #     self.curr_action_ = 0
    #     self.last_action = None
    #     return super()._reset_internal() 
    
    def step(self, action): 
        rwd = 0
        for _ in range(self.action_repeat):
            n_state, n_rwd, done, info = super().step(action) 
            rwd += n_rwd
            if done:
                break
        return n_state, rwd, done, info
        
