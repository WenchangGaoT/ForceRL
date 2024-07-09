# environment with the door model from original robosuite code
from copy import deepcopy

import numpy as np
import robosuite as suite
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
import robosuite.macros as macros
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.transform_utils import convert_quat


from robosuite.environments.base import MujocoEnv
import mujoco

class OriginalDoorEnv(MujocoEnv):
    def __init__(self, 
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
                 renderer_config=None): 
        
        self.action_scale = action_scale
        self.has_renderer = has_renderer
        self.has_offscreen_renderer = has_offscreen_renderer
        self.use_camera_obs = use_camera_obs
        self.placement_initializer = placement_initializer

        super().__init__(
            has_renderer=self.has_renderer,
            has_offscreen_renderer=self.has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            renderer=renderer,
            renderer_config=renderer_config
        )

        self.horizon = horizon 
        self._action_dim = 3

        # Set camera attributes
        self.camera_names =  (
            list(camera_names) if type(camera_names) is list or type(camera_names) is tuple else [camera_names]
        )
        self.num_cameras = len(self.camera_names)
        self.camera_heights = self._input2list(camera_heights, self.num_cameras)
        self.camera_widths = self._input2list(camera_widths, self.num_cameras)
        self.camera_depths = self._input2list(camera_depths, self.num_cameras)
        self.camera_segmentations = self._input2list(camera_segmentations, self.num_cameras)

        seg_is_nested = False
        for i, camera_s in enumerate(self.camera_segmentations):
            if isinstance(camera_s, list) or isinstance(camera_s, tuple):
                seg_is_nested = True
                break
        camera_segs = deepcopy(self.camera_segmentations)
        for i, camera_s in enumerate(self.camera_segmentations):
            if camera_s is not None:
                self.camera_segmentations[i] = self._input2list(camera_s, 1) if seg_is_nested else deepcopy(camera_segs)
        if self.use_camera_obs and not self.has_offscreen_renderer:
            raise ValueError("Error: Camera observations require an offscreen renderer!")
        if self.use_camera_obs and self.camera_names is None:
            raise ValueError("Must specify at least one camera name when using camera obs")
        
        # Set up backup renderer
        self.backup_renderer = mujoco.Renderer(self.sim.model._model)

    def visualize(self, vis_settings):
        return super().visualize(vis_settings=vis_settings)

    @property
    def _visualizations(self):
        """
        Visualization keywords for this environment

        Returns:
            set: All components that can be individually visualized for this environment
        """
        vis_settings = super()._visualizations
        return vis_settings
    
    @property
    def action_spec(self):
        '''
        bounds for the action space
        '''
        # TODO: check if this is correct
        low, high = np.concatenate(np.ones(3),np.zeros(3)).astype(float) , np.concatenate(np.ones(3),np.zeros(3)).astype(float)
        return low, high
    
    @property
    def action_dim(self):
        """
        Size of the action space

        Returns:
            int: Action space dimension
        """
        return self._action_dim
