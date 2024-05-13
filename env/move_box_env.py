import numpy as np

import os 
os.environ['MUJOCO_GL'] = 'osmesa'
from copy import deepcopy

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

class MoveBoxEnv(MujocoEnv):
    def __init__(self, 
                
                 table_full_size=(0.8, 0.8, 0.05),
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
                 renderer_config=None):
        self.has_renderer = has_renderer
        self.has_offscreen_renderer = has_offscreen_renderer
        self.use_camera_obs = use_camera_obs

        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

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
        self._action_dim = 6
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
    
    @staticmethod
    def _input2list(inp, length):
        """
        Helper function that converts an input that is either a single value or a list into a list

        Args:
            inp (None or str or list): Input value to be converted to list
            length (int): Length of list to broadcast input to

        Returns:
            list: input @inp converted into a list of length @length
        """
        # convert to list if necessary
        return list(inp) if type(inp) is list or type(inp) is tuple else [inp for _ in range(length)]
    
    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction, 
            table_offset=self.table_offset
        )
        mujoco_arena.set_origin([0, 0, 0])
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cube = BoxObject(
            name="cube",
            size_min=[0.020, 0.020, 0.020],  # [0.015, 0.015, 0.015],
            size_max=[0.022, 0.022, 0.022],  # [0.018, 0.018, 0.018])
            # size_min=[0.20, 0.20, 0.20],  # [0.015, 0.015, 0.015],
            # size_max=[0.22, 0.22, 0.22],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube,
                x_range=[-0.3, 0.3],
                y_range=[-0.3, 0.3],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
        self.model = ManipulationTask(
            mujoco_robots=[],
            mujoco_arena=mujoco_arena,
            mujoco_objects=self.cube,
        )
        

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)

    def _setup_observables(self):
        observables = super()._setup_observables() 
        if self.use_camera_obs:
            raise NotImplementedError("Camera observations are not supported for this environment")
        # setup cube pose as observable
        modality = "object"
        @sensor(modality=modality)
        def cube_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.cube_body_id])

        @sensor(modality=modality)
        def cube_quat(obs_cache):
            return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")
        
        sensors = [cube_pos, cube_quat]
        names = [s.__name__ for s in sensors]
        for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )
        return observables

        
    def _create_camera_sensors(self, cam_name, cam_w, cam_h, cam_d, cam_segs, modality="image"):
        convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]

        sensors = []
        names = []
        rgb_sensor_name = f"{cam_name}_image"
        depth_sensor_name = f"{cam_name}_depth"
        segmentation_sensor_name = f"{cam_name}_segmentation"

        @sensor(modality=modality)
        def camera_rgb(obs_cache):
            img = self.sim.render(
                camera_name=cam_name,
                width=cam_w,
                height=cam_h,
                depth=cam_d,
            )
            if cam_d:
                rgb, depth = img
                obs_cache[depth_sensor_name] = np.expand_dims(depth[::convention], axis=-1)
                return rgb[::convention]
            else:
                return img[::convention]
            
        sensors.append(camera_rgb)
        names.append(rgb_sensor_name)

        if cam_d:

            @sensor(modality=modality)
            def camera_depth(obs_cache):
                return obs_cache[depth_sensor_name] if depth_sensor_name in obs_cache else np.zeros((cam_h, cam_w, 1))

            sensors.append(camera_depth)
            names.append(depth_sensor_name)
        if cam_segs is not None:
            raise NotImplementedError("Segmentation sensors are not supported for this environment")
        
        return sensors, names
        

    def _create_segementation_sensor(self, cam_name, cam_w, cam_h, cam_s, seg_name_root, modality="image"):
        pass

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
                

    def _pre_action(self, action, policy_step=False):
         
         assert len(action) == self.action_dim, "environment got invalid action dimension -- expected {}, got {}".format(
            self.action_dim, len(action)
        )
        #  TODO: check action format
         self.sim.data.xfrc_applied[self.cube_body_id] = action

    def _check_success(self):
        """
        Runs superclass method by default
        """
        # TODO: do success check
        pass

    def reward(self, action=None):
        pos = self.sim.data.body_xpos[self.cube_body_id]
        return 1-np.linalg.norm(pos - np.array([0.3, 0.3, 0.3]))
        # return None


