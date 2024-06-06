import numpy as np

import os 
os.environ['MUJOCO_GL'] = 'osmesa'
from copy import deepcopy

import robosuite as suite
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
import robosuite.macros as macros
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.models.arenas import EmptyArena
from objects.custom_objects import DrawerObject
from robosuite.utils import OpenCVRenderer
from robosuite.utils.binding_utils import MjRenderContextOffscreen
from utils.renderer_modified import MjRendererForceVisualization
from robosuite.utils.transform_utils import convert_quat
import copy


from robosuite.environments.base import MujocoEnv
import mujoco
import cv2

class DrawerOpeningEnv(MujocoEnv):
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
                 renderer_config=None): 
        self.action_scale = action_scale
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
        self._action_dim = 3
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

        # set empty arena
        mujoco_arena = EmptyArena()

        # get the drawer object
        self.drawer = DrawerObject(
            name="drawer" 
        )

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.drawer)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.drawer,
                x_range=[0, 0], #No randomization
                y_range=[-0.14,-0.14], #No randomization
                rotation=(-np.pi/2.0 , -np.pi/2.0 ), #No randomization
                rotation_axis="z",
                ensure_object_boundary_in_range=False, 
                reference_pos=(-0.6, -1.0, 0.5)
            )

        # Create task
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[],
            mujoco_objects=[self.drawer])
    
    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.object_body_ids = dict()
        self.object_body_ids["drawer"] = self.sim.model.body_name2id(self.drawer.drawer_body)
        self.object_body_ids['drawer_handle_body'] = self.sim.model.body_name2id('drawer_handle_body')
        self.drawer_handle_site_id = self.sim.model.site_name2id(self.drawer.important_sites["handle"])
        self.slider_qpos_addr = self.sim.model.get_joint_qpos_addr(self.drawer.joints[0])
        self.handle_geom_name = "drawer_handle"
        self.render_arrow_end = np.array(self.sim.data.get_geom_xpos(self.handle_geom_name))


    # def _setup_observables(self):
    #     observables = super()._setup_observables() 
    #     modality = "object"
    #     return observables
      
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
        # create visualization screen or renderer
        if self.has_renderer and self.viewer is None:
            self.viewer = OpenCVRenderer(self.sim)

            # Set the camera angle for viewing
            if self.render_camera is not None:
                camera_id = self.sim.model.camera_name2id(self.render_camera)
                self.viewer.set_camera(camera_id)

        if self.has_offscreen_renderer:
            if self.sim._render_context_offscreen is None:
                render_context = MjRendererForceVisualization(self.sim, modify_fn=self.modify_scene,device_id=self.render_gpu_device_id)
                # render_context = MjRenderContextOffscreen(self.sim, device_id=self.render_gpu_device_id)
            self.sim._render_context_offscreen.vopt.geomgroup[0] = 1 if self.render_collision_mesh else 0
            self.sim._render_context_offscreen.vopt.geomgroup[1] = 1 if self.render_visual_mesh else 0

        # additional housekeeping
        self.sim_state_initial = self.sim.get_state()
        self._setup_references()
        self.cur_time = 0
        self.timestep = 0
        self.done = False

        # Empty observation cache and reset all observables
        self._obs_cache = {}
        for observable in self._observables.values():
            observable.reset()

        
        object_placements = self.placement_initializer.sample() 
        drawer_pos, drawer_quat, _ = object_placements[self.drawer.name]
        drawer_body_id = self.sim.model.body_name2id(self.drawer.root_body)
        self.sim.model.body_pos[drawer_body_id] = drawer_pos
        self.sim.model.body_quat[drawer_body_id] = drawer_quat
        self.handle_current_progress = self.sim.data.qpos[self.slider_qpos_addr]
        self.render_arrow_end = np.array(self.sim.data.get_geom_xpos(self.handle_geom_name))
        self.last_handle_xpos = self.sim.data.get_geom_xpos(self.handle_geom_name)

                
    def _pre_action(self, action, policy_step=False): 
        self.last_handle_pos = self.sim.data.qpos[self.slider_qpos_addr]
        # action = action/(np.linalg.norm(action)+1e-6) * self.action_scale
        self.render_arrow_end = np.array(self.sim.data.get_geom_xpos(self.handle_geom_name)) + np.array(action)
        assert len(action) == self.action_dim, "environment got invalid action dimension -- expected {}, got {}".format(
            self.action_dim, len(action)
        )
        #  TODO: check action format
        #  self.sim.data.xfrc_applied[self.object_body_ids['drawer_handle_body']] = action
        self.sim.data._data.qfrc_applied = [0]
        #  point = self.sim.data.body_xpos[self.object_body_ids['drawer_handle_body']]
        point = self.sim.data.get_geom_xpos(self.handle_geom_name)
        mujoco.mj_applyFT(self.sim.model._model, self.sim.data._data, action, np.zeros(3), point, self.object_body_ids['drawer_handle_body'], self.sim.data._data.qfrc_applied)
         
    def _check_success(self):
        """
        Runs superclass method by default
        """
        # TODO: do success check
        slider_qpos = self.sim.data.qpos[self.slider_qpos_addr]
        # print(slider_qpos > 0.3)
        return slider_qpos > 0.3

    def reward(self, action=None):
        pos = self.sim.data.body_xpos[self.object_body_ids['drawer_handle_body']]
        # return float(self._check_success())
        if self._check_success(): 
            return 10
        else:
            progress_reward = 1000 * (np.linalg.norm(self.last_handle_pos-0.3)-np.linalg.norm(self.sim.data.qpos[self.slider_qpos_addr]-0.3))
            current_handle_xpos = copy.deepcopy(self.sim.data.get_geom_xpos(self.handle_geom_name))
            # print("current_handle_xpos: ", current_handle_xpos)
            # print("self.last_handle_xpos: ", self.last_handle_xpos)
            # print()
            delta_handle_xpos = 1000* (current_handle_xpos - self.last_handle_xpos) 
            # print('delta_handle_xpos: ', delta_handle_xpos)
            self.last_handle_xpos = current_handle_xpos
            # delta_handle_pos = delta_handle_pos / (np.linalg.norm(delta_handle_pos) + 1e-7)
            valid_force_reward = np.dot(action, delta_handle_xpos) / (np.linalg.norm(action) + 1e-7)
            valid_force_reward  = valid_force_reward * np.sign(progress_reward)
            # print(action)

            # self.last_handle_xpos = current_handle_xpos
            # print(np.dot(action, delta_handle_xpos))
            reward = valid_force_reward + progress_reward

            # print(reward)
            return reward
            # return 1000 * (np.linalg.norm(self.last_handle_pos-0.3)-np.linalg.norm(self.sim.data.qpos[self.slider_qpos_addr]-0.3))
        # return 1-np.linalg.norm(self.sim.data.qpos[self.slider_qpos_addr]-0.3) if not self._check_success() else 1.0
        # return None

    def modify_scene(self, scene):
        rgba = np.array([0.5, 0.5, 0.5, 1.0])
        # point1 = self.sim.data.body_xpos[self.object_body_ids['drawer_handle_body']]
        point1 = self.sim.data.get_geom_xpos(self.handle_geom_name)
        # point2 = np.array([1.0, 1.0, 1.0])
        point2 = self.render_arrow_end
        radius = 0.05
        if scene.ngeom >= scene.maxgeom:
            return
        scene.ngeom += 1
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                      mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                      np.zeros(3), np.zeros(9), rgba.astype(np.float32))
        mujoco.mjv_connector(scene.geoms[scene.ngeom-1],
                           int(mujoco.mjtGeom.mjGEOM_ARROW), radius,
                           point1, point2)
        
    def render(self):
        super().render() 
    
    def _setup_observables(self):
        observables = super()._setup_observables() 
        if self.use_camera_obs:
            raise NotImplementedError("Camera observations are not supported for this environment")
        # setup cube pose as observable
        modality = "object"
        # @sensor(modality=modality)
        # def cube_pos(obs_cache):
        #     return np.array(self.sim.data.body_xpos[self.cube_body_id])

        @sensor(modality=modality)
        def handle_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.drawer_handle_site_id])

        # @sensor(modality=modality)
        # def cube_quat(obs_cache):
        #     return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")
        
        @sensor(modality=modality) 
        def handle_quat(obs_cache):
            return convert_quat(np.array(self.sim.data.body_xquat[self.drawer_handle_site_id]), to="xyzw")
        
        sensors = [handle_pos, handle_quat]
        names = [s.__name__ for s in sensors]
        for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )
        return observables