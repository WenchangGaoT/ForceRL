# environment with the door model from original robosuite code
# Issue: the self.last_force_point_xpos is not correctly reset, however in normal steps it is corret.
from copy import deepcopy

import numpy as np
import robosuite as suite
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
import robosuite.macros as macros
from robosuite.models.arenas import EmptyArena
from objects.custom_objects import OriginalDoorObject as DoorObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.transform_utils import convert_quat


from robosuite.environments.base import MujocoEnv
import mujoco

from robosuite.utils import OpenCVRenderer
from robosuite.utils.binding_utils import MjRenderContextOffscreen
from utils.renderer_modified import MjRendererForceVisualization
from scipy.spatial.transform import Rotation as R

class OriginalDoorEnv(MujocoEnv):
    def __init__(self, 
                 use_camera_obs=True, 
                 placement_initializer=None,
                 random_force_point = False,
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
                 action_scale=1,
                 reward_scale=10,
                 renderer_config=None, 
                 debug_mode=False, 

                 # Camera settings in case
                 agentview_camera_pos=[0.5986131746834771, -4.392035683362857e-09, 1.5903500240372423], 
                 agentview_camera_quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]): 
        
        self.action_scale = action_scale
        self.has_renderer = has_renderer
        self.has_offscreen_renderer = has_offscreen_renderer
        self.use_camera_obs = use_camera_obs
        self.placement_initializer = placement_initializer
        self.debug_mode = debug_mode 

        ## AO-Grasp required information
        self.agentview_camera_pos = agentview_camera_pos 
        self.agentview_camera_quat = agentview_camera_quat 

        self.camera_cfgs = {
            'agentview': {
                'trans': np.array(self.agentview_camera_pos), 
                'quat': np.array(self.agentview_camera_quat)
            }
        }


        ################################################################################################

        self.use_object_obs = True # always use low-level object obs for this environment
        self.random_force_point = random_force_point 

        
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
        self.reward_scale = reward_scale
        

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
        # self.backup_renderer = mujoco.Renderer(self.sim.model._model)

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

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=self.agentview_camera_pos,
            quat=self.agentview_camera_quat,
        )

        # initialize objects of interest
        self.door_name_prefix = "Door"
        self.door = DoorObject(
            name=self.door_name_prefix,
            friction=0.0,
            damping=0.0,
            lock=False, # we are not handling lathches for now
        )

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.door)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.door,
                x_range=[0.07, 0.09],
                y_range=[-0.01, 0.01],
                # rotation=(-np.pi / 2.0 - 0.25, -np.pi / 2.0),
                rotation=(-np.pi / 2.0 - 0.25, 0),

                # rotation=(-np.pi , np.pi),
                # rotation = (-np.pi, -np.pi),
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=(-0.2, -0.35, 0.8), # hard coded as original code
            )

        # Create task
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[],
            mujoco_objects=[self.door])
        
    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """

        super()._setup_references()
    
        # Additional object references from this env
        self.object_body_ids = dict()
        self.object_body_ids["door"] = self.sim.model.body_name2id(self.door.door_body)
        self.object_body_ids["frame"] = self.sim.model.body_name2id(self.door.frame_body)
        # self.object_body_ids["latch"] = self.sim.model.body_name2id(self.door.latch_body)
        self.door_handle_site_id = self.sim.model.site_name2id(self.door.important_sites["handle"])
        self.hinge_qpos_addr = self.sim.model.get_joint_qpos_addr(self.door.joints[0])
        
        self.panel_geom_name = self.door_name_prefix + "_panel"
        self.hinge_site_id = self.sim.model.site_name2id(self.door.important_sites["hinge"])
        self.hinge_position = self.sim.data.site_xpos[self.hinge_site_id]
        # hinge direction is normalized
        self.hinge_direction = self.door.hinge_direction / np.linalg.norm(self.door.hinge_direction)

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

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        if self.use_camera_obs:
            # Create sensor information
            sensors = []
            names = []
            for (cam_name, cam_w, cam_h, cam_d, cam_segs) in zip(
                self.camera_names,
                self.camera_widths,
                self.camera_heights,
                self.camera_depths,
                self.camera_segmentations,
            ):

                # Add cameras associated to our arrays
                cam_sensors, cam_sensor_names = self._create_camera_sensors(
                    cam_name, cam_w=cam_w, cam_h=cam_h, cam_d=cam_d, cam_segs=cam_segs, modality="image"
                )
                sensors += cam_sensors
                names += cam_sensor_names

            # If any the camera segmentations are not None, then we shrink all the sites as a hacky way to
            # prevent them from being rendered in the segmentation mask
            if not all(seg is None for seg in self.camera_segmentations):
                self.sim.model.site_size[:, :] = 1.0e-8

            # Create observables for these cameras
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            modality = "object"

            # Define sensor callbacks
            @sensor(modality=modality)
            def door_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.object_body_ids["door"]])

            @sensor(modality=modality)
            def handle_pos(obs_cache):
                return self._handle_xpos


            @sensor(modality=modality)
            def hinge_qpos(obs_cache):
                return np.array([self.sim.data.qpos[self.hinge_qpos_addr]])

            @sensor(modality=modality)
            def hinge_position(obs_cache):
                return self.hinge_position
            
            @sensor(modality=modality)
            def hinge_direction(obs_cache):
                return self.hinge_direction

            @sensor(modality=modality)
            def force_point(obs_cache):
                return self.relative_force_point_to_world(self.force_point)


            sensors = [door_pos, handle_pos, hinge_qpos, hinge_position, force_point, hinge_direction] 

            # if self.use_camera_obs:
            #     sensors.extend()
            names = [s.__name__ for s in sensors] 

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def sample_relative_force_point(self):
        '''
        sample a point on the door to apply force, the point is relative to the door frame
        '''
        if not self.random_force_point:
            return np.array([-0.2,0,0])
        else:
            panel_size = self.door.door_panel_size
            # random sample a point on the door panel
            x = np.random.uniform(-panel_size[0], panel_size[0] - 0.1)
            y = 0
            z = np.random.uniform(-panel_size[2], panel_size[2])
            return np.array([x,y,z])
    
    def relative_force_point_to_world(self, relative_force_point):
        '''
        convert the relative force point to world frame
        '''
        panel_pos = self.sim.data.get_geom_xpos(self.panel_geom_name)
        panel_rot = self.sim.data.get_geom_xmat(self.panel_geom_name)
        
        return deepcopy(panel_pos + np.dot(panel_rot, relative_force_point))


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

        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # We know we're only setting a single object (the door), so specifically set its pose
            door_pos, door_quat, _ = object_placements[self.door.name]
            door_body_id = self.sim.model.body_name2id(self.door.root_body)
            self.sim.model.body_pos[door_body_id] = door_pos
            self.sim.model.body_quat[door_body_id] = door_quat
        
        # sample a force point on the door
        self.force_point = self.sample_relative_force_point()
        self.force_point_world = deepcopy(self.relative_force_point_to_world(self.force_point))
        
        # get the initial render arrow end
        self.render_arrow_end = self.force_point_world

        # get the initial hinge qpos
        self.last_hinge_qpos = self.sim.data.qpos[self.hinge_qpos_addr]
        self.last_force_point_xpos = deepcopy(self.force_point_world)

        # reset force projection value
        self.current_step_force_projection = 0
        # print("initial force point: ", self.force_point_world)

    def _pre_action(self, action, policy_step=False):
        
        self.last_hinge_qpos = deepcopy(self.sim.data.qpos[self.hinge_qpos_addr])
        # get the arrow end position
        self.force_point_world = self.relative_force_point_to_world(self.force_point)
        self.render_arrow_end = self.force_point_world + action * self.action_scale

        assert len(action) == self.action_dim, "environment got invalid action dimension -- expected {}, got {}".format(
            self.action_dim, len(action))

        # manually set qfrc_applied to 0 to avoid the force applied in the last step
        self.sim.data._data.qfrc_applied = [0]

        # apply the force
        mujoco.mj_applyFT(self.sim.model._model, self.sim.data._data, 
                          action * self.action_scale, np.zeros(3), self.force_point_world, 
                          self.object_body_ids["door"], self.sim.data._data.qfrc_applied)

    def _check_success(self):
        # TODO:implement this
        hinge_qpos = self.sim.data.qpos[self.hinge_qpos_addr]
        # open 30 degrees
        return hinge_qpos > 0.45
    
    def reward(self, action = None):
        # TODO:implement this
        if self._check_success():
            # print("success")
            return 1
        else:
            
            current_force_point_xpos = deepcopy(self.relative_force_point_to_world(self.force_point))
            # print("current force point: ", current_force_point_xpos)
            # print("last force point: ", self.last_force_point_xpos)
            delta_force_point = current_force_point_xpos - self.last_force_point_xpos
            self.last_force_point_xpos = deepcopy(current_force_point_xpos)

            self.current_step_force_projection = np.dot(action, delta_force_point) / (np.linalg.norm(action) * np.linalg.norm(delta_force_point))
            # print("delta force point: ", delta_force_point)
            delta_force_point *= 1000
            valid_force_reward = np.dot(action, delta_force_point) / (np.linalg.norm(action) + 1e-7)
            valid_force_reward = np.abs(valid_force_reward)
            
            progress = self.sim.data.qpos[self.hinge_qpos_addr] - self.last_hinge_qpos
            
            # print("progress: ", progress)
            valid_force_reward = valid_force_reward * np.sign(progress)

            if np.abs(progress) < 1e-6:
                valid_force_reward = 0

            if self.debug_mode:
                # print the angle between the delta force point and the action
                print("angle between action and delta force point: ", np.arccos(np.dot(action, delta_force_point) / (np.linalg.norm(action) * np.linalg.norm(delta_force_point))) * 180 / np.pi)
            
            if np.isnan(valid_force_reward):
                print("nan reward")
            # print("valid force reward: ", valid_force_reward)
            return valid_force_reward * self.reward_scale
    
    @property
    def current_action_projection(self):
        return self.current_step_force_projection


    
    def modify_scene(self, scene):
        rgba = np.array([0.5, 0.5, 0.5, 1.0])

        # start and end point of the arrow
        point1 = self.force_point_world
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

    @property
    def _handle_xpos(self):
        """
        Grabs the position of the door handle handle.

        Returns:
            np.array: Door handle (x,y,z)
        """
        return self.sim.data.site_xpos[self.door_handle_site_id] 
    
    def display_cameras(self):
        print('-----------------------------------------')
        print(self.sim.model.camera_names)
    
    def _get_camera_config(self, camera_name):
        '''
        Get the information of cameras.
        Input:
            -camera_name: str, the name of the camera to be parsed 

        Returns:
            {
                
        }
        ''' 
        camera_id = self.sim.model.camera_name2id(camera_name)
        camera_pos = self.sim.data.cam_xpos[camera_id] 
        camera_rot_matrix = self.sim.data.cam_xmat[camera_id].reshape(3, 3)

        # Convert the rotation matrix to a quaternion
        camera_quat = R.from_matrix(camera_rot_matrix).as_quat()
        return {
            'camera_config': {
                'trans': camera_pos, 
                'quat': camera_quat
            }
        }
        