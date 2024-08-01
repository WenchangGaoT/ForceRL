import numpy as np
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import EmptyArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat

from objects.custom_objects import SelectedMicrowaveObject 
from scipy.spatial.transform import Rotation as R

import os 
os.environ['MUJOCO_GL'] = 'osmesa'


class RobotRevoluteOpening(SingleArmEnv):

    def __init__(
        self,
        robots,
        object_type = "microwave",
        object_model_idx = 1,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
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
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        agentview_camera_pos=[0.5986131746834771, -4.392035683362857e-09, 1.5903500240372423], 
        agentview_camera_quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]
    ):
        self.placement_initializer = placement_initializer
        self.table_full_size = (0.8, 0.3, 0.05)
        self.available_objects = ["microwave"]
        assert object_type in self.available_objects, "Invalid object type! Choose from: {}".format(self.available_objects)
        self.object_type = object_type
        self.object_model_idx = object_model_idx

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
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
            renderer_config=renderer_config,
            # agentview_camera_pos=[0.5986131746834771, -4.392035683362857e-09, 1.5903500240372423], 
            # agentview_camera_quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]
        )
        
        self.reward_scale = reward_scale
        self.use_object_obs = use_object_obs
        self.reward_shaping = reward_shaping
        self.placement_initializer = placement_initializer

        
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
        
    

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # set robot position
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        xpos = (20, xpos[1], xpos[2]) # Move the robot away
        self.robots[0].robot_model.set_base_xpos(xpos)

        # set empty arena
        mujoco_arena = EmptyArena()

        # set camera pose
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5986131746834771, -4.392035683362857e-09, 1.5903500240372423],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349],
        )

        # get the revolute object
        if self.object_type == "microwave":
            self.revolute_object = SelectedMicrowaveObject(name="drawer", microwave_number=self.object_model_idx, scale=False)

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.revolute_object)
        else:
            # self.placement_initializer = UniformRandomSampler(
            #     name="ObjectSampler",
            #     mujoco_objects=self.revolute_object,
            #     # x_range=[-1.2, -1.2], #No randomization
            #     # y_range=[-1, -1], #No randomization
            #     x_range=[0.1, 0.1], #No randomization
            #     y_range=[-1,-1], #No randomization
            #     rotation=(-np.pi/2, -np.pi/2 ), #No randomization
            #     rotation_axis="z",
            #     ensure_object_boundary_in_range=False, 
            #     reference_pos=(-0.6, -1.0, 0.5)
            # )
            self.obj_pos = np.array([-0.6, -1, 0.6])

            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.revolute_object,
                x_range=[0, 0], #No randomization
                y_range=[0, 0], #No randomization 
                z_offset=0.1,
                # x_range=[0, 0], #No randomization
                # y_range=[0,0], #No randomization
                rotation=(0, 0 ), #No randomization
                rotation_axis="z",
                ensure_object_boundary_in_range=False, 
                reference_pos=(-0.6, -1.0, 0.5)
                # reference_pos=(0, 0, 0)
            )

        # Create task
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.revolute_object)
    
    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.object_body_ids = dict()
        self.object_body_ids["drawer"] = self.sim.model.body_name2id(self.revolute_object.revolute_body)
        # self.revolute_object_handle_site_id = self.sim.model.site_name2id(self.revolute_object.important_sites["handle"])
        self.slider_qpos_addr = self.sim.model.get_joint_qpos_addr(self.revolute_object.joints[0])

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()
        modality = "object"
        pf = self.robots[0].robot_model.naming_prefix
        @sensor(modality=modality)
        def gripper_pos(obs_cache):
            return obs_cache[f"{pf}eef_pos"] if f"{pf}eef_pos" in obs_cache else np.zeros(3)
        @sensor(modality=modality)
        def gripper_quat(obs_cache):
            return obs_cache[f"{pf}eef_quat"] if f"{pf}eef_quat" in obs_cache else np.zeros(3)
        @sensor(modality=modality)
        def handle_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.revolute_object_handle_site_id])

        # @sensor(modality=modality)
        # def cube_quat(obs_cache):
        #     return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")
        
        @sensor(modality=modality) 
        def handle_quat(obs_cache):
            return convert_quat(np.array(self.sim.data.body_xquat[self.revolute_object_handle_site_id]), to="xyzw")
        # sensors = [gripper_pos, gripper_quat,handle_pos,handle_quat]
        sensors = []
        names = [s.__name__ for s in sensors]

        # Create observables
        for name, s in zip(names, sensors):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
            )

        return observables

    def _reset_internal(self):
        super()._reset_internal()
        object_placements = self.placement_initializer.sample()

        # We know we're only setting a single object (the drawer), so specifically set its pose
        drawer_pos, drawer_quat, _ = object_placements[self.revolute_object.name]
        drawer_body_id = self.sim.model.body_name2id(self.revolute_object.root_body)
        self.sim.model.body_pos[drawer_body_id] = drawer_pos
        self.sim.model.body_quat[drawer_body_id] = drawer_quat
        self.handle_current_progress = self.sim.data.qpos[self.slider_qpos_addr] 
        # self.sim.data.qpos[self.slider_qpos_addr] = 0.5

    def _check_success(self):
        # TODO: modify this to check if the drawer is fully open
        return 0
    
    def reward(self, action):
        return 0
    
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