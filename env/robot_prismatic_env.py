import numpy as np
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import EmptyArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat

from objects.custom_objects import EvalPrismaticObjects 
from scipy.spatial.transform import Rotation as R
from copy import deepcopy

import os 
os.environ['MUJOCO_GL'] = 'osmesa'


class RobotPrismaticEnv(SingleArmEnv):

    def __init__(
        self,
        robots,
        object_name = "trashcan-1",
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
        agentview_camera_quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349], 

        # Object rotation
        obj_rotation=(-np.pi/2, -np.pi / 2),
        x_range = (-1,-1),
        y_range = (0,0),
        move_robot_away=True,
    ):
        
        available_objects = EvalPrismaticObjects.available_objects()
        assert object_name in available_objects, "Invalid object!"

        self.object_name = object_name
        self.placement_initializer = placement_initializer
        self.table_full_size = (0.8, 0.3, 0.05)

        self.object_model_idx = object_model_idx 
        self.obj_rotation = obj_rotation 
        self.x_range = x_range
        self.y_range = y_range
        self.move_robot_away = move_robot_away

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
        
    # def move_robot_away(self): 
    #     print('Moving robot away')
    #     xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
    #     xpos = (20, xpos[1], xpos[2]) 
    #     self.robots[0].robot_model.set_base_xpos(xpos) 
    #     # self.sim.data.qpos[self.robots[0].robot_model.joint_qposadr[:3]] = xpos
    #     # self.robots[0].set_robot_joint_positions(xpos) 
    #     self.sim.forward()

    # def move_robot_back(self): 
    #     print('Moving robot back')
    #     self.robots[0].robot_model.set_base_xpos(self.robot_init_xpos)
    #     self.sim.forward()

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # set robot position
        # xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0]) 
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](0)
        # self.robot_init_xpos = xpos
        if self.move_robot_away:
            xpos = (20, xpos[1], xpos[2]) # Move the robot away
            self.robots[0].robot_model.set_base_xpos(xpos)
        else:
            xpos = (xpos[0], xpos[1] + 0.3, xpos[2]) # Move the robot away
            self.robots[0].robot_model.set_base_xpos(xpos)

        # set empty arena
        mujoco_arena = EmptyArena()

        # set camera pose
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5986131746834771, -4.392035683362857e-09, 1.5903500240372423],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349],
        )

        self.prismatic_object = EvalPrismaticObjects(name=self.object_name)

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.prismatic_object)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.prismatic_object,
                x_range = self.x_range, #No randomization
                y_range = self.y_range, #No randomization 
                z_offset=0.1,
                # x_range=[0, 0], #No randomization
                # x_range=[-1, -1], #No randomization
                # y_range=[-0.54,-0.54], #No randomization
                # y_range=[0,0], #No randomization
                rotation=self.obj_rotation, #No randomization
                rotation_axis="z",
                ensure_object_boundary_in_range=False, 
                reference_pos=(-0.6, -1.0, 0.5)
                # reference_pos=(0, 0, 0)
            )

        # Create task
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.prismatic_object)
    
    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        self.slider_qpos_addr = self.sim.model.get_joint_qpos_addr(self.prismatic_object.joint) 
        obj_id = self.sim.model.body_name2id(f'{self.prismatic_object.naming_prefix}main')
        self.obj_pos = self.sim.data.body_xpos[obj_id] 
        self.obj_quat = self.sim.data.body_xquat[obj_id]



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
            return np.array(self.sim.data.body_xpos[self.prismatic_object_handle_site_id])

        # @sensor(modality=modality)
        # def cube_quat(obs_cache):
        #     return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")
        
        @sensor(modality=modality) 
        def handle_quat(obs_cache):
            return convert_quat(np.array(self.sim.data.body_xquat[self.prismatic_object_handle_site_id]), to="xyzw")
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
    
    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step)

    def _reset_internal(self):
        super()._reset_internal()
        object_placements = self.placement_initializer.sample()

        # We know we're only setting a single object (the drawer), so specifically set its pose
        drawer_pos, drawer_quat, _ = object_placements[self.prismatic_object.name]
        drawer_body_id = self.sim.model.body_name2id(self.prismatic_object.root_body)
        self.sim.model.body_pos[drawer_body_id] = drawer_pos
        self.sim.model.body_quat[drawer_body_id] = drawer_quat
        self.handle_current_progress = self.sim.data.qpos[self.slider_qpos_addr] 

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

    @classmethod
    def available_objects(cls):
        available_objects = {
            "prismatic": "trashcan-1"
        }
        return available_objects