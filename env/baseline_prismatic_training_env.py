import numpy as np
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import EmptyArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat
import utils.baseline_utils as b_utils

from objects.baseline_objects import BaselineTrainPrismaticObjects
from scipy.spatial.transform import Rotation as R

from grasps.aograsp.get_pointclouds import get_aograsp_ply_and_config
from grasps.aograsp.get_affordance import get_affordance_main
from grasps.aograsp.get_proposals import get_grasp_proposals_main

from robosuite.models.base import MujocoModel
from robosuite.models.grippers import GripperModel
from utils.baseline_utils import baseline_read_cf_grasp_proposals, baseline_get_grasp_proposals, get_camera_info


from copy import deepcopy 
import imageio

import os 
os.environ['MUJOCO_GL'] = 'osmesa'



class BaselineTrainPrismaticEnv(SingleArmEnv):
    '''
    Environment for the prismatic object manipulation task.

    Args:
        rotate_around_robot: bool, whether to rotate the object around the robot (when True, x_range and y_range are ignored)
    '''
    def __init__(
        self,
        robots,
        object_name = "trashcan-1",
        scale_object=False, 
        object_scale=1.0,
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
        z_offset = 0.0,
        open_percentage = 0.4,
        rotate_around_robot = False,
        object_robot_distance = (0.5, 0.5),

        move_robot_away=True, 

        # video stuff
        cache_video=False,
        video_width=256,
        video_height=256,

        # params to avoid infinite recursion
        get_grasp_proposals_flag=False,
        skip_object_initialization=False,
    ):
        self.env_name = "BaselineTrainPrismaticEnv"
        
        available_objects = BaselineTrainPrismaticObjects.available_objects()
        assert object_name in available_objects, "Invalid object!"

        self.object_name = object_name
        self.placement_initializer = placement_initializer
        self.table_full_size = (0.8, 0.3, 0.05)

        self.obj_rotation = obj_rotation 
        self.x_range = x_range
        self.y_range = y_range
        self.z_offset = z_offset
        self.open_percentage = open_percentage

        self.move_robot_away = move_robot_away

        self.scale_object = scale_object
        self.object_scale = object_scale

        self.rotate_around_robot = rotate_around_robot
        self.object_robot_distance = object_robot_distance 
        self.cache_video=cache_video 
        self.video_width = video_width
        self.video_height = video_height 
        self.get_grasp_proposals_flag = get_grasp_proposals_flag
        self.skip_object_initialization = skip_object_initialization
        self.frames = []


        self.env_kwargs = {
            'robots': robots,
            'object_name': object_name,
            'obj_rotation': obj_rotation,
            'scale_object': scale_object,
            'object_scale': object_scale,
            'has_renderer': has_renderer,
            'use_camera_obs': use_camera_obs,
            'has_offscreen_renderer': has_offscreen_renderer,
            'camera_depths': camera_depths,
            'camera_segmentations': camera_segmentations,
            'controller_configs': controller_configs,
            'control_freq': control_freq,
            'horizon': horizon,
            'camera_names': camera_names,
            'camera_heights': camera_heights,
            'camera_widths': camera_widths,
            'rotate_around_robot': rotate_around_robot,
            'object_robot_distance': object_robot_distance,
            'move_robot_away': move_robot_away,
            'open_percentage': open_percentage,
        }

        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.pcd_wf_path = f'point_clouds/world_frame_pointclouds_baseline/world_frame_{object_name}_{object_scale}_{open_percentage}.ply'
        self.pcd_wf_no_downsample_path = f'point_clouds/world_frame_pointclouds_baseline/world_frame_{object_name}_{object_scale}_{open_percentage}_no_downsample.ply'
        self.camera_info_path = f'infos/camera_info_{object_name}_{object_scale}_{open_percentage}.npz'

        self.pcd_cf_dir = os.path.join(self.project_dir, "point_clouds/camera_frame_pointclouds")
        self.pcd_cf_path = os.path.join(self.pcd_cf_dir, f"camera_frame_{object_name}_{object_scale}_{open_percentage}.ply")

        self.proposal_dir = os.path.join(self.project_dir,"outputs/grasp_proposals/world_frame_proposals")
        self.proposal_path = os.path.join(self.proposal_dir, f"world_frame_{object_name}_{object_scale}_{open_percentage}_grasp.npz")
        
        self.camera_frame_proposal_dir = os.path.join(self.project_dir, "outputs/grasp_proposals/camera_frame_proposals")
        self.camera_frame_proposal_path = os.path.join(self.camera_frame_proposal_dir, f"camera_frame_{object_name}_{object_scale}_{open_percentage}_affordance.npz")

        
        self.affordance_dir = os.path.join(self.project_dir, "outputs/point_score")
        self.affordance_path = os.path.join(self.affordance_dir, f"camera_frame_{object_name}_{object_scale}_{open_percentage}_affordance.npz")

        self.env_model_dir = os.path.join(self.project_dir, "baselines/states")
        
        if self.get_grasp_proposals_flag:
            self.init_grasp_proposals()


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

        self.joint_range = self.prismatic_object.joint_range

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

        self.robot_base_xpos = deepcopy(xpos)

        # set empty arena
        mujoco_arena = EmptyArena()

        # set camera pose
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5986131746834771, -4.392035683362857e-09, 1.5903500240372423],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349],
        )

        self.prismatic_object = BaselineTrainPrismaticObjects(name=self.object_name, 
                                                     scaled=self.scale_object,
                                                     scale=self.object_scale,)
        # print("contact geom: ", self.prismatic_object.geom_check_grasp)
        # get the actual placement of the object
        actual_placement_x, actual_placement_y, actual_placement_rotation, actual_placement_reference_pos = self.get_object_actual_placement()   

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.prismatic_object)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.prismatic_object,
                x_range = actual_placement_x, #No randomization
                y_range = actual_placement_y, #No randomization 
                z_offset=self.z_offset,
                rotation=actual_placement_rotation, #No randomization
                rotation_axis="z",
                ensure_object_boundary_in_range=False, 
                reference_pos=actual_placement_reference_pos
                # reference_pos=(0, 0, 0)
            )

        # Create task
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.prismatic_object)
    
    def get_object_actual_placement(self):
        if self.rotate_around_robot:
            robot_pos = self.robot_base_xpos
            # sample a rotation angle from obj_rotation
            angle = np.random.uniform(self.obj_rotation[0], self.obj_rotation[1])
            # sample a distance from object_robot_distance
            distance = np.random.uniform(self.object_robot_distance[0], self.object_robot_distance[1])

            # calculate correct x and y position using distance and angle
            x = robot_pos[0] + distance * np.cos(angle)
            y = robot_pos[1] + distance * np.sin(angle)

            # we don't use the randomization for x and y
            actual_placement_x = (x, x)
            actual_placement_y = (y, y)
            actual_placement_rotation = (angle, angle)
            actual_placement_reference_pos = (0, 0, 0.5)
        else:
            actual_placement_x = self.x_range
            actual_placement_y = self.y_range
            actual_placement_rotation = self.obj_rotation
            actual_placement_reference_pos = (-0.6, -1.0, 0.5)

        return actual_placement_x, actual_placement_y, actual_placement_rotation, actual_placement_reference_pos

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
        self.joint_range = self.prismatic_object.joint_range

    def calculate_grasp_pos_relative(self):
        '''
        calculate grasp pos in the frame of the revolute object
        '''
        grasp_pos = np.array(self.final_grasp_pose)
        grasp_pos = grasp_pos - deepcopy(self.prismatic_body_pos)
        grasp_pos = np.dot(self.sim.data.get_body_xmat(self.prismatic_object.prismatic_body).T, grasp_pos)
        return grasp_pos

    def calculate_grasp_pos_absolute(self):
        '''
        calculate the grasp position in the world frame
        '''
        grasp_pos = np.array(self.grasp_pos_relative)
        grasp_pos = np.dot(self.sim.data.get_body_xmat(self.prismatic_object.prismatic_body), grasp_pos)
        grasp_pos += deepcopy(self.sim.data.get_body_xpos(self.prismatic_object.prismatic_body))
        return grasp_pos

    def calculate_grasp_quat_relative(self):
        '''
        in prismatics, the grasp quaternion is the same as the initial grasp quaternion
        '''
        
        grasp_quat = self.grasp_quat
        return grasp_quat
    
    def calculate_grasp_quat_absolute(self):
        '''
        calculate the grasp quaternion in the world frame
        '''
        return self.grasp_quat


    def calculate_joint_pos_absolute(self):
        '''
        calculate the joint position in the world frame
        '''
        joint_pos = np.array(self.joint_position_rel)
    
        joint_pos += deepcopy(self.sim.data.get_body_xpos(self.prismatic_object.prismatic_body))

        return joint_pos
    
    def calculate_joint_direction_absolute(self):
        '''
        calculate the joint direction in the world frame
        '''
        joint_dir = np.array(self.joint_direction_rel)
        joint_dir = np.dot(self.sim.data.get_body_xmat(self.prismatic_object.prismatic_body), joint_dir)
        return joint_dir

    def init_grasp_proposals(self):
        '''
        When initializing the environment, get the grasp proposals for the object
        '''
        camera_euler = np.array([ 45., 22., 3.])
        camera_euler_for_pos = np.array([camera_euler[-1], camera_euler[1], -camera_euler[0]])
        camera_rotation = R.from_euler("xyz", camera_euler_for_pos, degrees=True)
        camera_quat = R.from_euler("xyz", camera_euler, degrees=True).as_quat()
        camera_forward = np.array([1, 0, 0])
        world_forward = camera_rotation.apply(camera_forward)
        distance = 1.5
        camera_pos = -distance * world_forward

        self.camera_pos = camera_pos
        self.camera_quat = camera_quat

        reset_x_range = (1,1)
        reset_y_range = (1,1)

        # reset_joint_qpos = self.sim.data.qpos[self.slider_qpos_addr]
        reset_joint_qpos = 1.0
        if not os.path.exists(self.camera_frame_proposal_path):
            # get the point clouds and affordance
            print("Getting point clouds and affordance")
            pcd_wf_path, pcd_wf_no_downsample_path,camera_info_path = get_aograsp_ply_and_config(
                                env_name = self.env_name, 
                                env_kwargs=self.env_kwargs,
                                object_name=self.object_name, 
                                camera_pos=camera_pos,
                                camera_quat=camera_quat,
                                scale_factor=1,
                                pcd_wf_path=self.pcd_wf_path,
                                pcd_wf_no_downsample_path=self.pcd_wf_no_downsample_path,
                                camera_info_path=self.camera_info_path,
                                viz=False, 
                                need_o3d_viz=False,
                                reset_joint_qpos=reset_joint_qpos,
                                reset_x_range = reset_x_range, 
                                reset_y_range = reset_y_range,
            )
            pcd_cf_path, affordance_path = get_affordance_main(pcd_wf_path, camera_info_path, 
                                                    viz=False)
            
            run_cgn = True
            store_proposals = True

            get_grasp_proposals_main(
                pcd_cf_path, 
                affordance_path, 
                self.camera_info_path, 
                run_cgn=run_cgn, 
                viz=False, 
                save_wf_pointcloud=False,
                object_name=f"{self.object_name}_{self.object_scale}_{self.open_percentage}",
                top_k=10,
                store_proposals=store_proposals,
            )
        else:
            pcd_cf_path = self.pcd_cf_path
        
        self.proposal_points_cf, self.proposal_quat_cf, self.proposal_scrores = baseline_read_cf_grasp_proposals(
            pcd_cf_path,
            self.camera_frame_proposal_path)

    def get_grasp_proposals(self):


        camera_config = get_camera_info(self, 
                                        self.camera_pos, 
                                        self.camera_quat, 
                                        scale_factor=1,
                                        )
        top_k_pos_wf, top_k_quat_wf = baseline_get_grasp_proposals(
            camera_config,
            self.proposal_points_cf,
            self.proposal_quat_cf,
            self.proposal_scrores,
            top_k=10
        )
        
        print("object pos: ", self.obj_pos)
        top_k_pos_wf = top_k_pos_wf + self.obj_pos
        grasp_pos = top_k_pos_wf[1]
        grasp_quat = top_k_quat_wf[1]
        self.final_grasp_pose = grasp_pos + np.array([0, 0, -0.05])
        sci_rotation = R.from_quat(grasp_quat)
        further_rotation = R.from_euler('z', 90, degrees=True)
        sci_rotation = sci_rotation * further_rotation
        self.grasp_rotation_vector = sci_rotation.as_rotvec()
        self.grasp_quat = sci_rotation.as_quat()
        # get the relative position of the grasp against the object
        self.grasp_pos_relative = self.calculate_grasp_pos_relative()
        self.grasp_quat_relative = self.calculate_grasp_quat_relative()
        print("grasp pos relative: ", self.grasp_pos_relative)
        print("grasp pos absolute: ", self.calculate_grasp_pos_absolute())
        print("final grasp pose: ", self.final_grasp_pose)


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
        if self.get_grasp_proposals_flag:
            @sensor(modality=modality)
            def grasp_pos(obs_cache):
                return self.calculate_grasp_pos_absolute()
            @sensor(modality=modality)
            def grasp_quat(obs_cache):
                return self.grasp_quat
            @sensor(modality=modality)
            def grasp_rot(obs_cache):
                return self.grasp_rotation_vector
            @sensor(modality=modality)
            def joint_direction(obs_cache):
                return self.joint_direction
            @sensor(modality=modality)
            def open_progress(obs_cache):
                return np.array([(self.handle_current_progress - self.joint_range[0]) / (self.joint_range[1] - self.joint_range[0])])
            sensors = [gripper_pos, gripper_quat, grasp_pos, grasp_quat, grasp_rot, joint_direction, open_progress]
        else:
            sensors = [gripper_pos, gripper_quat]

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
        self.frames = [] 

        # if rotate_around_robot is True, need to reset the object placement parameters
        if self.rotate_around_robot:
            actual_placement_x, actual_placement_y, actual_placement_rotation, actual_placement_reference_pos = self.get_object_actual_placement()
            if not self.skip_object_initialization:
                self.placement_initializer.x_range = actual_placement_x
                self.placement_initializer.y_range = actual_placement_y
                self.placement_initializer.rotation = actual_placement_rotation
                self.placement_initializer.reference_pos = actual_placement_reference_pos

        if not self.skip_object_initialization:
            object_placements = self.placement_initializer.sample()

            # We know we're only setting a single object (the drawer), so specifically set its pose
            drawer_pos, drawer_quat, _ = object_placements[self.prismatic_object.name]
            drawer_body_id = self.sim.model.body_name2id(self.prismatic_object.root_body)
            self.sim.model.body_pos[drawer_body_id] = drawer_pos
            self.sim.model.body_quat[drawer_body_id] = drawer_quat
            
            self.set_open_percentage(self.open_percentage)
        self.sim.forward()

        self._update_reference_values()

        if self.get_grasp_proposals_flag:
            self.get_grasp_proposals()

    def _update_reference_values(self):
        self.handle_current_progress = self.sim.data.qpos[self.slider_qpos_addr] 
        self.last_slider_qpos = deepcopy(self.handle_current_progress)
        self.initial_slider_qpos = deepcopy(self.handle_current_progress)
        
        obj_id = self.sim.model.body_name2id(f'{self.prismatic_object.naming_prefix}main')   
        self.obj_pos = self.sim.data.body_xpos[obj_id] 
        self.obj_quat = self.sim.data.body_xquat[obj_id]

        self.prismatic_body = self.prismatic_object.prismatic_body
        self.prismatic_body_pos = self.sim.data.body_xpos[self.sim.model.body_name2id(self.prismatic_body)]
        
        self.joint_position_rel = self.prismatic_object.joint_pos_relative
        self.joint_direction_rel = self.prismatic_object.joint_direction / np.linalg.norm(self.prismatic_object.joint_direction)

        self.joint_position = self.calculate_joint_pos_absolute()
        self.joint_direction = self.calculate_joint_direction_absolute()
        self.prismatic_bosy_mat = self.sim.data.get_body_xmat(self.prismatic_body)
        self.gripper_pos = self.sim.data.get_site_xpos(self.robots[0].gripper.important_sites["grip_site"])

        

    def _check_success(self):
        # TODO: modify this to check if the drawer is fully open 
        joint_qpos = self.sim.data.qpos[self.slider_qpos_addr]

        joint_pos_relative_to_range = (joint_qpos - self.joint_range[0]) / (self.joint_range[1] - self.joint_range[0]) - self.open_percentage
        # open 30 degrees
        return joint_pos_relative_to_range > 0.8
    
    def _check_grasp(self, gripper, object_geoms):
        """
        Checks whether the specified gripper as defined by @gripper is grasping the specified object in the environment.

        Modified version, return true if one of the gripper geom group is in contact with the object geom group

        Args:
            gripper (GripperModel or str or list of str or list of list of str): If a MujocoModel, this is specific
            gripper to check for grasping (as defined by "left_fingerpad" and "right_fingerpad" geom groups). Otherwise,
                this sets custom gripper geom groups which together define a grasp. This can be a string
                (one group of single gripper geom), a list of string (multiple groups of single gripper geoms) or a
                list of list of string (multiple groups of multiple gripper geoms). At least one geom from each group
                must be in contact with any geom in @object_geoms for this method to return True.
            object_geoms (str or list of str or MujocoModel): If a MujocoModel is inputted, will check for any
                collisions with the model's contact_geoms. Otherwise, this should be specific geom name(s) composing
                the object to check for contact.

        Returns:
            bool: True if the gripper is grasping the given object
        """
        # Convert object, gripper geoms into standardized form
        if isinstance(object_geoms, MujocoModel):
            o_geoms = object_geoms.contact_geoms
        else:
            o_geoms = [object_geoms] if type(object_geoms) is str else object_geoms
        if isinstance(gripper, GripperModel):
            g_geoms = [gripper.important_geoms["left_fingerpad"], gripper.important_geoms["right_fingerpad"]]
        elif type(gripper) is str:
            g_geoms = [[gripper]]
        else:
            # Parse each element in the gripper_geoms list accordingly
            g_geoms = [[g_group] if type(g_group) is str else g_group for g_group in gripper]
            
        # Search for collisions between each gripper geom group and the object geoms group
        for g_group in g_geoms:
            if self.check_contact(g_group, o_geoms):
                return True
        return False


    def check_grasp(self):
        '''
        Check if robot successfully grasps the object
        '''
        return self._check_grasp(self.robots[0].gripper, self.prismatic_object.geom_check_grasp)

    def reward(self, action):
        '''
        staged reward function
        '''
        self.handle_current_progress = self.sim.data.qpos[self.slider_qpos_addr]
        stage = self.get_stage()
        if self.get_grasp_proposals_flag:
            rwd = self.staged_reward(stage, self.gripper_pos)
        else:
            rwd = 0
        return rwd
    
    def get_stage(self):
        '''
        get the stage of the current task

        stage 0: check_grasp is False
        stage 1: check_grasp is True, handle_current_progress < 0.8
        stage 2: check_grasp is True, handle_current_progress >= 0.8
        '''
        # self.handle_current_progress = self.sim.data.qpos[self.slider_qpos_addr]
        # print("handle_current_progress: ", self.handle_current_progress)
        task_percentage = (self.handle_current_progress - self.joint_range[0]) / (self.joint_range[1] - self.joint_range[0])

        if task_percentage >= 0.8:
            return 2
        elif not self.check_grasp():
            return 0
        # elif task_percentage >= 0.8:
        #     return 2
        else:
            return 1
        # elif self.check_grasp() and task_percentage < 0.8:
        #     return 1
        
        

    def staged_reward(self, stage, gripper_pos):
        '''
        three part of the reward function
        reward_stage: 0, 1, 2
        reward_end_effector: reward for the end effector, encourage the end effector to be close to the grasp position
        reward_drawer: reward for the drawer, encourage the drawer to be opened
        '''
        eef_mult = 1.0
        stage_mult = 1.0
        drawer_mult = 1.0

        # print("stage: ", stage)

        reward_stage_list = [0,1,2]
        reward_stage = reward_stage_list[stage] * stage_mult

        dist = np.linalg.norm(gripper_pos - self.calculate_grasp_pos_absolute())
        reward_end_effector = (1 - np.tanh(5.0 * dist)) * eef_mult
        
        reward_drawer = (self.handle_current_progress - self.joint_range[0]) / (self.joint_range[1] - self.joint_range[0])
        reward_drawer = reward_drawer * drawer_mult

        reward_stop = self._check_success() * 10

        if stage == 0:
            reward = reward_stage + reward_end_effector
        elif stage == 1:
            reward_end_effector = 1
            reward = reward_stage + reward_end_effector + reward_drawer
        else:
            reward_end_effector = 1
            reward = reward_stage + reward_end_effector + reward_drawer + reward_stop
        
        self.last_slider_qpos = deepcopy(self.handle_current_progress)
        # print("reward: ", reward)
        # print("reward drawer: ", reward_drawer) 
        return reward
        
    def penalty(self, action):
        '''
        penalty function
        '''
        stage = self.get_stage()
        if stage == 0:
            drawer_movement = np.abs(self.handle_current_progress - self.initial_slider_qpos)
            # print("drawer_movement: ", drawer_movement)
            if drawer_movement > 0.01:
                # if hit drawer, then task is failed
                return -1
        return 0


    def step(self, action):
        '''
        step function, terminate the episode if the drawer is hit
        '''
        obs, reward, done, info = super().step(action)
        if self.cache_video and self.has_offscreen_renderer:
                frame = self.sim.render(self.video_width, self.video_height, camera_name='frontview')
                # print(frame.shape)
                frame = np.flip(frame, 0)
                
                self.frames.append(frame)
        # reward = self.reward(action)

        # # check penalty
        # penalty = self.penalty(action)
        # if penalty != 0:
        #     done = True
        #     self.done = True
        #     reward = -10

        # print("reward: ", reward)

        return obs, reward, done, info

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

    def set_open_percentage(self, percent):
        '''
        Set the open percent of the drawer
        Input:
            -percent: float, the percent of the drawer to be opened
        '''
        # get the joint range
        joint_range = self.prismatic_object.joint_range
        self.joint_range = joint_range

        self.sim.data.qpos[self.slider_qpos_addr] = percent * (joint_range[1] - joint_range[0]) + joint_range[0]
        # self.sim.forward()

    @classmethod
    def available_objects(cls):
        available_objects = {
            "prismatic": ["train-drawer-1"],
        }
        return available_objects 
    
    def save_video(self, video_path='videos/robot_prismatic.mp4'):
        imageio.mimsave(video_path, self.frames, fps=120)
        # imageio.imwrite(video_path, self.frames[0])