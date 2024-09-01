# object for baseline training

from robosuite.models.objects import MujocoXMLObject
from utils.sim_utils import fill_custom_xml_path
from utils.model_uilts import scale_object_new
from robosuite.utils.mjcf_utils import array_to_string, find_elements, xml_path_completion
import numpy as np
import os

class BaselineTrainRevoluteObjects(MujocoXMLObject):
    """
    Sample Objects with revolute joints (used in Train)
    Args:
        name (str): Name of the object
    """

    def __init__(self, name, scaled = False, scale = 1.0, 
                 friction = 3.0, damping = 1.0):
        available_names = ["train-door-counterclock-1", "door_original", "train-microwave-1", "train-dishwasher-1"]
        assert name in available_names, "Object name must be one of {}".format(available_names)

        if not scaled:
            xml_path = f"{name}/{name}.xml"
        
        else:
            # call function to scale the object
            xml_path_original = fill_custom_xml_path(f"{name}/{name}.xml")
            scale_object_new(xml_path_original, None, [scale, scale, scale])
            xml_path = f"{name}/{name}-scaled.xml"
        super().__init__(
            fill_custom_xml_path(xml_path), name=name, joints=None, obj_type="all",duplicate_collision_geoms=True
            )

        self.revolute_body = self.naming_prefix + "link_0"
        self.joint = self.naming_prefix + "joint_0"

            # set the door friction and damping to 0 for training
        self._set_door_friction(friction)
        self._set_door_damping(damping) 
    
    @staticmethod
    def available_objects():
        available_objects = [
            'train-microwave-1', 
            'train-dishwasher-1',
        ]
        return available_objects

    def _set_door_friction(self, friction):
      
        joint = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.joint}, return_first=True)
        joint.set("frictionloss", array_to_string(np.array([friction])))

    def _set_door_damping(self, damping):

        joint = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.joint}, return_first=True)
        joint.set("damping", array_to_string(np.array([damping])))
    
    @property
    def joint_direction(self):
        '''
        Returns:
            str: joint direction
        '''
        joint = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.joint}, return_first=True)
        joint_direction_text = joint.get("axis")
        joint_direction = joint_direction_text.split(" ")
        joint_direction = [float(x) for x in joint_direction]
        return joint_direction
    
    @property
    def joint_pos_relative(self):
        '''
        Returns:
            str: joint position relative to the object
        '''
        joint = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.joint}, return_first=True)
        joint_pos = joint.get("pos")
        joint_pos = joint_pos.split(" ")
        joint_pos = [float(x) for x in joint_pos]
        return joint_pos
    
    @property
    def joint_range(self):
        '''
        Returns:
            str: joint ransge
        '''
        joint = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.joint}, return_first=True)
        joint_range = joint.get("range")
        joint_range = joint_range.split(" ")
        joint_range = [float(x) for x in joint_range]
        return joint_range
    
    @property
    def geom_check_grasp(self):
        '''
        get list of geoms used to check grasp
        '''
        contact_geoms = self.contact_geoms
        geom_for_grasp = [g for g in contact_geoms if "panel" in g or "handle" in g]
        return geom_for_grasp
    

class BaselineTrainPrismaticObjects(MujocoXMLObject):
    def __init__(self, name,scaled = False, scale = 1.0):

        self.available_obj_list = self.available_objects()
        assert name in self.available_obj_list, "Invalid object! Please use another name"

        if not scaled:
            xml_path = f"{name}/{name}.xml"
        
        else:
            # call function to scale the object
            xml_path_original = fill_custom_xml_path(f"{name}/{name}.xml")
            scale_object_new(xml_path_original, None, [scale, scale, scale])
            xml_path = f"{name}/{name}-scaled.xml"
        super().__init__(
            fill_custom_xml_path(xml_path), name=name, joints=None, obj_type="all",duplicate_collision_geoms=True
            )


        self.set_panel_size()
        self.prismatic_body = self.naming_prefix + "link_0"
        self.joint = self.naming_prefix + "joint_0"
        
    def set_panel_size(self,scaled=False, scale=1.0):
        self.panel_geom_size = {
            "train-drawer-1": [(-0.,0.), (-0.1, 0.1), (-0.3, 0.3)],
        }
        if scaled:
            self.panel_geom_size["train-drawer-1"] = [(-0.,0.), (-0.1*scale, 0.1*scale), (-0.3*scale, 0.3*scale)]

    @staticmethod
    def available_objects():
        available_objects = [
            'train-drawer-1', 
            # 'cabinet-1', 
            # 'cabinet-2', 
            # 'cabinet-3', 
        ]
        return available_objects
    
    @property
    def joint_pos_relative(self):
        joint = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.joint}, return_first=True)
        joint_pos = joint.get("pos")
        joint_pos = joint_pos.split(" ")
        joint_pos = [float(x) for x in joint_pos]
        return joint_pos
    
    @property
    def joint_direction(self):
        '''
        Returns:
            str: joint direction
        '''
        joint = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.joint}, return_first=True)
        joint_direction = joint.get("axis")
        joint_direction = joint_direction.split(" ")
        joint_direction = [float(x) for x in joint_direction]
        return joint_direction
    
    @property
    def joint_range(self):
        '''
        Returns:
            str: joint ransge
        '''
        joint = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.joint}, return_first=True)
        joint_range = joint.get("range")
        joint_range = joint_range.split(" ")
        joint_range = [float(x) for x in joint_range]
        return joint_range
    
    @property
    def geom_check_grasp(self):
        '''
        get list of geoms used to check grasp
        '''
        contact_geoms = self.contact_geoms
        geom_for_grasp = [g for g in contact_geoms if "panel" in g or "handle" in g]
        return geom_for_grasp