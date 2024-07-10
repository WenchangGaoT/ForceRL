from robosuite.models.objects import MujocoXMLObject
from utils.sim_utils import fill_custom_xml_path
from robosuite.utils.mjcf_utils import array_to_string, find_elements, xml_path_completion
import numpy as np


class DrawerObject(MujocoXMLObject):
    """
    Door with handle (used in Drawer)

    Args:
    """

    def __init__(self, name):
        xml_path = "drawer/drawer.xml"
        super().__init__(
            fill_custom_xml_path(xml_path), name=name, joints=None, obj_type="all", duplicate_collision_geoms=True
        )

        # Set relevant body names
        self.drawer_body = self.naming_prefix + "link_1"
        self.drawer_joint = self.naming_prefix + "joint_1"

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of door handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle"})
        return dic
    
class SelectedDrawerObject(MujocoXMLObject):
    def __init__(self, name, xml_path):
        super().__init__(
            fill_custom_xml_path(xml_path), name=name, joints=None, obj_type="all", duplicate_collision_geoms=True
        )

        # Set relevant body names
        self.drawer_body = self.naming_prefix + "link_1"
        self.drawer_joint = self.naming_prefix + "joint_1"


class OriginalDoorObject(MujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """
    def __init__(self, name, friction=None, damping=None, lock=False):
        xml_path = "door_original/door_original.xml"
        super().__init__(
            fill_custom_xml_path(xml_path), name=name, joints=None, obj_type="all", duplicate_collision_geoms=True
        )
        self.door_body = self.naming_prefix + "door"
        self.frame_body = self.naming_prefix + "frame"
        # self.latch_body = self.naming_prefix + "latch"
        self.hinge_joint = self.naming_prefix + "hinge"
        self.lock = False # we don't have lock for this door
        self.friction = friction
        self.damping = damping

        if self.friction is not None:
            self._set_door_friction(self.friction)
        if self.damping is not None:
            self._set_door_damping(self.damping)
    
    def _set_door_friction(self, friction):
        """
        Helper function to override the door friction directly in the XML

        Args:
            friction (3-tuple of float): friction parameters to override the ones specified in the XML
        """
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("frictionloss", array_to_string(np.array([friction])))

    def _set_door_damping(self, damping):
        """
        Helper function to override the door friction directly in the XML

        Args:
            damping (float): damping parameter to override the ones specified in the XML
        """
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("damping", array_to_string(np.array([damping])))

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of door handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle"})
        dic.update({"hinge": self.naming_prefix + "hinge_site"})
        return dic
    
    @property
    def door_panel_size(self):
        '''
        Returns:
            tuple: size of the door panel
        '''

        # TODO: hard-coded value for now, is there a way to extract this from the XML?
        return (0.22, 0.02, .29)
    
    @property
    def hinge_direction(self):
        '''
        Returns:
            str: hinge direction
        '''
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        return hinge.get("axis")
