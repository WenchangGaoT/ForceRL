from robosuite.models.objects import MujocoXMLObject
from utils.sim_utils import fill_custom_xml_path
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