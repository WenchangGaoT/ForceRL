    # def change_scale(self, new_scale):
    #     '''
    #     Change the scale of the object
    #     '''
    #     # find all bodies
    #     bodies = find_elements(root=self.worldbody, tags="body")
    #     for body in bodies:
    #         # get the scale
    #         # scale = body.get("scale")
    #         # scale = scale.split(" ")
    #         # scale = [float(x) for x in scale]
    #         # # change the scale
    #         # scale = [x * new_scale for x in scale]
    #         body.set("scale", array_to_string(new_scale))

    #     # find all geoms
    #     geoms = find_elements(root=self.worldbody, tags="geom")
    #     for geom in geoms:
    #         # get the scale
    #         # scale = geom.get("size")
    #         # scale = scale.split(" ")
    #         # scale = [float(x) for x in scale]
    #         # # change the scale
    #         # scale = [x * new_scale for x in scale]
    #         geom.set("scale", array_to_string(new_scale))