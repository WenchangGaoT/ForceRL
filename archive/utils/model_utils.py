def scale_object(original_xml_path, output_name, scale=(0.3, 0.3, 0.3)):
    dir_name = os.path.dirname(os.path.abspath(original_xml_path)) 
    print(dir_name)
    files = os.listdir(dir_name)
    print(files) 
    assets = {} 
    for f in files:
        if f.endswith('.stl'):
            assets[f] = read_stl(os.path.join(dir_name, f))
            # assets[f].replace('/', '//')
    # print(dir_name) 
    print(assets)
    print(original_xml_path)
    mjcf_model = mjcf.from_path(original_xml_path, assets=assets)
    print(type(mjcf_model))
    print(mjcf_model.asset.__dir__()) 
    meshes = mjcf_model.find_all('mesh') 
    print('num meshes: ', len(meshes))
    for mesh in meshes:
        mesh.scale = list(scale)
        mesh.file.vfs_filename = mesh.name + '.stl'
        print(mesh.file.get_vfs_filename())
    
    geoms = mjcf_model.find_all('geom')
    for geom in geoms:
        if isinstance(geom.pos, np.ndarray):
            # print("modifying!")
            geom.pos *= scale[0]
    # print(geoms)

    bodies = mjcf_model.find_all('body')
    # print(bodies)
    print('num bodies: ', len(bodies))
    for body in bodies:
        print(type(body.pos)) 
        if isinstance(body.pos, np.ndarray):
            body.pos *= scale[0]

    joints = mjcf_model.find_all('joint') 
    print('num joints: ', len(joints))
    for joint in joints:
        print(type(joint.pos)) 
        if isinstance(joint.pos, np.ndarray):
            joint.pos *= scale[0] 
    
    # print(mjcf_model.to_xml_string())
    with open(output_name, 'w') as f:
        f.write(mjcf_model.to_xml_string())
    # return mjcf_model
    # for asset in mjcf_model.asset._children:
    #     print(type(asset))
        # if asset['mesh'] is not None:
        #     print(asset)