# utils about URDF, MJCF and XML stuff



import os
import sys
from shutil import copyfile

import mujoco

import trimesh



def print_usage():
    print("""python compile_mjcf_model.py input_file output_file""")


def compile_mjcf_model(input_file, output_file):
    """Loads a raw mjcf file and saves a compiled mjcf file.

    This avoids mujoco-py from complaining about .urdf extension.
    Also allows assets to be compiled properly.

    Example:
        $ python compile_mjcf_model.py source_mjcf.xml target_mjcf.xml
    """
    input_folder = os.path.dirname(input_file)

    tempfile = os.path.join(input_folder, ".robosuite_temp_model.xml")
    copyfile(input_file, tempfile)

    model = mujoco.MjModel.from_xml_path(tempfile)
    xml_string = model.get_xml()
    with open(output_file, "w") as f:
        f.write(xml_string)

    os.remove(tempfile)

def convert_obj_to_stl(file_dir, output_dir):
    # find all obj files in the directory
    obj_files = [f for f in os.listdir(file_dir) if f.endswith('.obj')]
    # create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for obj_file in obj_files:
        obj_path = os.path.join(file_dir, obj_file)
        mesh = trimesh.load(obj_path)
        mesh.export(file_obj=os.path.join(output_dir, obj_file.replace('.obj', '.stl')))
        print(f"Converted {obj_file} to {obj_file.replace('.obj', '.stl')}")
    print(f"Converted {len(obj_files)} obj files to stl files")

def ascii_to_binary_stl(file_dir, outpur_dir):
    # convert all ascii stl files to binary stl files for mujoco
    stl_files = [f for f in os.listdir(file_dir) if f.endswith('.stl')]
    # create output directory if it does not exist
    if not os.path.exists(outpur_dir):
        os.makedirs(outpur_dir)
    
    for stl_file in stl_files:
        stl_path = os.path.join(file_dir, stl_file)
        mesh = trimesh.load(stl_path)
        mesh.export(file_obj=os.path.join(outpur_dir, stl_file), file_type='stl')
        print(f"Converted {stl_file} to binary stl")
    print(f"Converted {len(stl_files)} stl files to binary stl files")
    


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print_usage()
        exit(0)

    func_name = sys.argv[1]
    input_dir = sys.argv[2]
    output_dir = sys.argv[3]
    # execute the function
    if func_name == "convert_obj_to_stl":
        convert_obj_to_stl(input_dir, output_dir)
    elif func_name == "ascii_to_binary_stl":
        ascii_to_binary_stl(input_dir, output_dir)
    elif func_name == "compile_mjcf_model":
        compile_mjcf_model(input_dir, output_dir)
    else:
        print(f"Function {func_name} not found")
        print_usage()
        exit(0)
