import h5py
import numpy as np
import os

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
states_dir = os.path.join(proj_dir, "baselines/states")
states_file_path = os.path.join(states_dir, "CipsBaselineTrainRevoluteEnv.h5")

data_dict = {}
with h5py.File(states_file_path, 'r') as hf:
    for name in hf.keys():
        group = hf[name]
        # Load array and XML string
        array = group['state'][:]
        xml = group['model'][()].decode('utf-8')
        print(type(xml))
        # Store in the dictionary
        data_dict[name] = (array, xml)
