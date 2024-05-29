import argparse
import json
import mujoco 

import numpy as np

import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_controller_config
from robosuite.renderers import load_renderer_config
from robosuite.utils.input_utils import *
# from env.drawer_opening import DrawerOpeningEnv
from test_env.test_drawer_opening import DrawerOpeningEnvTest

controller_name = "OSC_POSE"
controller_configs = suite.load_controller_config(default_controller=controller_name)

import os 
os.environ['MUJOCO_GL'] = 'osmesa'

env = suite.make(
    "DrawerOpeningEnvTest",
    # robots="Kinova3",
    has_renderer=True,
    use_camera_obs=False,
    has_offscreen_renderer=False,
    camera_depths = False,
    camera_segmentations = "element",
    # controller_configs=controller_configs,
    control_freq = 20,
    horizon=10000,
    camera_names = "agentview",
    camera_heights = 256,
    camera_widths = 256
)

obs = env.reset()
# force = np.array([0.2, 0.2 , 0.3])
force = np.array([0, 4. , 0])
torque = np.array([0,0,0])
point = np.array([0,0,0])
qfrc_target = np.zeros(env.sim.model.nv) 
for i in range(env.sim.model._model.nbody):
    print(f'{i}: {env.sim.model.body_id2name(i)}')

env.render()
while True:
    env.step(force)
    env.render()
