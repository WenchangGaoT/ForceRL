import numpy as np

import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_controller_config
from robosuite.renderers import load_renderer_config
from robosuite.utils.input_utils import *
from env.move_box_env import MoveBoxEnv
import mujoco


env = suite.make(
    "MoveBoxEnv",
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=False,
    # reward_shaping=True,
    control_freq=20,
    horizon=100
)
geom_name = [cube_geo for cube_geo in env.sim.model.geom_names if 'cube' in cube_geo]

# geom_id = [env.sim.model.geom_name2id(cube_geo) for cube_geo in env.sim.model.geom_names if 'cube' in cube_geo]
print(geom_name)
force = np.array([0.01,0,0])
torque = np.array([0,0,0])
point = np.array([0,0,0])
qfrc_target = np.zeros(env.sim.model.nv) 
print(env.sim.model.geom_name2id('cube_g0_vis'))
# print(env.sim.model._model.nbody)
for i in range(env.sim.model._model.ngeom):
    print(f'{i}: {env.sim.model.geom_id2name(i)}')


action = np.zeros(6)
for i in range(2):
    _ = env.step(action)
    env.render()
while True:
    point = env.sim.data.body_xpos[env.cube_body_id]
    mujoco.mj_applyFT(env.sim.model._model, env.sim.data._data, force, torque, point, env.sim.model.body_name2id('cube_main'), env.sim.data._data.qfrc_applied)
    print(env.sim.data.qfrc_applied)
    _ = env.step(action)
    env.render()


