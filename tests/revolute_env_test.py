import robosuite as suite
from env.robot_revolute_env import RobotRevoluteOpening
import numpy as np

controller_name = "OSC_POSE"
controller_configs = suite.load_controller_config( default_controller=controller_name)

env:RobotRevoluteOpening = suite.make(
     "RobotRevoluteOpening",
    robots="Panda",
    has_renderer=True,
    use_camera_obs=True,
    has_offscreen_renderer=True,
    camera_depths = True,
    camera_segmentations = "element",
    controller_configs=controller_configs,
    control_freq = 20,
    horizon=10000,
    camera_names = "agentview",
    camera_heights = 1024,
    camera_widths = 1024
)

obs = env.reset()
env.render()
while True:
    action = np.zeros(7)
    env.step(action)
    env.render()