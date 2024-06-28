from env.robot_drawer_opening_new_drawer import RobotRandomDrawerOpening
import robosuite as suite
import numpy as np


controller_name = "OSC_POSE"
controller_configs = suite.load_controller_config(default_controller=controller_name)
env:RobotRandomDrawerOpening = suite.make(
    "RobotRandomDrawerOpening",
    robots="Panda",
    has_renderer=True,
    use_camera_obs=True,
    has_offscreen_renderer=True,
    camera_depths = True,
    camera_segmentations = "element",
    controller_configs=controller_configs,
    control_freq = 20,
    horizon=10000,
    camera_names = ["birdview", "agentview", "frontview", "sideview"],
    camera_heights = [1024,256,512,1024],
    camera_widths = [1024,1024,1024,1024]
)


obs = env.reset()
env.render()
for i in range(1000):
    action = np.zeros(7)
    obs, reward, done, info = env.step(action)
    env.render()