import robosuite as suite
from env.robot_revolute_env import RobotRevoluteOpening
from env.train_multiple_revolute_env import MultipleRevoluteEnv
import numpy as np
import time

controller_name = "OSC_POSE"
controller_configs = suite.load_controller_config( default_controller=controller_name)

env:MultipleRevoluteEnv = suite.make(
     "MultipleRevoluteEnv",
     object_name = "door_original",
     random_force_point = True,
    # init_door_angle = (-np.pi + np.pi/4, -np.pi + np.pi/4),
    init_door_angle = (-np.pi,-np.pi),
    has_renderer=True,
    use_camera_obs=True,
    has_offscreen_renderer=True,
    camera_depths = True,
    camera_segmentations = "element",
    horizon=1000,
    camera_names = ["agentview", "sideview", "frontview", "birdview"],
    camera_heights = [1024,1024, 1024, 1024],
    camera_widths = [1024, 256, 1024, 1024],
    render_camera = "sideview",
    # save_video = True,
    # video_width=256,
    # video_height=1024,
)

obs = env.reset()
obs = env.reset()
env.render()
for _ in range(1000):
    action = np.array([1,0.0,0.0])
    obs, _,_,_ = env.step(action)
    print(obs["hinge_qpos"])
    env.render()
    # time.sleep(0.1)

env.save_video("test_revolute_video.gif")
    