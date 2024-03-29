import robosuite as suite 
from env.move_box_env import MoveBoxEnv
import numpy as np

# env = MoveBoxEnv(has_renderer=True) 
env = suite.make(
    "MoveBoxEnv", 
    has_renderer=True, 
    has_offscreen_renderer=False, 
    use_camera_obs=False, 
    # reward_shaping=True,
    control_freq=20, 
)

# action_space = env.action_space 
# observation_space = env.observation_space
obs = env.reset()
for _ in range(100):
    action = np.array([0, 0, 0, 0, 0, 0])
    
    obs, reward, done, info = env.step(action)
    # print(obs['cube_quat'])
    env.render()