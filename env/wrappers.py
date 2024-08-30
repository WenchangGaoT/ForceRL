from robosuite.wrappers import Wrapper
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

 

class ActionRepeatWrapperNew(Wrapper):
    def __init__(self, env, action_repeat):
        super().__init__(env)
        self.action_repeat = action_repeat

    def step(self, action): 
        rwd = 0
        for _ in range(self.action_repeat):
            n_state, n_rwd, done, info = self.env.step(action)
            rwd += n_rwd
            if done:
                break
        return n_state, rwd, done, info
    
    
class ParallelEnvsWrapper(Wrapper):
    def __init__(self, envs: list):
        super().__init__() 
        self.envs = envs

    def step(self, action): 
        n_obs, n_rwd, n_done, n_info = [], [], [], []
        for env in self.envs:
            obs, rwd, done, info = env.step(action)
            n_obs.append(obs) 
            n_rwd.append(rwd)
            n_done.append(done)
            n_info.append(info)
        n_obs = np.concatenate(n_obs)
        n_rwd = np.array(n_rwd)
        n_done = np.array(n_done)
        n_info = np.array(n_info)
        return n_obs, n_rwd, n_done, n_info
    
    def reset(self):
        n_obs = []
        for env in self.envs:
            obs = env.reset()
            n_obs.append(obs)
        n_obs = np.concatenate(n_obs)
        return n_obs