import torch
import numpy as np
import copy


class StateInference:
    '''
    Given a critic, state bounds, and action sequence, infer the state
    '''
    def __init__(self,
                 critic: torch.nn.Module,
                axis_position_bounds: np.ndarray, 
                resample_times: int = 1, 
                num_samples = 4096,
                device = "cuda") -> None:
        
        self.critic = critic
        self.critic.eval()
        self.axis_position_bounds = axis_position_bounds
        self.resample_times = resample_times
        self.num_samples = num_samples
        self.axis_position_samples, self.axis_direction_samples = self.generate_state()
        self.axis_samples = np.concatenate([self.axis_position_samples, self.axis_direction_samples], axis=1)
        self.device = device    

    def generate_state(self):
        '''
        Generate a state within the bounds
        '''
        # sample axis positions
        axis_position_samples = np.random.uniform(self.axis_position_bounds[0], self.axis_position_bounds[1], (self.num_samples, len(self.axis_position_bounds[0])))

        # sample axis directions
        axis_direction_samples = np.random.rand(self.num_samples, len(self.axis_position_bounds[0]))
        # normalize
        axis_direction_samples = axis_direction_samples / np.linalg.norm(axis_direction_samples, axis=1)[:, None]

        ground_truth_axis_position = np.array([ 0.4412294, -0.390203,   0.9371408])
        ground_truth_axis_direction = np.array([ 0., 0., 1.])
        ground_truth_axis_direction = ground_truth_axis_direction / np.linalg.norm(ground_truth_axis_direction)

        # make all axis_direction_samples [0, 0, 1],for debugging
        # axis_direction_samples = np.repeat(ground_truth_axis_direction.reshape(1, -1), self.num_samples, axis=0)

        # put groung truth axis in the samples,for debugging
        axis_position_samples = np.concatenate([axis_position_samples, ground_truth_axis_position.reshape(1, -1)], axis=0)
        axis_direction_samples = np.concatenate([axis_direction_samples, ground_truth_axis_direction.reshape(1, -1)], axis=0)


        return axis_position_samples, axis_direction_samples


    def infer_state_from_critic(
            self,
            eef_pose_sequence:np.ndarray,
    ):
        '''
        sample states within the bounds, 
        feed state-action pairs to the critic,
        choose the state with the highest value
        '''
        delta_eef_pose = eef_pose_sequence[1:] - eef_pose_sequence[:-1]

        axis_samples_original = copy.deepcopy(self.axis_samples)
        # make state-action pairs
        axis_samples = np.repeat(self.axis_samples, len(delta_eef_pose), axis=0)
        actions = np.tile(delta_eef_pose, (self.num_samples+1, 1))
        # normalize the actions
        actions = actions / np.linalg.norm(actions, axis=1)[:, None] * 5
        eef_pose_sequence= np.tile(eef_pose_sequence[:-1], (self.num_samples+1, 1))

        # print("Axis Samples: ", axis_samples)

        # calculate the observations
        obs_samples = []
        for i in range(len(eef_pose_sequence)):
            obs = np.concatenate([
                axis_samples[i, 3:],
                (eef_pose_sequence[i] - axis_samples[i, :3] - np.dot(eef_pose_sequence[i] - axis_samples[i, :3], axis_samples[i, 3:]) * axis_samples[i, 3:]),
            ])
            obs_samples.append(obs)
        
        obs_samples = np.array(obs_samples)


        # feed to critic
        states = torch.tensor(obs_samples, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        
        # print("States: ", states)

        values = self.critic(states,actions).detach().cpu().numpy()

        # add the values from each sequence
        values = values.reshape(self.num_samples+1, -1)
        values = values.sum(axis=1)
        print("Values: ", values)

        # choose the state with the highest value
        max_value_idx = np.argmax(values)
        print("Max Value: ", values[max_value_idx])
        axis = axis_samples_original[max_value_idx]

        return axis
    

    def get_ground_truth_state_value(
            self, 
            eef_pose_sequence: np.ndarray,
            ground_truth_axis : np.ndarray
    ):
        '''
        Get the value of the ground truth state
        '''
        delta_eef_pose = eef_pose_sequence[1:] - eef_pose_sequence[:-1]
        
        ground_truth_axis = ground_truth_axis.reshape(1, -1)
        ground_truth_axis = np.repeat(ground_truth_axis, len(delta_eef_pose), axis=0)
        
        ground_truth_obs = []
        for i in range(len(delta_eef_pose)):
            obs = np.concatenate([
                ground_truth_axis[i, 3:],
                (eef_pose_sequence[i] - ground_truth_axis[i,:3] - np.dot(eef_pose_sequence[i] - ground_truth_axis[i,:3], ground_truth_axis[i,3:]) * ground_truth_axis[i,3:]),
            ])
            ground_truth_obs.append(obs)
        
        ground_truth_obs = np.array(ground_truth_obs)
        actions = delta_eef_pose
        actions = actions / np.linalg.norm(actions, axis=1)[:, None] * 5

        states = torch.tensor(ground_truth_obs, dtype=torch.float32).to(self.device)
        # print("ground_truth_obs: ", ground_truth_obs)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)

        values = self.critic(states, actions).detach().cpu().numpy()

        values = values.reshape(len(ground_truth_obs), -1)
        values = values.sum(axis=1)
        print("Ground Truth Value: ", values)

        return values.sum()



