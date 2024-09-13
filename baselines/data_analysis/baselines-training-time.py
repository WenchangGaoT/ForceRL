import os
import numpy as np
import tensorboard
from tensorboard.backend.event_processing import event_accumulator
import fnmatch


def extract_training_duration(tensorboard_file):
    print("extracting training duration from", tensorboard_file)
    ea = event_accumulator.EventAccumulator(tensorboard_file)
    ea.Reload()
    
    # Extract all the events
    wall_times = ea.Scalars('rollout/ep_rew_mean')  # You can use any tag here
    if not wall_times:
        return None

    # Get the start and end wall clock time
    start_time = wall_times[0].wall_time
    end_time = wall_times[-1].wall_time
    
    # Return the duration in seconds
    return end_time - start_time

def get_avg_std_training_time(logdir, joint_type = "prismatic"):
    all_durations = []
    
    # Iterate over all tensorboard logs in the directory
    for root, dirnames, files in os.walk(logdir):
        # print("dirnames", dirnames)
        for file in files:
            if fnmatch.fnmatch(file, 'events.out.tfevents.*'):
                rel_path = os.path.relpath(root, logdir)
                print("rel_path", rel_path)
                path_parts = rel_path.split(os.sep)
                if joint_type in rel_path and (("trial" in rel_path) or ("trail" in rel_path)):
                    tensorboard_file = os.path.join(root, file)
                    duration = extract_training_duration(tensorboard_file)
                    if duration is not None:
                        all_durations.append(duration)
    
    # Convert to numpy array for easier computation
    all_durations = np.array(all_durations)

    # Compute the average and standard deviation
    mean_duration = np.mean(all_durations)
    std_duration = np.std(all_durations)
    
    return mean_duration, std_duration
# Example usage:
baseline_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_directory = os.path.join(baseline_directory, "logs")
mean, std = get_avg_std_training_time(log_directory, joint_type="revolute")
print("Average Training Time:", mean)
print("Standard Deviation of Training Time:", std)
