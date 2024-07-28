import matplotlib.pyplot as plt 
import json 
import numpy as np
import os


def plot_curriculum(algorithm, log_dir, n_runs): 
    train_rwds = []
    eval_rwds = [] 
    train_projections = [] 
    eval_projections = []
    for r in range(n_runs):
        with open(f'{log_dir}/{algorithm}_reward_train_{r}.json', 'r') as f:
            train_rwds.append(np.array(json.load(f))) 
        with open(f'{log_dir}/{algorithm}_reward_eval_{r}.json', 'r') as f:
            eval_rwds.append(json.load(f)) 
        with open(f'{log_dir}/{algorithm}_projection_train_{r}.json', 'r') as f:
            train_projections.append(json.load(f)) 
        with open(f'{log_dir}/{algorithm}_projection_eval_{r}.json', 'r') as f:
            eval_projections.append(json.load(f)) 
    # plt.plot() 
    # print(train_rwds[0])

    train_rwd_stack = np.vstack(train_rwds) 
    train_rwd_mean = np.mean(train_rwd_stack, axis=0)
    train_rwd_std = np.std(train_rwd_stack, axis=0) 

    eval_rwd_stack = np.vstack(eval_rwds) 
    eval_rwd_mean = np.mean(eval_rwd_stack, axis=0)
    eval_rwd_std = np.std(eval_rwd_stack, axis=0) 

    train_projection_stack = np.vstack(train_projections) 
    train_projection_mean = np.mean(train_projection_stack, axis=0)
    train_projection_std = np.std(train_projection_stack, axis=0) 

    eval_projection_stack = np.vstack(eval_projections) 
    eval_projection_mean = np.mean(eval_projection_stack, axis=0)
    eval_projection_std = np.std(eval_projection_stack, axis=0) 

    fig, axes = plt.subplots(4, 1, figsize=(5*2, 6*2))

    axes[0].plot(train_rwd_mean, label='mean') 
    axes[0].fill_between(range(train_rwd_mean.shape[0]), train_rwd_mean-train_rwd_std, train_rwd_mean+train_rwd_std, alpha=0.5, label='std') 

    axes[0].set_xlabel('episodes')
    axes[0].set_ylabel('train reward')
    axes[0].set_title(f'{algorithm} train reward') 
    axes[0].legend() 

    axes[1].plot(train_projection_mean, label='mean') 
    axes[1].fill_between(range(train_projection_mean.shape[0]), train_projection_mean-train_projection_std, train_projection_mean+train_projection_std, alpha=0.5, label='std') 

    axes[1].set_xlabel('episodes')
    axes[1].set_ylabel('train projection')
    axes[1].set_title(f'{algorithm} train projection') 
    axes[1].legend() 

    axes[2].plot(eval_rwd_mean, label='mean') 
    axes[2].fill_between(range(eval_rwd_mean.shape[0]), eval_rwd_mean-eval_rwd_std, eval_rwd_mean+eval_rwd_std, alpha=0.5, label='std') 

    axes[2].set_xlabel('episodes')
    axes[2].set_ylabel('eval reward')
    axes[2].set_title(f'{algorithm} eval reward') 
    axes[2].legend() 

    axes[3].plot(eval_projection_mean, label='mean') 
    axes[3].fill_between(range(eval_projection_mean.shape[0]), eval_projection_mean-eval_projection_std, eval_projection_mean+eval_projection_std, alpha=0.5, label='std') 

    axes[3].set_xlabel('episodes')
    axes[3].set_ylabel('eval projection')
    axes[3].set_title(f'{algorithm} eval projection') 
    axes[3].legend() 

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    name = "door_increasing_random_td3"
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_dir,"force_experiments/outputs")
    plot(name, log_dir ,1)
    # plot('td3', 10)