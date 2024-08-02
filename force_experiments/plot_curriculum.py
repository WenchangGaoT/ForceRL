import matplotlib.pyplot as plt 
import json 
import numpy as np
import os


def plot_curriculum(algorithm, log_dir, n_runs, 
                    curricula_schedule, 
                    episode_schedule): 
    train_rwds = [[] for _ in curricula_schedule]
    eval_rwds = [] 
    train_success_rates = [[] for _ in curricula_schedule] 
    eval_success_rates = []
    curriculas = curricula_schedule
    for r in range(n_runs):
        # train_rwds.append([]) 
        # train_success_rates.append([])
        for c_idx, c in enumerate(curricula_schedule):
            with open(f'{log_dir}/{algorithm}_reward_train_{r}_curriculum_{c_idx}.json', 'r') as f:
                train_rwds[c_idx].append(np.array(json.load(f))) 
            # with open(f'{log_dir}/{algorithm}_reward_eval_{r}_curriculum_{c_idx}.json', 'r') as f:
            #     eval_rwds.append(json.load(f)) 
            with open(f'{log_dir}/{algorithm}_success_rate_train_{r}_curriculum_{c_idx}.json', 'r') as f:
                train_success_rates[c_idx].append(json.load(f)) 
            # with open(f'{log_dir}/{algorithm}_success_rate_eval_{r}_curriculum_{c_idx}.json', 'r') as f:
            #     eval_success_rates.append(json.load(f)) 
    # plt.plot() 
    # print(train_rwds[0])

    # train_rwd_stacks = [np.vstack(r) for r in train_rwds] 
    train_rwd_mean = [np.mean(rwd_stack, axis=0) for rwd_stack in train_rwds]
    train_rwd_std = [np.std(rwd_stack, axis=0) for rwd_stack in train_rwds] 
    print(train_rwd_mean[0])
    print(len(train_rwd_mean))
    # print(train_rwds[7]) 
    print(len(train_rwds))

    train_sr_mean = [np.mean(sr_stack, axis=0) for sr_stack in train_success_rates] 
    train_sr_std = [np.std(sr_stack, axis=0) for sr_stack in train_success_rates] 
    print(train_sr_mean[0].shape)

    # eval_rwd_stack = np.vstack(eval_rwds) 
    # eval_rwd_mean = np.mean(eval_rwd_stack, axis=0)
    # eval_rwd_std = np.std(eval_rwd_stack, axis=0) 

    # train_projection_stack = np.vstack(train_projections) 
    # train_projection_mean = np.mean(train_projection_stack, axis=0)
    # train_projection_std = np.std(train_projection_stack, axis=0) 

    # eval_projection_stack = np.vstack(eval_projections) 
    # eval_projection_mean = np.mean(eval_projection_stack, axis=0)
    # eval_projection_std = np.std(eval_projection_stack, axis=0)  

    cur_ep = 0

    fig, axes = plt.subplots(2, 1, figsize=(5*2, 6*2)) 

    train_rwd_mean_plot = np.concatenate(train_rwd_mean)
    train_rwd_std_plot = np.concatenate(train_rwd_std)
    print(train_rwd_mean_plot.shape)
    train_sr_mean_plot = np.concatenate(train_sr_mean)
    train_sr_std_plot = np.concatenate(train_sr_std)


    # for e_idx, e_len in enumerate(episode_schedule):
    axes[0].plot(range(train_rwd_mean_plot.shape[0]), train_rwd_mean_plot, label='mean', color='blue') 
    axes[0].fill_between(range(train_rwd_mean_plot.shape[0]), train_rwd_mean_plot-train_rwd_std_plot, train_rwd_mean_plot+train_rwd_std_plot, alpha=0.5, label='std') 

    axes[0].set_xlabel('episodes')
    axes[0].set_ylabel('train reward')
    axes[0].set_title(f'{algorithm} train reward') 
    axes[0].legend() 

    axes[1].plot(range(train_sr_mean_plot.shape[0]), train_sr_mean_plot, label='mean') 
    axes[1].fill_between(range(train_rwd_std_plot.shape[0]), train_sr_mean_plot-train_sr_std_plot, train_sr_mean_plot+train_sr_std_plot, alpha=0.5, label='std') 

    axes[1].set_xlabel('episodes')
    axes[1].set_ylabel('success rate')
    axes[1].set_title(f'{algorithm} success rate') 
    axes[1].legend() 
    # cur_ep += e_len

        # axes[2].plot(eval_rwd_mean, label='mean') 
        # axes[2].fill_between(range(eval_rwd_mean.shape[0]), eval_rwd_mean-eval_rwd_std, eval_rwd_mean+eval_rwd_std, alpha=0.5, label='std') 

        # axes[2].set_xlabel('episodes')
        # axes[2].set_ylabel('eval reward')
        # axes[2].set_title(f'{algorithm} eval reward') 
        # axes[2].legend() 

    # axes[3].plot(eval_projection_mean, label='mean') 
    # axes[3].fill_between(range(eval_projection_mean.shape[0]), eval_projection_mean-eval_projection_std, eval_projection_mean+eval_projection_std, alpha=0.5, label='std') 

    # axes[3].set_xlabel('episodes')
    # axes[3].set_ylabel('eval projection')
    # axes[3].set_title(f'{algorithm} eval projection') 
    # axes[3].legend() 

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    name = "curriculum_door_continuous_random_point_td3"
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_dir,"force_experiments/outputs")
    plot_curriculum(name, log_dir ,10, 
                    curricula_schedule=[
                                        (-0.25, 0.25),
                                        (-np.pi / 2.0, 0), 
                                        (-np.pi/2.0 - 0.25, -np.pi/2.0 + 0.25),
                                        (-np.pi / 4.0, np.pi / 2.0),
                                        (-np.pi / 2.0, np.pi / 2.0),
                                        (0, np.pi),
                                        (-np.pi, np.pi)
                                        ], 
                    episode_schedule=[100, 100, 100, 200, 200, 300, 300])
    # plot('td3', 10)