from matplotlib import pyplot as plt
import pickle as pkl
import numpy as np
from sklearn.neighbors.kde import KernelDensity

metric_names = ['population','pvi','compactness','projected_votes','race']

def get_kdes(training_data):
    kde_dict = {}
    for task_str,task_data in training_data.items():
        task_kde = KernelDensity(kernel = 'gaussian', bandwidth=0.2).fit(task_data)
        kde_dict[task_str] = task_kde #task_obs_fun
    return kde_dict

def plot_kde(estimator,n_points=100):
    X = np.linspace(n_points)

def plot_hist(data_dict,task_strings=[],kdes=None):
    '''plots the distribution of each dimensions across data for a task.
    If no task is given, plot over entire dataset.
    '''

    if len(task_strings) < 1:
        pass
    else:
        num_metrics = list(data_dict.values())[0].shape[1]
        fig, ax = plt.subplots(len(task_strings),num_metrics, figsize=(20,4*len(task_strings)), sharex=True)
        if len(ax.shape) < 2:
            # hack to to index the axes if there's only one task being plotted
            ax =np.array((ax,))
        for i,task_str in enumerate(task_strings):
            raw_data = data_dict[task_str]
            # if kdes is not None:
            #     kde = kdes[task_str]
            #     kde_estimates = np.exp(kde.score_samples(raw_data))
            for metric in range(num_metrics):
                ax_ = ax[i,metric]
                data = raw_data[:,metric]
                if metric == 0:
                    ax_.set_ylabel(task_str)
                ax_.hist(data)
                ax_.xaxis.set_tick_params(labelbottom=True)
                ax_.set_xlabel(metric_names[metric])
                kde = KernelDensity(kernel = 'gaussian', bandwidth=0.2).fit(data.reshape(-1,1))
                X = np.linspace(np.min(data),np.max(data),100).reshape(-1,1)
                ax_.plot(X,len(data)*np.exp(kde.score_samples(X)))
                # if kdes is not None:
                #     ax[i,metric].plot(raw_data[:,metric],kde_estimates)
                    
        plt.show()

        

if __name__=='__main__':
    with open('resources/243_raw_data.pkl','rb') as infile:
        raw_data = pkl.load(infile)
    with open('resources/stripped_normalization.pkl','rb') as infile:
        means, stds = pkl.load(infile)
    normalized_data = {task_str : (data - means)/stds for task_str,data in raw_data.items()}
    kde_dict = get_kdes(normalized_data)
    sample_task = list(normalized_data.keys())[np.random.randint(len(normalized_data.keys()))]
    sample_task1 = list(normalized_data.keys())[np.random.randint(len(normalized_data.keys()))]
    plot_hist(normalized_data,[sample_task, sample_task1],kde_dict)