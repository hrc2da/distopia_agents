import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
from os import listdir

emissions_file = '../agents/resources/conditional_one_hots.pkl'

with open(emissions_file, 'rb') as em_file:
    emissions_probs = pkl.load(em_file)

normalization_file = '../agents/resources/normalization.pkl'

with open(normalization_file, 'rb') as z_file:
    means, stds = pkl.load(z_file)
    means = np.array(means)
    stds = np.array(stds)


def standardize(metric_arr):
    return (metric_arr - means) / stds


def load_data(fp):
    task_data = []
    task_data_append = task_data.append
    with open(fp) as infile:
        for line in infile:
            step_data = np.array(list(map(float, eval(line)[2])))
            task_data_append(standardize(step_data))
            # step_data = list(map(float, eval(line)[2]))
            # task_data_append(step_data)
    return np.array(task_data)


def fp_to_key(fp):
    s = fp[:-4]
    s_list = s.split(",")
    f_list = list(map(float, s_list))
    active_i = 0
    return str(f_list)


def gaussian_1d(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def plot_task_obs(data, fp, key):
    means, cov = emissions_probs[key]
    fig, axs = plt.subplots(3, 3)
    for i in range(7):
        n, bins, _ = axs[int(i / 3), i % 3].hist(data[:, i], bins=20)
        domain = np.linspace(np.min(bins)-0.2, np.max(bins)+0.2, 120)
        axs[int(i / 3), i % 3].plot(domain, np.max(n) * gaussian_1d(domain, means[i], np.sqrt(cov[i, i])))
    # plt.show()
    plt.savefig(fp + "_hist.png")


fps = listdir("data/logs/raw")
for fp in fps:
    key = fp_to_key(fp)
    data = load_data("data/logs/raw/" + fp)
    plot_task_obs(data, "data/logs/graphs/" + fp, key)

# fp = "../agents/resources/stripped_raw_deltas_not_standardized.pkl"
# with open(fp, "rb") as infile:
#     data = pkl.load(infile)
# for key, data in data.items():
#     plot_task_obs(data, "data/logs/graphs/"+key+"_deltas", key)
