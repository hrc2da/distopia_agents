from bfilter import BayesFilter
from bfagent import GreedyAgent
import itertools
import numpy as np
import sys
import pickle as pkl
from scipy.stats import multivariate_normal

emissions_file = 'resources/conditional_one_hots.pkl'

with open(emissions_file, 'rb') as em_file:
    emissions_probs = pkl.load(em_file)

normalization_file = 'resources/normalization.pkl'

with open(normalization_file, 'rb') as z_file:
    means, stds = pkl.load(z_file)
    means = np.array(means)
    stds = np.array(stds)


def transition_fn(x, y, z):
    return 1/14


def observation_fn(observation, task):
    means, cov = emissions_probs[str(list(task))]
    return multivariate_normal.pdf(observation, mean=means, cov=cov)


def standardize(metric_arr):
    return (metric_arr - means) / stds


def parse_task(task_desc):
    tasks = []
    if "|" in task_desc:
        split_task_descs = task_desc.split("|")
    else:
        split_task_descs = [task_desc]
    for task_desc in split_task_descs:
        # get task
        task_str = task_desc[task_desc.index("(")+1:task_desc.index(")")]
        task = np.array([int(i) for i in task_str.split(',')])
        # get string
        if "[" in task_desc and "]" in task_desc:
            steps = int(task_desc[task_desc.index("[")+1:task_desc.index("]")])
        else:
            steps = 35
        tasks.append((task, steps))
    return tasks


def stringify_task(task):
    s = "["
    for e in task:
        s += str(e) + ","
    s = s[:-1] + "]"
    return s


metrics = ['population', 'pvi', 'compactness', 'projected_votes', 'race', 'income', 'area']
five_states = list(itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]))
six_states = list(itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]))
seven_states = list(itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1],
                                      [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]))

tasks = []

for i in range(len(metrics)):
    task_up = np.zeros(len(metrics))
    task_up[i] = 1
    task_down = np.zeros(len(metrics))
    task_down[i] = -1
    tasks.append(task_up)
    tasks.append(task_down)

all_tasks = []
with open('task.txt', 'r') as f:
    for task_desc in f:
        given_tasks = parse_task(task_desc)
        all_tasks.append(given_tasks)

gagent = GreedyAgent()
gagent.set_metrics(metrics)

for task_trajectory in all_tasks:
    task_path = ""
    all_designs = []
    all_metrics = []
    for task, steps in task_trajectory:
        task_path += stringify_task(task) + "(" + str(steps) + ")>"
        gagent.set_task(task)
        designs, metrics = gagent.run(steps)
        all_designs += designs
        all_metrics += metrics

    bfilter = BayesFilter(tasks, transition_fn, observation_fn)
    with open('results/'+task_path[:-1]+'_belief_file.txt', 'w') as f:
        for metric in all_metrics:
            action = None
            #b_star = bfilter.prediction_step(action)
            b = bfilter.observation_step(standardize(metric))
            #f.write(str("Pred: " + str(b_star)))
            f.write(str("Obs: " + str(b)))
            f.write('\n')
