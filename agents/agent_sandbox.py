from bfilter import BayesFilter
from bfagent import GreedyAgent
import itertools
import numpy as np
import sys
import pickle as pkl
from scipy.stats import multivariate_normal

emissions_file = '/home/dev/research/distopia/logs/conditional_one_hots.pkl'

with open(emissions_file, 'rb') as em_file:
	emissions_probs = pkl.load(em_file)

normalization_file = '/home/dev/research/distopia/logs/normalization.pkl'

with open(normalization_file,'rb') as z_file:
	means, stds = pkl.load(z_file)
	means = np.array(means)
	stds = np.array(stds)

def transition_fn(x,y,z):
	return 1/14
def observation_fn(observation,task):
	means,cov = emissions_probs[str(list(task))]
	return multivariate_normal.pdf(observation,mean=means,cov=cov)
def standardize(metric_arr):
	return (metric_arr - means) / stds
	
metrics = ['population', 'pvi', 'compactness', 'projected_votes', 'race', 'income', 'area']
five_states = list(itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]))
six_states = list(itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]))
seven_states = list(itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]))

tasks = []

for i in range(len(metrics)):
    task_up = np.zeros(len(metrics))
    task_up[i] = 1
    task_down = np.zeros(len(metrics))
    task_down[i] = -1
    tasks.append(task_up)
    tasks.append(task_down)


n_steps = int(sys.argv[1])
with open('task.txt', 'r') as f:
	task = f.readline()
task = np.array([int(i) for i in task.split(',')])
gagent = GreedyAgent()
gagent.set_metrics(metrics)
gagent.set_task(task)
all_designs, all_metrics = gagent.run(n_steps)

bfilter = BayesFilter(tasks, transition_fn, observation_fn)

with open('belief_file.txt', 'w') as f:
	for metric in all_metrics:
		action = None
		#b_star = bfilter.prediction_step(action)
		b = bfilter.observation_step(standardize(metric))
		#f.write(str("Pred: " + str(b_star)))
		f.write(str("Obs: " + str(b)))
		f.write('\n')