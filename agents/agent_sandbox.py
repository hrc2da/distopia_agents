from bfilter import BayesFilter
from bfagent import GreedyAgent
import itertools
import numpy as np
import sys
def transition_fn(x,y,z):
	return 0.5
def observation_fn(x,y):
	return 0.5
metrics = ['population', 'pvi', 'compactness', 'projected_votes', 'race', 'income', 'area']
five_states = list(itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]))
six_states = list(itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]))
seven_states = list(itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]))
n_steps = sys.argv[1]
with open('task.txt', 'r') as f:
	task = f.readline()
task = np.array([int(i) for i in task.split(',')])
gagent = GreedyAgent()
gagent.set_metrics(metrics)
gagent.set_task(task)
all_designs, all_metrics = gagent.run(n_steps)

bfilter = BayesFilter(seven_states, transition_fn, observation_fn)

with open('belief_file.txt', 'w') as f:
	for metric in all_metrics:
		action = None
		b_star = bfilter.prediction_step(action)
		b = bfilter.observation_step(metric)
		f.write(str("Pred: " + str(b_star)))
		f.write(str("Obs: " + str(b)))
		f.write('\n')