from bfilter import BayesFilter
from bfagent import GreedyAgent
import itertools
import numpy as np
import sys
import pickle as pkl
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, Queue, Manager
from threading import Thread
from tqdm import tqdm
from sklearn.neighbors.kde import KernelDensity
import json

raw_deltas_file = 'resources/stripped_raw_deltas_not_standardized.pkl'

emissions_file = 'resources/conditional_one_hots.pkl'
emissions_file = 'resources/stripped_emissions.pkl'
combined_emissions_file = 'resources/combined_emissions.pkl'

#deltas_emissions_file = 'resources/conditional_one_hots_deltas.pkl'
deltas_emissions_file = 'resources/conditional_one_hots_deltas_standardized_wrt_observations.pkl'


data_dict_file = 'resources/raw_data.pkl'

with open(data_dict_file, 'rb') as data_file:
    training_data = pkl.load(data_file)

with open(emissions_file, 'rb') as em_file:
	emissions_probs = pkl.load(em_file)

with open(deltas_emissions_file, 'rb') as dem_file:
	deltas_emissions_probs = pkl.load(dem_file)

with open(combined_emissions_file, 'rb') as dem_file:
	combined_emissions_probs = pkl.load(dem_file)

# with open(raw_deltas_file, 'rb') as rdem_file:
# 	raw_deltas = pkl.load(rdem_file)
# 	deltas_emissions_probs = {task: (np.mean(metrics,0),np.cov(metrics.T)) for task,metrics in raw_deltas.items()}
# 	for k,v in deltas_emissions_probs.items():
		
# 		print(np.linalg.matrix_rank(v[1]))
	

normalization_file = 'resources/normalization.pkl'
normalization_file = 'resources/stripped_normalization.pkl'
with open(normalization_file, 'rb') as z_file:
	means, stds = pkl.load(z_file)
	metric_means = np.array(means)
	metric_stds = np.array(stds)

deltas_normalization_file = 'resources/deltas_normalization.pkl'

with open(deltas_normalization_file, 'rb') as z_file:
	dmeans, dstds = pkl.load(z_file)
	delta_means = np.array(dmeans)
	delta_stds = np.array(dstds)

combined_normalization_file = 'resources/combined_normalization.pkl'

with open(combined_normalization_file, 'rb') as z_file:
	cmeans, cstds = pkl.load(z_file)
	combined_means = np.array(cmeans)
	combined_stds = np.array(cstds)


def metric_standardize(metric_arr):
	return (metric_arr - metric_means) / metric_stds

def delta_standardize(metric_arr):
	return (metric_arr - delta_means) / delta_stds

def combined_standardize(combined_arr):
	return (combined_arr - combined_means) / combined_stds


kde_dict = {}
for task in training_data.keys():
	task_str = str(np.array(eval(task)))
	task_data = training_data[str(task)]
	task_data = metric_standardize(task_data)
	task_kde = KernelDensity(kernel = 'gaussian', bandwidth=0.2).fit(task_data)
	task_obs_fun = lambda x : np.exp(task_kde.score(x))
	kde_dict[task_str] = task_kde #task_obs_fun


def transition_fn(initial_state, resulting_state, action):
	return 1/14

def deltas_transition_fn(initial_state, resulting_state, action):
	means, cov = deltas_emissions_probs[str(list(resulting_state))]
	return multivariate_normal.pdf(action, mean=means, cov=cov)

# def observation_fn(observation, task, dx = 0.01):
#     means, cov = emissions_probs[str(list(task))]
#     return multivariate_normal.pdf(observation, mean=means, cov=cov)

'''
Kernel Density Estimation Observation FN
'''

# def observation_fn(observation, task, dx = 0.01):
#     return kde_dict[str(task)]

def observation_fn(observation, task, dx = 0.01):
    return np.exp(kde_dict[str(task)].score(observation.reshape(1,5))) #(observation.reshape(1,5))





def deltas_observation_fn(observation, task):
	means, cov = deltas_emissions_probs[str(list(task))]
	print(task)
	import pdb; pdb.set_trace()
	print(np.linalg.matrix_rank(cov))
	return multivariate_normal.pdf(observation, mean=means, cov=cov)

def combined_observation_fn(observation, task):
	means, cov = combined_emissions_probs[str(list(task))]
	return multivariate_normal.pdf(observation, mean=means, cov=cov)

def construct_observation(observation, delta):
	'''Construct an observation from metrics plus deltas

	Takes raw values, standardizes, and concats
	'''
	return np.concat([metric_standardize(observation),metric_standardize(delta)])

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

	
	

def plot_test(task, obs_metrics, beliefs):
		# plot task posteriors
	task = str(list(map(float, task)))
	fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11,8), gridspec_kw={'height_ratios': [1, 2, 2]})

	stats = emissions_probs[task]
	m, cov = stats
	print("{}:{}\n".format(task, str(m)))
	ax1.bar(np.arange(len(m)), m)
	ax1.set_title("Observation Means for Task {}".format(task))
	# ax1.title(metric_names)

	# plot metrics from run
	colors = ['b','g','r','c','y','m','k']
	obs = {k: [] for k in metric_names}
	for metric_step in obs_metrics:
		s_metric_step = metric_standardize(metric_step)
		for i in range(len(metric_step)):
			obs[metric_names[i]].append(s_metric_step[i])
	for i, (k, v) in enumerate(obs.items()):
		ax2.plot(v, label=k, color=colors[i])
	ax2.legend(loc="upper left", bbox_to_anchor=(1,1))
	ax2.set_title("Observation Trace (z-normalized)")

	# plot model's beliefs

	separated_b = {k: [v] for k, v in beliefs[0].items()}
	for step in beliefs[1:]:
		for k, v in step.items():
			separated_b[k].append(v)
	markers = ['.','D']
	fill = ['full','none']
	linestyles = ['-','--']
	for i, (key, data) in enumerate(separated_b.items()):
		if np.max(data) < 0.1:
			key = str() #don't add zeroish curves to the legend
		ax3.plot(data, label=key, color = colors[i//2%len(colors)],linestyle=linestyles[i%2])
		
	ax3.legend(loc="upper left", fontsize='small', bbox_to_anchor=(1,1))
	ax3.set_title("Belief Trace")

	plt.tight_layout()
	plt.savefig(task+"_"+str(time.time())+".png")

def did_it_get_it(task, beliefs, threshold = 0.7):
	for b in beliefs:
		relevant_belief = b[task]
		if relevant_belief >= threshold:
			return True
	return False


metrics = ['population', 'pvi', 'compactness', 'projected_votes', 'race']#, 'income', 'area']
metric_names = ['population', 'pvi', 'compactness', 'projected_votes', 'race']#, 'income', 'area']
#five_states = list(itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]))
#six_states = list(itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]))
#seven_states = list(itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1],
#									  [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]))
tasks = list(map(np.array,itertools.product([-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.])))
# tasks = []

# for i in range(len(metric_names)):
# 	task_up = np.zeros(len(metric_names))
# 	task_up[i] = 1
# 	task_down = np.zeros(len(metric_names))
# 	task_down[i] = -1
# 	tasks.append(task_up)
# 	tasks.append(task_down)

all_tasks = []
with open('task.txt', 'r') as f:
	for task_desc in f:
		given_tasks = parse_task(task_desc)
		all_tasks.append(given_tasks)

gagent = GreedyAgent()
gagent.set_metrics(metrics)



#correct = 0
def process_task(task_trajectory,results_queue):
#for task_trajectory in all_tasks:
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
	beliefs = []
	beliefsappend = beliefs.append
	with open('results/'+task_path[:-1]+'_belief_file_'+str(time.time())+'.txt', 'w') as f:
		last_metric = all_metrics[0] #standardize(all_metrics[0])
		for metric in all_metrics[1:]:
			#action = None
			cur_metric = metric_standardize(metric) #standardize(metric)
			#delta = cur_metric - last_metric
			#combined = combined_standardize(np.concatenate([cur_metric,delta]))
			last_metric = cur_metric
			#b_star = bfilter.prediction_step(delta)
			#b = bfilter.observation_step(cur_metric)
			b = bfilter.observation_step(cur_metric)

			#f.write(str("Pred: " + str(b_star)))
			f.write(str("Obs: " + str(b)))
			f.write('\n')
			beliefsappend(b)
	correct = did_it_get_it(str(np.array(task,dtype=float)),beliefs)
	if results_queue is not None:
		results_queue.put({'correct': correct, 'task':task.tolist(),'metrics':np.array(metrics).tolist(),'beliefs':beliefs})
	plot_test(task, all_metrics, beliefs)
	print("Correct: {}".format(correct))


m = Manager()
results_queue = m.Queue()
def progress_monitor():
	num_correct = 0
	with open('results/results_file_{}'.format(time.time()), 'w+') as results_file:
		for i in tqdm(range(len(all_tasks))):
			result = results_queue.get()
			#print(result)
			results_file.write(json.dumps(result)+'\n')
			num_correct += result['correct']
	print("************************************")
	print("Total Correct: {}".format(num_correct))
	print("************************************")

if len(all_tasks) == 1:
		process_task(all_tasks[0],None)
else:
	thread = Thread(target=progress_monitor)
	thread.start()

	queued_tasks = [(task, results_queue) for task in all_tasks]

	with Pool(8) as pool:
		results = pool.starmap(process_task, (queued_tasks))
