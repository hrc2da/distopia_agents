import json
import numpy as np

results_dict = {}
total_correct = 0
total_eval = 0
total_steps = 0
total_precision = 0
def how_often_did_it_get_it(task, beliefs, threshold = 0.7):
	correct = 0
	for b in beliefs:
		relevant_belief = b[task]
		if threshold is None:
			max_task = list(b.keys())[0]
			max_belief = b[max_task]
			for t,v in b.items():
				if v > max_belief:
					max_belief = v
					max_task = t
			if max_task == task:
				correct += 1
		elif relevant_belief >= threshold:
			correct += 1
	return correct/len(beliefs)

def did_it_get_it(task, beliefs, threshold = 0.7):
	for b in beliefs:
		relevant_belief = b[task]
		if threshold is None:
			max_task = list(b.keys())[0]
			max_belief = b[max_task]
			for t,v in b.items():
				if v > max_belief:
					max_belief = v
					max_task = t
			if max_task == task:
				return True
		elif relevant_belief >= threshold:
			return True
	return False

def when_did_it_get_it(task, beliefs, threshold = 0.7):
	for i,b in enumerate(beliefs):
		relevant_belief = b[task]
		if threshold is None:
			max_task = list(b.keys())[0]
			max_belief = b[max_task]
			for t,v in b.items():
				if v > max_belief:
					max_belief = v
					max_task = t
			if max_task == task:
				return i
		elif relevant_belief >= threshold:
			return i
	return len(beliefs)

with open('results/results_file_1565843345.4031608') as infile:
	for line in infile:
		result = json.loads(line)
		task = result['task']
		taskstr = str(np.array(task, dtype=float))
		#correct = result['correct']
		correct = did_it_get_it(taskstr,result['beliefs'],threshold=None)
		n_steps = when_did_it_get_it(taskstr,result['beliefs'], threshold=None)
		precision = how_often_did_it_get_it(taskstr,result['beliefs'],threshold=None)
		if taskstr in results_dict:
			results_dict[taskstr]['correct'] += correct
			results_dict[taskstr]['total'] += 1
			results_dict[taskstr]['total_steps'] += n_steps
			results_dict[taskstr]['precision'] += precision
		else:
			results_dict[taskstr] = {'correct':0, 'total': 0, 'total_steps': 0, 'precision': 0}
			results_dict[taskstr]['correct'] += correct
			results_dict[taskstr]['total'] += 1
			results_dict[taskstr]['total_steps'] += n_steps
			results_dict[taskstr]['precision'] += precision
		total_correct += correct
		total_eval += 1
		total_steps += n_steps
		total_precision += precision

print("Total Recall: {}/{} = {:.4f}, Avg Precision: {:.4f}, {:.2f} steps".format(total_correct,total_eval,total_correct/total_eval, total_precision/total_eval, total_steps/total_eval))
for task, results in results_dict.items():
	print("Task {}: {}/{} = {:.4f}, {:.4f}, {:.2f} steps".format(task,results['correct'],results['total'],
																	results['correct']/results['total'],
																	results['precision']/results['total'], 
																	results['total_steps']/results['total']))
