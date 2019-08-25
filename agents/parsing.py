import pickle as pkl
import numpy as np
import argparse

def is_reset(old_design, new_design):
	'''Detect if there is more than one difference between designs
  
	Returns True if this is a resest, False if not
	'''
	change_counter = 0
	# loop through the new design and count changes from old design
	for k,v in new_design.items():
		for block in v:
			if block not in old_design[k]: # not sure if this is the right compare
				change_counter += 1 
			if change_counter > 1:
				return True
	return False

def get_deltas_no_resets(data,designs):
	deltas = {}
	for k,v in data.items():
		d = designs[k]
		deltas[k] = []
		last_metrics = v[0]
		last_design = d[0]
		assert type(last_metrics) == np.ndarray
		for i,vect in enumerate(v[1:],1):
			cur_design = d[i]
			if is_reset(last_design,cur_design):
				last_design = cur_design
				last_metrics = vect
				continue
			delta = vect-last_metrics
			if np.linalg.norm(delta,1) > 0:
				deltas[k].append(vect-last_metrics)
			# update either way (it's a wash if metrics hasn't changed)
			last_metrics = vect
			last_design = cur_design
		deltas[k] = np.array(deltas[k])
		print(deltas[k].shape)	
	return deltas

def get_combined_no_resets(data,designs):
	combined = {}
	for k,v in data.items():
		d = designs[k]
		combined[k] = []
		last_metrics = v[0]
		last_design = d[0]
		assert type(last_metrics) == np.ndarray
		for i,vect in enumerate(v[1:],1):
			cur_design = d[i]
			if is_reset(last_design,cur_design):
				last_design = cur_design
				last_metrics = vect
				continue
			delta = vect-last_metrics
			if np.linalg.norm(delta,1) > 0:
				combined_vect = np.concatenate([vect,delta])
				assert combined_vect.shape[0] == len(vect) + len(delta)
				combined[k].append(combined_vect)
				
			# update either way (it's a wash if metrics hasn't changed)
			last_metrics = vect
			last_design = cur_design
		combined[k] = np.array(combined[k])
		print(combined[k].shape)	
	return combined


filename = 'resources/one_hot_raw.pkl'
filename = 'resources/five_norms_raw.pkl'
parser = argparse.ArgumentParser("parse a globbed pickle")
parser.add_argument('-f','--file', help='the input file')

args = parser.parse_args()
filename = args.file

with open(filename, 'rb') as infile:
	data,designs = pkl.load(infile)
	combined = get_combined_no_resets(data,designs)
	all_data = np.concatenate(list(combined.values()))
	pop_mean = np.mean(all_data,0)
	pop_std = np.std(all_data,0)
	standardized_data = {task: (c - pop_mean)/pop_std for task,c in combined.items()}
	emissions = {task: (np.mean(c,0),np.cov(c.T)) for task,c in standardized_data.items()}
with open('resources/combined_emissions.pkl', 'wb+') as outfile:
	pkl.dump(emissions,outfile)
with open('resources/combined_normalization.pkl', 'wb+') as outfile:
	pkl.dump((pop_mean,pop_std),outfile)
with open('resources/combined_raw_data.pkl', 'wb+') as outfile:
	pkl.dump(combined,outfile)
with open('resources/raw_data.pkl', 'wb+') as outfile:
	pkl.dump({t:c[:,:5] for t,c in combined.items()},outfile)
metric_emissions = {str(eval(task)[:5]): (np.mean(c[:,:5],0),np.cov(c[:,:5].T)) for task,c in standardized_data.items()}
with open('resources/stripped_emissions.pkl', 'wb+') as outfile:
	pkl.dump(metric_emissions,outfile)
with open('resources/stripped_normalization.pkl', 'wb+') as outfile:
	pkl.dump((pop_mean[:len(pop_mean)//2],pop_std[:len(pop_std)//2]),outfile)
