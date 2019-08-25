import glob
import re
import pickle as pkl
import numpy as np
import argparse

argparser = argparse.ArgumentParser(description="pickle the raw data")
argparser.add_argument('-o','--outfile', help='the filename to print the pkl to')
args = argparser.parse_args()
outf_name = args.outfile

data = {}
designs = {}
for logfile in glob.glob("logs/*[0-9]_log*"):
	raw_task_str = re.search(r'[\- 0-9 \. \,]+',logfile).group()
	
	task_arr = list(map(float,raw_task_str.split(",")))
	task_str = str(task_arr)
	
	task_data = []
	task_data_append = task_data.append
	task_designs = []
	task_designs_append = task_designs.append
	with open(logfile) as infile:
		for line in infile:
			step_data = list(map(float,eval(line)[2]))
			design = eval(line)[3]
			task_data_append(step_data)
			task_designs_append(design)
	if task_str not in data.keys():
		data[task_str] = np.array(task_data)
		designs[task_str] = task_designs
	else:
		data[task_str] = np.concatenate([data[task_str],task_data])
		designs[task_str] += task_designs

print(data.keys())
print([v.shape for v in data.values()])
with open("resources/{}".format(outf_name),"wb+") as outfile:
	pkl.dump((data,designs),outfile)
