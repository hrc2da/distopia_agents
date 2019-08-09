import numpy as np
from bfagent import GreedyAgent

import argparse
import os
import time
import csv

metrics = ['population','pvi','compactness','projected_votes','race','income','area']


tasks = []

for i in range(len(metrics)):
    task_up = np.zeros(len(metrics))
    task_up[i] = 1
    task_down = np.zeros(len(metrics))
    task_down[i] = -1
    tasks.append(task_up)
    tasks.append(task_down)




argparser = argparse.ArgumentParser(description="Run a Greedy Agent task")

argparser.add_argument('task', type=int)
argparser.add_argument('n_steps', type=int)

args = argparser.parse_args()

task_id = args.task
n_steps = args.n_steps


greedy_agent = GreedyAgent(metrics=metrics)
greedy_agent.set_task(tasks[task_id])
designs, metrics = greedy_agent.run(n_steps)


def task_to_str(task_arr):
    task_str = ''
    print(task_arr)
    for w in task_arr:
        if w > 0:
            task_str += '+'
        elif w == 0:
            task_str += '0'
        else:
            task_str += '-'
    return task_str

outfile_name = task_to_str(tasks[task_id])

t = str(time.time())
out_dir = os.path.join('logs',outfile_name)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
with open(os.path.join(out_dir,outfile_name+'_'+t+'_outcomes.csv'), 'w+') as outfile:
    writer = csv.writer(outfile)
    for row in metrics:
        if row is None:
            row = [None]*len(ga.metrics)
        writer.writerow(row)
with open(os.path.join(out_dir,outfile_name+'_'+t+'_designs.csv'), 'w+') as outfile:
    writer = csv.writer(outfile)
    for row in designs:
        writer.writerow(row)