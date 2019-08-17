import numpy as np
import argparse

import itertools

argparser = argparse.ArgumentParser(description="Generate a sandbox task file.")

argparser.add_argument('-n','--n_steps', default=100, type=int, help="the number of steps to run each agent")
argparser.add_argument('-r', '--reps', default=1, type=int, help="the number of times to do the whold set")
argparser.add_argument('-d','--dim', default=5, type=int, help="the dimension of the task space (i.e. the number of metric outcomes)")
argparser.add_argument('--one_hots',action="store_true", help="if true, only test one-hots, not full set of tasks")
argparser.add_argument('-o', default="auto_task.txt", help="output filename")
args = argparser.parse_args()

n_steps = args.n_steps
dim = args.dim
one_hots = args.one_hots
filename = args.o
reps = args.reps

if one_hots is True:
    tasks = []
    for i in range(dim):
        task_up = np.zeros(dim)
        task_up[i] = 1
        task_down = np.zeros(dim)
        task_down[i] = -1
        tasks.append(task_up)
        tasks.append(task_down)
else:
    tasks = list(map(np.array,itertools.product(*[[-1., 0., 1.]]*dim)))

task_strings = []
# str is like (1,0,0,-1,0)[100]
for rep in range(reps):
    for task in tasks:
        task_str = '('
        for t in task[:-1]:
            task_str += str(int(t)) + ','
        task_str += str(int(task[-1])) + ')'
        task_str += '[{}]\n'.format(n_steps)
        task_strings.append(task_str)

with open(filename, 'w+') as outfile:
    for task in task_strings:
        outfile.write(task)

