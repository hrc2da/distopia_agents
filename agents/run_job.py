import numpy as np
from bfagent import GreedyAgent

import argparse
import os
from multiprocessing import Pool, Queue
from tqdm import tqdm
from threading import Thread
import itertools

metrics = ['population', 'pvi', 'compactness', 'projected_votes', 'race']


tasks = []

for i in range(len(metrics)):
    task_up = np.zeros(len(metrics))
    task_up[i] = 1
    task_down = np.zeros(len(metrics))
    task_down[i] = -1
    tasks.append(task_up)
    tasks.append(task_down)

tasks = tasks * 9

#tasks = list(map(np.array,itertools.product([-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.])))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Run a Greedy Agent task")

    argparser.add_argument('n_steps', type=int)
    args = argparser.parse_args()

    n_steps = args.n_steps
    status_queue = Queue()

    def progress_monitor():
        for i in tqdm(range(len(tasks) * n_steps)):
            status_queue.get()

    def process_task(i, task):
        greedy_agent = GreedyAgent(metrics=metrics)
        greedy_agent.set_task(task)

        out_dir = os.path.join(os.path.dirname(__file__), 'logs')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        task_str = ','.join(map(str, task))
        with open(os.path.join(out_dir, task_str + '_exc_log_'+str(i)), 'w+') as exc_logger:
            with open(os.path.join(out_dir, task_str + '_log_'+str(i)), 'w+') as logger:
                return greedy_agent.run(n_steps, logger, exc_logger, status_queue)


    thread = Thread(target=progress_monitor)
    thread.start()

    with Pool(96) as pool:
        results = pool.starmap(process_task, list(enumerate(tasks)))

    for res, config in results:
        print('{}: {}'.format(','.join(map(str, config)), res))
