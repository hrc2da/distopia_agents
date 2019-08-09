from distopia.app.agent import VoronoiAgent
from distopia.mapping._voronoi import ColliderException
from random import randint
import numpy as np
from copy import deepcopy


class DistopiaAgent:
    def reset(self):
        raise NotImplementedError

    def set_task(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError


def gencoordinates(m, n, j, k, seen=None):
    '''Generate random coordinates in range x: (m,n) y:(j,k)

    instantiate generator and call next(g)

    based on:
    https://stackoverflow.com/questions/30890434/how-to-generate-random-pairs-of-
    numbers-in-python-including-pairs-with-one-entr
    '''
    if seen is None:
        seen = set()
    x, y = randint(m, n), randint(j, k)
    while len(seen) < (n + 1 - m)**2:
        while (x, y) in seen:
            x, y = randint(m, n), randint(m, n)
        seen.add((x, y))
        yield (x, y)
    return


# TODO: need to update occupied when changing state
class GreedyAgent(DistopiaAgent):
    # scalar_value is mean over districts
    # scalar_std is standard deviation between districts
    # scalar_maximum is max over districts
    # s is state metric object (for this metric)
    # d is list of district objects (for all metrics)
    metric_extractors = {

        #overall normalization plan: run one-hots in either direction to get rough bounds
        # then z-normalize and trim on edges
        #'population' : lambda s,d : s.scalar_std,
        # standard deviation of each district's total populations (-1)
        # normalization: [0, single_district std] 
        'population' : lambda s,d : np.std([dm.metrics['population'].scalar_value for dm in d]),
        # mean of district margin of victories (-1)
        # normalization: [0,1]
        'pvi' : lambda s,d : s.scalar_maximum,
        # minimum compactness among districts (maximize the minimum compactness, penalize non-compactness) (+1)
        # normalization: [0,1]
        'compactness' : lambda s,d : np.min([dm.metrics['compactness'].scalar_value for dm in d]),
        # mean ratio of democrats over all voters in each district (could go either way)
        # normalization: [0,1]
        'projected_votes' : lambda s,d : np.mean([dm.metrics['projected_votes'].scalar_value/dm.metrics['projected_votes'].scalar_maximum for dm in d]),
        # std of ratio of nonminority to minority over districts
        # normalization: [0, ]
        'race' : lambda s,d : np.std([dm.metrics['race'].scalar_value/dm.metrics['race'].scalar_maximum for dm in d]),
        # scalar value is std of counties within each district. we take a max (-1) to minimize variance within district (communities of interest)
        'income' : lambda s,d : np.max([dm.metrics['income'].scalar_value for dm in d]),
        #'education' : lambda s,d : s.scalar_std,

        # maximum sized district (-1) to minimize difficulty of access
        # normalization [0,size of wisconsin]
        'area' : lambda s,d: s.scalar_maximum
    }

    def __init__(self, x_lim=(100, 900), y_lim=(100, 900),
                    step_size=5, step_min=2, step_max=50,
                    metrics=[], task=[]):
        self.x_min, self.x_max = x_lim
        self.y_min, self.y_max = y_lim
        self.step = step_size
        self.step_min = step_min
        self.step_max = step_max
        self.occupied = set()
        self.coord_generator = gencoordinates(self.x_min, self.x_max,
                                              self.y_min, self.y_max, self.occupied)
        self.evaluator = VoronoiAgent()
        self.evaluator.load_data()
        if metrics == []:
            self.set_metrics(self.evaluator.metrics)
        else:
            for m in metrics:
                assert m in self.evaluator.metrics
            self.set_metrics(metrics)
        if task == []:
            self.set_task([1 for i in range(len(self.metrics))])
        else:
            assert len(task) == len(self.metrics)
            self.set_task(task)


    def set_metrics(self, metrics):
        '''Define an array of metric names
        '''
        self.metrics = metrics

    def set_task(self, task):
        self.reward_weights = task

    def reset(self, initial=None, n_districts=8, max_blocks_per_district=5):
        '''Initialize the state randomly.
        '''
        self.occupied.clear()
        if initial is not None:
            self.state = initial
            return self.state

        else:
            self.state = {}
            # Place one block for each district, randomly
            for i in range(n_districts):
                self.state[i] = [next(self.coord_generator)]

            # add more blocks...
            for i in range(n_districts):
                # get all blocks used in districts that aren't the current one
                other_blocks = []
                for k, v in self.state.items():
                    if k != i:
                        other_blocks += v
                other_blocks = np.array(other_blocks)
                # generate at most max_blocks_per_district new blocks per district
                for j in range(randint(0, max_blocks_per_district-1)):
                    district_blocks = np.array(self.state[i])
                    district_centroid = np.mean(district_blocks, axis=0)
                    distances = np.sqrt(np.sum(np.square(other_blocks - district_centroid), axis=1))
                    closest_pt = other_blocks[np.argmin(distances)]
                    new_block = (closest_pt + district_centroid)/2
                    new_block_coords = (new_block[0], new_block[1])
                    self.state[i].append(new_block_coords)
                    self.occupied.add(new_block_coords)
            return self.state

    def get_neighborhood(self, n_steps):
        '''Get all the configs that have one block n_steps away from the current
        '''
        neighborhood = []
        state = self.state
        for district_id, district in state.items():
            for block_id, block in enumerate(district):
                neighborhood += self.get_neighbors(district_id, block_id)
        return neighborhood

    def get_sampled_neighborhood(self, n_blocks, n_directions, resample=False):
        '''Sample n_blocks * n_direction neighbors.
        
        take n blocks, and move each one according to m direction/angle pairs
        ignore samples that are prima facie invalid (out of bounds or overlaps)
        if resample is true, then sample until we have n_blocks * n_directions
        otherwise, just try that many times.
        '''
        neighbors = []
        n_districts = len(self.state)
        for i in range(n_blocks):
            # sample from districts, then blocks
            # this biases blocks in districts with fewer blocks
            # i think this is similar to how humans work however
            district_id = np.random.randint(n_districts)
            district = self.state[district_id]
            block_id = np.random.randint(len(district))
            x,y = district[block_id]
            for j in range(n_directions):
                mx,my = self.get_random_move(x,y)
                valid_move = self.check_boundaries(mx,my)
                if valid_move:
                    neighbor = deepcopy(self.state)
                    neighbor[district_id][block_id] = (mx, my)
                    neighbors.append(neighbor)
                elif resample == True:
                    # don't use this yet, need to add a max_tries?
                    while not valid_move: 
                        mx,my = self.get_random_move(x,y)
                        valid_move = self.check_boundaries(mx,my)
        return neighbors

    def get_random_move(self, x, y):
        dist,angle = (np.random.randint(self.step_min, self.step_max),
                        np.random.uniform(2*np.pi))
        return (x + np.cos(angle) * dist, y + np.sin(angle) * dist)

    def check_boundaries(self, x, y):
        '''Return true if inside screen boundaries
        '''
        if x < self.x_min or x > self.x_max:
            return False
        if y < self.y_min or y > self.y_max:
            return False
        return True

    def get_neighbors(self, district, block):
        '''Get all the designs that move "block" by one step.


        ignores moves to coords that are occupied or out of bounds
        '''
        neighbors = []

        moves = [np.array((self.step, 0)), np.array((-self.step, 0)),
                 np.array((0, self.step)), np.array((0, -self.step))]

        constraints = [lambda x, y: x < self.x_max,
                        lambda x, y: x > self.x_min,
                        lambda x, y: y < self.y_max,
                        lambda x, y: y > self.y_min]

        x, y = self.state[district][block]

        for i, move in enumerate(moves):
            mx, my = (x, y) + move
            if constraints[i](mx, my) and (mx, my) not in self.occupied:
                new_neighbor = deepcopy(self.state)
                new_neighbor[district][block] = (mx, my)
                neighbors.append(new_neighbor)

        return neighbors

    def check_legal_districts(self, districts):
        if len(districts) == 0:
            return False
        # TODO: consider checking for len == 8 here as well
        for d in districts:
            if len(d.precincts) == 0:
                return False
        return True

    def get_metrics(self, design):
        '''Get the vector of metrics associated with a design 

        returns m-length np array
        '''
        try:
            districts = self.evaluator.get_voronoi_districts(design)
            state_metrics, districts = self.evaluator.compute_voronoi_metrics(districts)
            if self.check_legal_districts(districts) == False:
                return None
            metric_dict = {}
            for state_metric in state_metrics:
                metric_name = state_metric.name
                if metric_name in self.metrics:
                    metric_dict[metric_name] = self.metric_extractors[metric_name](state_metric, districts)
            metrics = np.array([metric_dict[metric] for metric in self.metrics])
            #metrics = np.array([self.metric_extractors[metric](state_metrics, districts) for metric in self.metrics])
            return metrics
        except ColliderException as e:
            print(e)
            return None

    def get_reward(self, metrics):
        '''Get the scalar reward associated with metrics
        '''
        if metrics is None:
            return float("-inf")
        else:
            return np.dot(self.reward_weights, metrics)

    def run(self, n_steps, initial=None, eps=0.9, eps_decay=0.9, eps_min=0.1, n_tries_per_step = 5):
        '''runs for n_steps and returns traces of designs and metrics
        '''
        self.reset(initial)
        all_designs = []
        all_metrics = []
        i = 0
        last_reward = float("-inf")
        no_valids = 0
        samples = 0
        resets = 0
        randoms = 0
        while i < n_steps:
            i += 1
            print(i)
            for j in range(n_tries_per_step):
                samples += 1
                neighborhood = self.get_sampled_neighborhood(4,2)
                metrics = [self.get_metrics(n) for n in neighborhood]
                rewards = [self.get_reward(m) for m in metrics]
                best_idx = np.argmax(rewards)
                if rewards[best_idx] == float("-inf"):
                    no_valids += 1
                if rewards[best_idx] > last_reward:
                    break
            # if there's no legal states then just reset
            if rewards[best_idx] < last_reward or rewards[best_idx] == float("-inf"):
                last_reward = float("-inf")
                if rewards[best_idx] == float("-inf"):
                    print("No valid moves! Resetting!")
                else:
                    print("No better move! Resetting!")
                # what should I do here? this means there's nowhere to go that's legal
                i -= 1 # not sure if this is right, but get the step back. will guarantee n_steps
                # alternatives, restart and add an empty row, or just skip this step
                resets += 1
                self.reset(initial)
                continue
            if np.random.rand() < eps:
                randoms += 1
                # mask out the legal options
                legal_mask = np.array([1 if r > float("-inf") else 0 for r in rewards], dtype=np.float32)
                # convert to probability
                legal_mask /= np.sum(legal_mask)
                best_idx = np.random.choice(np.arange(len(rewards)), p=legal_mask)
            if eps > eps_min:
                eps *= eps_decay
            if eps < eps_min:
                eps = eps_min
            last_reward = rewards[best_idx]
            # TODO: need to update occupied when changing state
            self.state = neighborhood[best_idx]
            all_designs.append(self.state)
            all_metrics.append(metrics[best_idx])
        print("n_steps: {}, samples: {}, resets: {}, none_valids: {}, randoms: {}".format(n_steps, samples, resets, no_valids, randoms))
        return all_designs, all_metrics


if __name__ == '__main__':
    ga = GreedyAgent(metrics=['population','pvi','compactness','projected_votes','race','income','area'])
    ga.set_task([0,0,0,1,0,0,0])
    print(ga.reset())
    designs, metrics = ga.run(200)
    import csv
    with open('outcomes.csv', 'w+') as outfile:
        writer = csv.writer(outfile)
        for row in metrics:
            if row is None:
                row = [None]*len(ga.metrics)
            writer.writerow(row)
    with open('designs.csv', 'w+') as outfile:
        writer = csv.writer(outfile)
        for row in designs:
            writer.writerow(row)
