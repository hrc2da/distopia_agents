# Resources

The following resources should be in this directory.

Please note that due to file size, some of them may not be version-controlled in this repo.


#### stripped_normalization.pkl
_metric means and stds used by the agent to standardize outcomes in reward calculations_ 
* <means = [%f, %f, %f, %f, %f], stds = [%f, %f, %f, %f, %f]> 
* This is a tuple of the 5-dimensional (over metric feature) means and standard deviations of the outcome metrics observed while running the greedy search agent on the 10 single-objective optimization tasks (the one-hots). 
* This data is calculated using the file `five_norms_raw.pkl` in the script `parsing.py`.
* Run `means, stds = pkl.load(<pointer to this file>)` to get a single 5x1 means vector and a single 5x1 stds vector.

#### five_norms_raw.pkl
_explored designs and outcomes from running single-objective agents_
* <outcomes = {'[1. 0. 0. 0. 0.]': n_samples x 5 array...},designs = {'[1. 0. 0. 0. 0.]': fiducial dict}>
* This file consists of a list of a tuple of dictionaries containing raw (non-standardized) metric outcomes keyed on task,
 and designs (fiducial locations) associated with those outcomes, keyed on task, respectively. This data was collected to
 "train" a belief model over single-objective tasks, as well as to provide rough standardization parameters to run
 agents that explore multi-objective tasks (which first standardize the data then weight by the objective task vector in
 order to calculate reward). 
* This data is generated from `run_job.py`, which logs the current objective vector, the design, and the outcomes for
each step that each agent takes. The script generates a set of files named after the task that is being run. These files are combined and parsed using `glob2pkl.py`.
* Run `data, designs = pkl.load(<file pointer>)` to load. There are up to 18000 steps for each of these (note that not all the jobs finished).
Sizes for each of the tasks are as follows: 
    * `[0.0, 0.0, -1.0, 0.0, 0.0]: (18000, 5)`
    * `[0.0, 0.0, 0.0, 1.0, 0.0]: (18000, 5)`
    * `[0.0, 0.0, 1.0, 0.0, 0.0]: (15429, 5)`
    * `[-1.0, 0.0, 0.0, 0.0, 0.0]: (18000, 5)`
    * `[0.0, 0.0, 0.0, 0.0, -1.0]: (18000, 5)`
    * `[0.0, 0.0, 0.0, 0.0, 1.0]: (16839, 5)`
    * `[0.0, -1.0, 0.0, 0.0, 0.0]: (18000, 5)`
    * `[1.0, 0.0, 0.0, 0.0, 0.0]: (18000, 5)`
    * `[0.0, 1.0, 0.0, 0.0, 0.0]: (18000, 5)`
    * `[0.0, 0.0, 0.0, -1.0, 0.0]: (18000, 5)`

#### 243_raw_data.pkl
