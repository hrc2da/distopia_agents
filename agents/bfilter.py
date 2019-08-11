import numpy as np
#from pandas import *
a = np.matrix([[5,3,0,0,-4,-1],
              [3,1,0,0,-2,-1],
              [4,4,1,1,-2,-4],
              [0,0,-2,-1,5,4],
              [-1,0,0,0,3,4],
              [-1,0,0,0,3,4],
              [5,5,1,1,-5,-5]])

cov = np.cov(a.T)

b0 = np.array([1,1,0,0,0,0])
b1 = np.array([0,1,0,0,0,0])
b2 = np.array([0,0,0,0,1,1])
b3 = np.array([0,-1,0,0,0,0])

a = np.dot(cov,b0)
b = np.dot(cov,b1)
c = np.dot(cov,b2)

class BayesFilter:
  def __init__(self,states,transition_fn, observation_fn, prior=None):
    '''
    states - list of possible states
    prior - initial belief distribution over states
    transition_fn - function that takes a prior state x, a new state x0, 
                    and an action u, and returns the probability P(x|u,x0)
    observation_fn - function that takes 
    '''
    self.states = states
    self.transition_fn = transition_fn
    self.observation_fn = observation_fn
    if prior is not None:
      assert len(prior) == len(states)
      self.belief = prior
    else:
      # if no prior, initialize with uniform probability
      self.belief = {str(state) : 1/len(states) for state in states}
    
    
  def prediction_step(self, action):
    '''Updates belief distribution according to prior beliefs and action.
    
    Bel'(x) = sum_x'(P(x|u,x')Bel(x'))
    '''
    # Loop over possible states and update the belief that I am in each state
    # using Bel'(x) = sum_x'(P(x|u,x')Bel(x'))
    # that is, for each state x, sum over the probability that I am in that 
    # state now given all nonzero beliefs of prior states and the action

    # question: should I update these in place? I think that will break it...
    # for each state, x in the space
    new_belief = {}
    states = self.states
    belief_0 = self.belief
    for x in states:
      new_belief[str(x)] = 0
      # for each possible prior state, x0 in the space (including itself)
      for x0 in states:
        bel_x0 = belief_0[str(x0)]
        # calculate the probability that I end up in this state, x
        p = self.transition_fn(x0,x,action)
        # weight it by my belief that I was in x0 to begin with
        new_belief[str(x)] += p*bel_x0
    
    self.belief = new_belief
    return self.belief
    
    
  def observation_step(self, observation):
    '''Updates belief distribution according to observation
    
    
    Bel'(x) = normalized P(z|x)Bel(x)
    '''
    # Loop over possible states and update the belief that I am in each state
    # using Bel'(x) = normalized P(z|x)Bel(x), where z is the current obs
    
    new_belief = {}
    states = self.states
    belief = self.belief
    normalizing_factor = 0
    for x in states:
      new_belief[str(x)] = self.observation_fn(observation,x) * belief[str(x)]
      normalizing_factor += new_belief[str(x)]
    
    for b in new_belief:
      new_belief[str(b)] /= normalizing_factor
      
    self.belief = new_belief
    return self.belief   
  
  def get_belief(self):
    return self.belief
    
  def set_belief(self, belief):
    self.belief = belief
    return self.belief
  


def one_bit_low_prob_transition(x0,x,a):
  '''Transitions to only one-bit away problem formulations, with low prob
  
  Note: thie is independent of a
  '''
  diff = x0 - x
  if np.linalg.norm(diff,1) > 1:
    return 

