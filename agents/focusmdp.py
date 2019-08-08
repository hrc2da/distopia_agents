import roslibpy
#import rospy
import numpy as np
import time
import json 


class FocusPolicy:
    def act(self,state,q_table):
        raise NotImplementedError()

class WeightedRandomPolicy(FocusPolicy):
    '''Usually does nothing but occasionally acts randomly
    '''
    def __init__(self,eps=0.2):
        # eps is the probability of acting randomly
        # otherwise do nothing
        self.eps = eps
    def act(self,state,q_table):
        if np.random.rand() < self.eps:
            n_actions = q_table.shape[-1]
            print("randomizing")
            rand_act =  np.random.randint(n_actions)
            assert rand_act < n_actions
            return rand_act
        else:
            print("maintain")
            return state[0] # maintain the current state
class FocusMDP:
    '''
        Steps are triggered by user actions
        Listen to changes on the table (/evaluated_designs and /tuio_control)
        on each change, user the user's action to calculate the reward, and then update
        the q-table.
        State space:
            cur_focus: len(actions)
            normalized and binned metrics: len(actions) * 5
    '''

    metrics =[
        'age','education','income','occupation','population','projected_votes','race'
    ]
    foci = {m : i for i,m in enumerate(metrics)}

    def __init__(self, policy, n_actions=7, n_bins=5, alpha=0.5, gamma=0.2):

        self.policy = policy
        self.alpha = alpha
        self.gamma = gamma 
        self.n_actions = n_actions
        n_foci = n_actions
        self.actions = np.arange(n_actions)
        self.user_move_count = 0
        self.state = np.zeros(1 + n_foci, dtype=int)
        self.q_dim = [n_foci] + [n_bins for i in range(n_actions)] + [n_actions]
        self.q_table = np.zeros(self.q_dim)
        self.setup_ros()
        self.step(0)




    def setup_ros(self,host='localhost',port=9090):
        print("Connecting to ROS")
        self.ros = ros = roslibpy.Ros(host='localhost', port=9090)
        ros.run()
        self.start_ros_topics()


    def start_ros_topics(self):
        print("Starting ROS Topics2")
        # subscribers
        self.subscribers = []
        # this is a dict containing the district assignments, histograms, and metrics
        # this is published every time that a fiducial moves
        sub_designs_topic = roslibpy.Topic(
            self.ros, '/evaluated_designs', 'std_msgs/String')
        sub_designs_topic.subscribe(
            self.handle_evaluated_designs)

        self.subscribers.append(sub_designs_topic)

        # # this contains block locations and is fired when fiducials move
        # sub_blocks_topic = roslibpy.Topic(
        #     self.ros, '/blocks', 'std_msgs/String')
        # sub_blocks_topic.subscribe(
        #     self.handle_blocks)

        # self.subscribers.append(sub_blocks_topic)

        # this contains messages about the focus. only triggered when the focus block moves
        sub_tuio_topic = roslibpy.Topic(
            self.ros, '/tuio_control', 'std_msgs/String')
        sub_tuio_topic.subscribe(
            self.handle_tuio_control)

        self.subscribers.append(sub_tuio_topic)
        #logging.info('Started ros-bridge publishers')

        # publisher to move the focus 
        self.pub_focus_action_topic = roslibpy.Topic(
            self.ros, '/focus_action', 'std_msgs/Int8'
        )
        self.pub_focus_action_topic.advertise()

        print("running")



    def handle_evaluated_designs(self,message):
        '''parses changes to the design.
        Update current state based on 
        state is [focus_id, metric_1_binned, metric_2_binned,...]
        call agent_step(state,reward)
        '''
        packet = json.loads(message['data'])
        reward = 1 # because the human didn't move focus
        # don't update the focus_id, which is handled by tuio_control
        state = np.zeros(len(self.state), dtype=int)
        state[0] = self.state[0]
        # TODO: calculate state[1:]
        self.agent_step(state,reward)
        return

    def handle_blocks(self,message):
        return

    def handle_tuio_control(self,message):
        packet = json.loads(message['data'])
        if packet['cmd'] != 'focus_state':
            return
        else:
            focus = int(self.foci[packet['param']])
            if focus == self.state[0]:
                # if the focus has not changed, then ignore
                return
            elif focus == self.last_action:
                # TODO: this might not get hit if we update state in step (we should)
                # this works if we maintain a strict one move turn
                # concern is that if the focus block moves away and comes back
                # my action will not have changed
                # but this will not occur if I act on every human move
                # this is my move
                state = self.state[:]
                state[0] = focus
                self.update_state(state)
            else:
                reward = -1
                state = self.state[:]
                state[0] = focus
                self.agent_step(state,reward)
        return


    def publish_action(self, action):
        self.pub_focus_action_topic.publish(roslibpy.Message({'data':int(action)}))

    def update_q(self,state,action,reward,result):
        '''
            perform bellman update on q-table
            state = prior state
            action = action taken
            reward = reward from action
            result = resulting state
        '''
        lookup_idx = np.hstack([state[:], [action]])
        # Q(S,A) = Q(S,A) + a(R + gQ(S',A') - Q(S,A)); a = learning rate, g = discount
        q0 =  self.q_table[lookup_idx]
        next_q = np.max(self.q_table[result]) #greedy q-update
        self.q_table[lookup_idx] += self.alpha * (reward + self.gamma * next_q - q0)
        return


    def update_state(self, state):
        self.state = state
    def agent_step(self, state, reward):
        '''Updates the state, retrains, selects an action, and steps
        '''
        prior_state = self.state
        prior_action = self.last_action
        self.update_state(state)
        self.update_q(prior_state,prior_action,reward,state)
        action = self.policy.act(self.state,self.q_table)
        self.step(action)

    def step(self,action):
        self.last_action = action
        #random_action = np.random.randint(self.n_actions)
        self.publish_action(action)
    def run(self):
        try:
            while(True):
                time.sleep(1)
        except KeyboardInterrupt:
            self.pub_focus_action_topic.unadvertise()
            for sub in self.subscribers:
                sub.unsubscribe()
        finally:
            print("Exiting!")


if __name__=='__main__':
    wrp = WeightedRandomPolicy()
    fmdp = FocusMDP(wrp)
    fmdp.run()
    # try:
    #     while(True):
    #         time.sleep(10)
    # except KeyboardInterrupt:
    #     print("Finished") 