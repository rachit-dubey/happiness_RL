''' HappyQLearningAgentClass.py: Class for a basic happiness-based QLearningAgent '''

# Python imports.
import random
import numpy
import time
from collections import defaultdict
import math
# Other imports.
from simple_rl.agents.AgentClass import Agent
from simple_rl.experiments import Experiment
from simple_rl.tasks import GridWorldMDP
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState

class HappyQLearningAgent(Agent):
    ''' Implementation for a Q Learning Agent '''

    def __init__(self, actions, name="Q-Agent-", alpha=0.1, gamma=0.99, gamma_rpe = 0.05, epsilon=0.1, explore="uniform", 
        anneal=False, custom_q_init=None, default_q=0, w1 = 0.34, w2 = 0.33,
         w3 = 0.33, w4 = 0.33, average_reward_rate = 0.5, lr_dynamic = True, aspiration_dynamic = False):
        '''
        Args:
            actions (list): Contains strings denoting the actions.
            name (str): Denotes the name of the agent.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Exploration term.
            explore (str): One of {softmax, uniform}. Denotes explore policy.
            custom_q_init (defaultdict{state, defaultdict{action, float}}): a dictionary of dictionaries storing the initial q-values. Can be used for potential shaping (Wiewiora, 2003)
            default_q (float): the default value to initialize every entry in the q-table with [by default, set to 0.0]
        '''
        name_ext = "-" + explore if explore != "uniform" else ""
        name_ext2 = str(w1)+"-"+str(w2)+"-"+str(w3)+"-"+str(w4)+"-"+str(epsilon)+"-"+str(gamma_rpe)+"-"+str(average_reward_rate)
        Agent.__init__(self, name=name + name_ext2+ name_ext, actions=actions, gamma=gamma)

        # Set/initialize parameters and other relevant classwide data
        self.alpha, self.alpha_init = alpha, alpha
        self.epsilon, self.epsilon_init = epsilon, epsilon
        self.step_number = 0
        self.error = 0
        self.anneal = anneal
        self.lr_dynamic = lr_dynamic #is the learning static or dynamic? 
        self.default_q = default_q # 0 # 1 / (1 - self.gamma)
        self.explore = explore
        self.custom_q_init = custom_q_init
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.gamma_rpe = gamma_rpe #discount rate to compute RPE
        self.aspiration_level = average_reward_rate
        self.aspiration_dynamic = aspiration_dynamic #is the aspiration level static or dynamic?
        self.happiness = 0 #initialize happines to be zero in the beginning
        self.cumulative_happiness = 0

        self.cumulative_reward_previous = 0 #what was the cumulative reward previously? 
        self.positive_rewards_accumulated = 0
        # Q Function:
        if self.custom_q_init:
            self.q_func = self.custom_q_init
        else:
            self.q_func = defaultdict(lambda: defaultdict(lambda: self.default_q))        
        
        self.count = 0
        # Key: state
        # Val: dict
            #   Key: action
            #   Val: q-value


    def get_parameters(self):
        '''
        Returns:
            (dict) key=param_name (str) --> val=param_val (object).
        '''
        param_dict = defaultdict(int)

        param_dict["alpha"] = self.alpha
        param_dict["gamma"] = self.gamma
        param_dict["epsilon"] = self.epsilon_init
        param_dict["anneal"] = self.anneal
        param_dict["explore"] = self.explore

        return param_dict

    # --------------------------------
    # ---- CENTRAL ACTION METHODS ----
    # --------------------------------

    def act(self, state, reward, counterfactual_state=1, counterfactual_reward=1,cumulative_reward=1,step=1,learning=True, 
        agent2=None, episode_number = 1):
        '''
        Args:
            state (State)
            reward (float)

        Returns:
            (str)

        Summary:
            The central method called during each time step.
            Retrieves the action according to the current policy
            and performs updates given (s=self.prev_state,
            a=self.prev_action, r=reward, s'=state)
        '''

        if learning:
            self.update(self.prev_state, self.prev_action, reward, state, counterfactual_reward,
                counterfactual_state, cumulative_reward, step, agent2=agent2, episode_number = episode_number)
        if self.explore == "softmax":
            # Softmax exploration
            action = self.soft_max_policy(state)
        else:
            # Uniform exploration
            action = self.epsilon_greedy_q_policy(state)

        self.prev_state = state
        self.prev_action = action
        self.step_number += 1

        # Anneal params.
        if learning and self.anneal:
            self._anneal()

        return action, self.happiness
    
    def counterfactual_policy(self,action):
        '''
        Args:
            (str): action -- last action taken by the policy

        Returns:
            (str): action -- action taken by a counterfactual random policy (apart from the current action taken by the agent)
        '''
        set_of_actions = self.actions
        set_of_actions2 = [n for n in set_of_actions if n != action] #remove the action already taken
        counter_action = numpy.random.choice(set_of_actions2) #obtain a random action (not just taken)
        return counter_action
    
    def epsilon_greedy_q_policy(self, state):
        '''
        Args:
            state (State)

        Returns:
            (str): action.
        '''
        # Policy: Epsilon of the time explore, otherwise, greedyQ.
        if numpy.random.random() > self.epsilon:
            # Exploit.
            action = self.get_max_q_action(state)
        else:
            # Explore
            action = numpy.random.choice(self.actions)

        return action

    def soft_max_policy(self, state):
        '''
        Args:
            state (State): Contains relevant state information.

        Returns:
            (str): action.
        '''
        return numpy.random.choice(self.actions, 1, p=self.get_action_distr(state))[0]

    # ---------------------------------
    # ---- Q VALUES AND PARAMETERS ----
    # ---------------------------------

    def update(self, state, action, reward, next_state, counter_reward, counter_state, 
        cumulative_reward, step, agent2=None, episode_number = 1):
        '''
        Args:
            state (State)
            action (str)
            reward (float)
            next_state (State)
            
            counter_reward 
            counter_state
            
        Summary:
            Updates the internal Q Function according to the Bellman Equation. (Classic Q Learning update)
        '''
        # If this is the first state, just return.
       
        if step > 0 and self.count == 0 and agent2 is not None:
            self.count = 1
            self.end_values(agent2) #assign values to states near the start or end

        if state is None:
            self.prev_state = next_state
            return


        # Update the Q Function.
        max_q_curr_state = self.get_max_q_value(next_state)
        prev_q_val = self.get_q_value(state, action)
        
        #to verify: think about whether we should add the r to r.p.e or not..
        rpe = reward+self.gamma_rpe*max_q_curr_state-prev_q_val
        
        #compute counterfactual regret -- downward comparison
        max_q_hypothetical_state = self.get_max_q_value(counter_state)
        #be sad when other agents better, be happy when you better
        down_regret = reward+self.gamma*max_q_curr_state-counter_reward-self.gamma*max_q_hypothetical_state         
        #to compute regret, take random action (other than action taken from state, observe the r(s,a')

        #first compute static/dynamic aspiration level
        if self.aspiration_dynamic:
            #approach for episodic RL -- increase/decrease the aspiration level as a function of num steps taken 
            #or maybe episode number
            '''
            a = 0.2/episode_number
            b = 0.2 #min aspiration level we want
            self.aspiration_level = min(a,b) #don't let the aspiration drop too much
            #print(episode_number, self.aspiration_level)
            '''
            #########
            #approach 2 = have high aspiration level in the beginning, and then decrease the aspiration level

            if step < 2500:
                self.aspiration_level = 0.001
            else:
                self.aspiration_level = 0.0001   
            
            #approach 1 = if you accumulate positive rewards, then increase your aspiration level 
            # by some percent, else you keep the aspiration level same as before (and we initialize it to be low)
            '''            
            if cumulative_reward > self.cumulative_reward_previous:
                self.positive_rewards_accumulated = self.positive_rewards_accumulated + 1 #increase pos rewards accumulated
                a = (self.positive_rewards_accumulated/self.step_number) + 4*(self.positive_rewards_accumulated/self.step_number) #whatever you have achieved so far, aim for n% more
                
                #b = 0.02 #min aspiration we want
                
                #temp = max(a,b) #don't let the aspiration drop too

                c = 0.2 #max aspiration level we want

                self.aspiration_level = min(a,c) #don't let the aspiration level exceed 0.06
                print(self.aspiration_level)
            else: #agent isn't making progress, keep aspiration level the previous one
                self.aspiration_level = self.aspiration_level
            
            self.cumulative_reward_previous = cumulative_reward #update cumulative reward
            '''
            #in essence the above code updates aspiration level whenever the agent gets 1 positive reward
            
            '''approach 2 = as soon as you accumulate positive rewards, reduce the aspiration level 
            (as you are now away from goal), rest of the time, keep aspiration level equal to positive rewards/step so far
            '''
            '''
            if cumulative_reward > self.cumulative_reward_previous:
                self.positive_rewards_accumulated = self.positive_rewards_accumulated + 1 #increase pos rewards accumulated
                self.aspiration_level = 0 #as soon as you reach the positive reward, you will be teleported, so reduce that aspiration
            else: #no goal reached, keep aspiration level proportional to what you received so far
                self.aspiration_level = 0.01 #(self.positive_rewards_accumulated/self.step_number) + 0.00001
            
            self.cumulative_reward_previous = cumulative_reward #update cumulative reward
            #print(self.aspiration_level)
            '''
        else:
            if cumulative_reward > self.cumulative_reward_previous:
                self.positive_rewards_accumulated = self.positive_rewards_accumulated + 1 #increase pos rewards accumulated

            self.aspiration_level = self.aspiration_level

        up_regret = reward - self.aspiration_level #we could also try G_max(T) - r+V(s), although the r will be very high

        self.happiness = self.w1*reward+self.w2*rpe+self.w3*down_regret+self.w4*up_regret
        self.cumulative_reward_previous = cumulative_reward
        self.cumulative_happiness = self.cumulative_happiness+self.happiness

        #if self.step_number % 100 == 0:
            #print(reward,self.happiness, rpe, cumulative_reward, self.cumulative_happiness, self.step_number)
        happiness_error = self.happiness + self.gamma*max_q_curr_state - prev_q_val #compute rpe based on overall happiness
        
        #if step > 2500 and step < 5000 and next_state.x == 13 and next_state.y == 13:
        #    print(step, happiness_error)    

        if self.lr_dynamic:  #is learning rate set to be dynamic?      
            #compute learning rate i.e. alpha based on happiness R.P.E error
            self.error = abs(happiness_error) + 0.9*self.error #we accumulate the errors to calculate the learning rate        
            self.alpha = self.compute_alpha(math.tan(self.error)) #pass the error thru tan to make it 0-1
        else:
            self.alpha = self.alpha #otherwise alpha is value set originally

        if agent2 is not None and state[0] < 4 and state[1] < 4 and step > 12500: #don't update the start states afte 5k steps 
        #(just for diagnosis)
            self.q_func[state][action] = prev_q_val
        elif agent2 is not None and 10 < state[0] < 14 and 10 < state[1] < 14 and step > 0: 
            self.q_func[state][action] = prev_q_val    
        else:    
            self.q_func[state][action] = prev_q_val + self.alpha * (happiness_error)    

        #if happiness_error > 0:
        #    print(step, state, prev_q_val, self.q_func[state][action], happiness_error)

    def start_values(self, agent2):
        for i in range(8):
            for j in range(8):
                state = GridWorldState(i+1,j+1)   
                self.q_func[state]['down'] =  agent2.q_func[state]['down']
                self.q_func[state]['stay'] =  agent2.q_func[state]['stay']
                self.q_func[state]['up'] =  agent2.q_func[state]['up']
                self.q_func[state]['left'] =  agent2.q_func[state]['left']
                self.q_func[state]['right'] =  agent2.q_func[state]['right']   

    def end_values(self, agent2):
        for i in range(4):
            for j in range(4):
                state = GridWorldState(13-i,13-j)
                self.q_func[state]['down'] =  agent2.q_func[state]['down']
                self.q_func[state]['stay'] =  agent2.q_func[state]['stay']
                self.q_func[state]['up'] =  agent2.q_func[state]['up']
                self.q_func[state]['left'] =  agent2.q_func[state]['left']
                self.q_func[state]['right'] =  agent2.q_func[state]['right']  


    def _anneal(self):
        # Taken from "Note on learning rate schedules for stochastic optimization, by Darken and Moody (Yale)":
        self.alpha = self.alpha_init / (1.0 +  (self.step_number / 1000.0)*(self.episode_number + 1) / 2000.0 )
        self.epsilon = self.epsilon_init / (1.0 + (self.step_number / 1000.0)*(self.episode_number + 1) / 2000.0 )

    def _compute_max_qval_action_pair(self, state):
        '''
        Args:
            state (State)

        Returns:
            (tuple) --> (float, str): where the float is the Qval, str is the action.
        '''
        # Grab random initial action in case all equal
        best_action = random.choice(self.actions)
        max_q_val = float("-inf")
        shuffled_action_list = self.actions[:]
        random.shuffle(shuffled_action_list)

        # Find best action (action w/ current max predicted Q value)
        for action in shuffled_action_list:
            q_s_a = self.get_q_value(state, action)
            if q_s_a > max_q_val:
                max_q_val = q_s_a
                best_action = action

        return max_q_val, best_action

    def get_max_q_action(self, state):
        '''
        Args:
            state (State)

        Returns:
            (str): denoting the action with the max q value in the given @state.
        '''
        return self._compute_max_qval_action_pair(state)[1]

    def get_max_q_value(self, state):
        '''
        Args:
            state (State)

        Returns:
            (float): denoting the max q value in the given @state.
        '''
        return self._compute_max_qval_action_pair(state)[0]

    def get_value(self, state):
        '''
        Args:
            state (State)

        Returns:
            (float)
        '''
        return self.get_max_q_value(state)

    def get_q_value(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns:
            (float): denoting the q value of the (@state, @action) pair.
        '''
        return self.q_func[state][action]

    def get_action_distr(self, state, temp=0.002):

        '''
        Args:
            state (State)
            temp (float): Softmax temperature parameter (lower the temp, more greedy the policy is).

        Returns:
            (list of floats): The i-th float corresponds to the probability
            mass associated with the i-th action (indexing into self.actions)
        '''
        all_q_vals = []
        for i, action in enumerate(self.actions):
            all_q_vals.append(self.get_q_value(state, action))

        # Softmax distribution.
        #print(all_q_vals)
        total = sum([numpy.exp(qv/temp) for qv in all_q_vals])
        softmax = [numpy.exp(qv/temp) / total for qv in all_q_vals]
        return softmax, all_q_vals

    def reset(self):
        self.step_number = 0
        self.episode_number = 0
        if self.custom_q_init:
            self.q_func = self.custom_q_init
        else:
            self.q_func = defaultdict(lambda : defaultdict(lambda: self.default_q))
        Agent.reset(self)

    def end_of_episode(self):
        '''
        Summary:
            Resets the agents prior pointers.
        '''
        if self.anneal:
            self._anneal()
        Agent.end_of_episode(self)

    def print_v_func(self):
        '''
        Summary:
            Prints the V function.
        '''
        for state in self.q_func.keys():
            print(state, self.get_value(state))
            
    def compute_alpha(self,rpe):
        "a simple implementation to calculate learning rate based on rpe"
        
        if 0 <= abs(rpe) <= 0.1:
            alpha = 0.1
        elif abs(rpe) > 0.1:
            alpha = 0.9
            
        return alpha
       
    
    def print_q_func(self):
        '''
        Summary:
            Prints the Q function.
        '''
        if len(self.q_func) == 0:
            print("Q Func empty!")
        else:
            for state, actiond in self.q_func.items():
                #if state[0] < 6 and state[1] < 6: #this line only for getting values near start state.
                    print(state)
                    for action, q_val in actiond.items():
                        print("    ", action, q_val)
