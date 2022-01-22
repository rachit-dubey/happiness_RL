#!/usr/bin/env python

# Python imports.
from __future__ import print_function
import argparse
import srl_example_setup
import pandas as pd
from collections import Counter

# Other imports.
from simple_rl.agents.HappyQLearningAgentClass import HappyQLearningAgent
from simple_rl.run_happy_experiments import run_single_agent_on_mdp, plan_single_agent_on_mdp, run_agents_on_mdp
from simple_rl.tasks import GridWorldMDP
from simple_rl.tasks.grid_world.GridWorldMDPClass import make_grid_world_from_file
from simple_rl.planning import ValueIteration
from simple_rl.utils.make_custom_mdp import make_custom_mdp
from collections import defaultdict
import random
import numpy as np

import argparse
import sys

def parse_args():
    # Add all arguments
    parser = argparse.ArgumentParser()
    #parser.add_argument("-v", type=str, default="learning", nargs='?', help="Choose the visualization type (one of {value, policy, agent, learning or interactive}).")
    #args = parser.parse_args()
    parser.add_argument('-c', help = "index number")
    parser.add_argument('-agent', default=0, help="Choose agent, All(0) or Compare(0).")
    parser.add_argument('-folder', default='exp1/', help="Choose agent, All(0) or Compare(0).")
    args = parser.parse_args(sys.argv[1:])
    
    return args.c, args.agent, args.folder

def main():
    # Setup MDP, Agents.

        lifetime = 12500
        num_episodes = 1
        num_lavas = 0
        num_sinks = 0    
        lr = 0.5 #learning rate
        size = 13
        e = 0.1 #epsilon
        teleport = True
        lr_dynamic = False
        sp = 0.1 #slip_prob
        
        plot_steps = 500 #num of steps to plot the value, policy, and counts for
        plot_value = True
        plot_counts = False
        plot_policy = False
        
        _,agent,_ = parse_args() #0 for all, 1 for compare
        _,_,f = parse_args()
        folder = 'results/' + f

        # Choose viz type.
        viz = "value"

        if int(agent) == 0:
            print('all', num_lavas, num_sinks)
            w1 = 0.9 
            w2 = 0.6
            w3 = 0.1 
            r = 0.001 #average reward rate
        elif int(agent) == 1:
            print('compare', num_lavas, num_sinks)
            w1 = 0.0                      
            w2 = 0.6
            w3 = 0.3 
            r = 0.01

        default_q = 0.0 #initial q-values

        aspiration = False #is dynamic aspiration True or False?

        non_stationary = False
        frequency_change = 2

        reset_at_terminal = False #is it episodic or continous task?
        is_goal_terminal = False #set goal_terminal to be True if episodic, else False

        mdp = make_custom_mdp(size = size, make_walls = True, num_goals = 1, 
        	num_lavas = num_lavas, num_sinks = num_sinks,
            gamma=0.99,slip_prob=sp,sink_prob=0.95,step_cost=0.00, lava_cost=1.0, 
            is_goal_terminal=is_goal_terminal, is_teleportation=teleport)    
         
        ql_agent = HappyQLearningAgent(actions=mdp.get_actions(), w1 = w1, w2 =w2, w3 = 0.0, w4 = w3, 
         average_reward_rate =r, epsilon=e, gamma_rpe= 0.00, alpha=lr, lr_dynamic=lr_dynamic, 
         aspiration_dynamic=aspiration, default_q=default_q)

        if viz == "value":
            #shows value of states during learning is performed
            run_single_agent_on_mdp(ql_agent, mdp, episodes=num_episodes, steps=lifetime, 
                non_stationary = non_stationary,
                plot_value=plot_value, plot_counts = plot_counts, plot_policy=plot_policy, 
                plot_steps = plot_steps, reset_at_terminal=reset_at_terminal, folder=folder,
                frequency_change=frequency_change)

        elif viz == "episodic_value":
            #shows value of states during learning is performed
            c,_,_ = parse_args()
            filename1 = 'episodic_returns/optimistic_agent_0.5_'+str(c)+'.pkl'
            run_single_agent_on_mdp(ql_agent, mdp, episodes=num_episodes, steps=lifetime, 
            	non_stationary = non_stationary, plot_value=False, plot_counts = False, plot_policy=False, plot_steps = plot_steps, 
                reset_at_terminal=reset_at_terminal, filename = filename1, frequency_change=frequency_change)  

        elif viz == "average":
            # visualize average count and value of states visited by one agent            
            _, _, _, _, _, accumulated_visit_counts = run_single_agent_on_mdp(ql_agent, 
            	mdp, episodes=1, steps=lifetime)
            
            accumulated_values = np.zeros((size,size))
            for s in list(ql_agent.q_func.keys()):
                accumulated_values[s.x-1][s.y-1] = ql_agent.get_value(s)
            
            for i in range(19):                
                #re-intialize q-agent and mdp again
                mdp = make_custom_mdp(size = size, make_walls = True, num_goals = 1, 
                	num_lavas = num_lavas, num_sinks = num_sinks,gamma=0.99,slip_prob=sp,
                	sink_prob=0.95, step_cost=0.00, lava_cost=1.0, 
                	is_goal_terminal=is_goal_terminal, is_teleportation=teleport)    
         
                ql_agent = HappyQLearningAgent(actions=mdp.get_actions(), 
                	w1 = w1, w2 =w2, w3 = 0.0, w4 = w3, average_reward_rate =r, 
                	epsilon=e, gamma_rpe= 0.00, alpha=lr, lr_dynamic=lr_dynamic, 
                	aspiration_dynamic=aspiration, default_q=default_q)
                
                _, _, _, _, _, visit_counts = run_single_agent_on_mdp(ql_agent, mdp, episodes=1, steps=lifetime)            
                accumulated_visit_counts = np.dstack((accumulated_visit_counts, visit_counts))
                
                values = np.zeros((size,size))                
                for s in list(ql_agent.q_func.keys()):
                    values[s.x-1][s.y-1] = ql_agent.get_value(s)
                accumulated_values = np.dstack((accumulated_values, values))
            
            average_values = np.mean(accumulated_values, axis=2)
            average_visit_counts = np.mean(accumulated_visit_counts, axis=2)
            mdp.visualize_counts(ql_agent, average_visit_counts, lifetime, folder)
            #mdp.visualize_counts(ql_agent, average_values, lifetime, folder)   

        elif viz == "quick_evaluation":
        # shows rewards obtained over n trials.
            a1 = []
            _,_,_,cumulative_episodic_reward,_,_ = run_single_agent_on_mdp(ql_agent, mdp, episodes=1, 
                steps=lifetime, non_stationary=non_stationary, frequency_change=frequency_change) 
            a1.append(cumulative_episodic_reward)

            for i in range(99):
                #re-intialize q-agent and mdp again
                mdp = make_custom_mdp(size = size, make_walls = True, num_goals = 1, 
                	num_lavas = num_lavas, num_sinks = num_sinks,gamma=0.99,slip_prob=sp,sink_prob=0.95,
                	step_cost=0.00, lava_cost=1.0, is_goal_terminal=is_goal_terminal, 
                	is_teleportation=teleport)    
         
                ql_agent = HappyQLearningAgent(actions=mdp.get_actions(), 
                	w1 = w1, w2 =w2, w3 = 0.0, w4 = w3, average_reward_rate =r, 
                	epsilon=e, gamma_rpe= 0.00, alpha=lr, lr_dynamic=lr_dynamic, 
                	aspiration_dynamic=aspiration, default_q=default_q)

                _,_,_,cumulative_episodic_reward,_,_ = run_single_agent_on_mdp(ql_agent, mdp, episodes=1, 
                    steps=lifetime, non_stationary=non_stationary, frequency_change=frequency_change)      
                a1.append(cumulative_episodic_reward)
            print(np.mean(a1), np.std(a1))

        elif viz == "learning":
            # Show agent's interaction with the environment; at the end, the learnt policy is also shown
            mdp.visualize_learning(ql_agent, delay=0.0001, num_ep=num_episodes, num_steps=lifetime,
                                   non_stationary = non_stationary, frequency_change = frequency_change)

        elif viz == "plan":
            # This basically runs the agent for n iterations and then prints the reward obtained by a greey 
            # agent if it follows that learnt policy for n steps
            run_single_agent_on_mdp(ql_agent, mdp, episodes=1, steps=lifetime, 
                non_stationary = non_stationary, frequency_change = frequency_change)
            
            plan_single_agent_on_mdp(ql_agent, mdp, episodes=1, steps=1000, 
                non_stationary = non_stationary, frequency_change = frequency_change)


if __name__ == "__main__":
    main()
