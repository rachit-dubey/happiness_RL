#!/usr/bin/env python
'''
Code for running experiments where RL agents interact with an MDP.

Instructions:
    (1) Create an MDP.
    (2) Create agents.
    (3) Set experiment parameters (instances, episodes, steps).
    (4) Call run_agents_on_mdp(agents, mdp) (or the lifelong/markov game equivalents).

    -> Runs all experiments and will open a plot with results when finished.

Author: David Abel (cs.brown.edu/~dabel/)
'''

# Python imports.
from __future__ import print_function
import time
import argparse
import os
import math
import sys
import copy
import numpy as np
from collections import defaultdict
import random
import numpy as np
import scipy.stats
import pickle
import matplotlib.pyplot as plt
import functools
from functools import partial
from joblib import Parallel, delayed #parallel computing

# Non-standard imports.
from simple_rl.planning import ValueIteration
from simple_rl.experiments import Experiment
from simple_rl.utils import chart_utils
from simple_rl.agents import *
from simple_rl.tasks import *
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState

import csv

def run_agents_on_mdp(agents,
                        mdp,
                        instances=5,
                        episodes=100,
                        steps=200,
                        clear_old_results=True,
                        rew_step_count=1,
                        track_disc_reward=False,
                        open_plot=True,
                        verbose=False,
                        reset_at_terminal=False,
                        cumulative_plot=True,
                        dir_for_plot="results",
                        experiment_name_prefix="",
                        track_success=False,
                        success_reward=None, env_num=0, 
                        counterfactual_regret=True,
                        dynamic_aspiration = False,
                        non_stationary = False, 
                        frequency_change = 4):
    '''
    Args:
        agents (list of Agents): See agents/AgentClass.py (and friends).
        mdp (MDP): See mdp/MDPClass.py for the abstract class. Specific MDPs in tasks/*.
        instances (int): Number of times to run each agent (for confidence intervals).
        episodes (int): Number of episodes for each learning instance.
        steps (int): Number of steps per episode.
        clear_old_results (bool): If true, removes all results files in the relevant results dir.
        rew_step_count (int): Number of steps before recording reward.
        track_disc_reward (bool): If true, track (and plot) discounted reward.
        open_plot (bool): If true opens the plot at the end.
        verbose (bool): If true, prints status bars per episode/instance.
        reset_at_terminal (bool): If true sends the agent to the start state after terminal.
        cumulative_plot (bool): If true makes a cumulative plot, otherwise plots avg. reward per timestep.
        dir_for_plot (str): Path
        experiment_name_prefix (str): Adds this to the end of the usual experiment name.
        track_success (bool): If true, tracks whether each run is successful and generates an additional success plot at the end.
        success_reward (int): If set, determines the success criteria.        
        env_num (int): is the environment number on which the agent is being run
        non_stationary(bool): whether MDP is non-stationary
        frequency_change(int): how many times the environment changes in lifetime

    Summary:
        Runs each agent on the given mdp according to the given parameters.
        Stores results in results/<agent_name>.csv and automatically
        generates a plot and opens it.
    '''
    # Experiment (for reproducibility, plotting).
    exp_params = {"instances":instances, "episodes":episodes, "steps":steps}
    experiment = Experiment(agents=agents,
                            mdp=mdp,
                            params=exp_params,
                            is_episodic= episodes > 1,
                            clear_old_results=clear_old_results,
                            track_disc_reward=track_disc_reward,
                            count_r_per_n_timestep=rew_step_count,
                            cumulative_plot=cumulative_plot,
                            dir_for_plot=dir_for_plot,
                            experiment_name_prefix=experiment_name_prefix,
                            track_success=track_success,
                            success_reward=success_reward,
                            env_num = env_num)

    # Record how long each agent spends learning.
    print("Running experiment: \n" + str(experiment))
    time_dict = defaultdict(float)

    # Run each of the agent on the mdp parallely
    f_partial = functools.partial(run_single_agent_on_mdp, mdp=mdp, episodes=episodes, steps=steps, experiment=experiment, verbose=verbose, track_disc_reward=track_disc_reward, reset_at_terminal=reset_at_terminal, counterfactual_regret=counterfactual_regret,non_stationary = non_stationary, 
        dynamic_aspiration= dynamic_aspiration, frequency_change = frequency_change)
    
    Parallel(n_jobs=6)(delayed(f_partial)(agent=agent) for agent in agents)
    print(f_partial)

def run_single_agent_on_mdp(agent, mdp, episodes, steps, experiment=None, verbose=False, 
	track_disc_reward=False, reset_at_terminal=False, resample_at_terminal=False, 
	counterfactual_regret=True,     dynamic_aspiration=False, non_stationary = False, frequency_change = 4, 
	plot_counts = False, plot_value = False, plot_policy=False, plot_steps = 1000, agent2=None, 
    filename = 'default.pkl', folder = 'results/', plot_entropy=False):
    '''
    Summary:
        Main loop of a single MDP experiment.
        counterfactual_regret (bool): If yes, then compute alternative counterfactual reward by a random agent. 
        
        non_stationary is boolean that says whether the MDP is non_stationary
        frequency_change tells how many times the MDP changes in the agent's lifetime
        
    Returns:
        (tuple): (bool:reached terminal, int: num steps taken, list: cumulative discounted reward per episode)
    '''

    if not os.path.exists(folder):
        os.makedirs(folder)

    if reset_at_terminal and resample_at_terminal:
        raise ValueError("(simple_rl) ExperimentError: Can't have reset_at_terminal and resample_at_terminal set to True.")

    value_per_episode = [0] * episodes
    gamma = mdp.get_gamma()
    c = 0 #c is a counter to oscillate the goal location from start to end back and forth -- we need to make this more flexible
    
    printer = 0 #counter to print intrinsic rewards after reaching goal state -- for diagnosis

    ind_counts = 0 #individual count of specific states -- for diagnosis
    #entropy of policy over lifetime learning -- for diagnosis
    entropy_bottom_left = []
    entropy_upper_left = []
    entropy_bottom_right = []
    entropy_upper_right = [] 

    # For each episode.

    f1 = open(folder+'all15.csv', 'w')
    with f1:
        writer = csv.writer(f1)  
    with open(filename, 'wb') as f:
        for episode in range(1, episodes + 1):        
            cumulative_episodic_reward = 0
            cumulative_happiness = 0
            visit_counts = np.zeros((mdp.height, mdp.width)) #this keeps count of visit times of each state.

            # Compute initial state/reward.
            state = mdp.get_init_state()
            visit_counts[state[0]][state[1]] = visit_counts[state[0]][state[1]] + 1 #increase visit count of initial state
            reward = 0
            happiness = 0
            #episode_start_time = time.clock()
            counterfactual_reward = 0
            counterfactual_state = mdp.get_init_state()
            
            index1 = 0 #for changing goal locations, as of now, goal locations fluctate between the 4 corners
            index2 = 0
            positions1 = [(1,1), (13,1)] #to fix, make this non-hard coded
            positions2 = [(1,13), (13,13)]
            
            for step in range(1, steps + 1):

                # step time
                #step_start = time.clock()

                #if non-stationary, and steps matches frequency_change, then change the goal location of the mdp  
                a = list(range(int(steps/frequency_change), int(steps+1), int(steps/frequency_change))) 
                
                if non_stationary:
                    if step in a:
                            if c == 0:
                                c = 1
                                if index1 >= len(positions1): #if gone over, then cycle through list again
                                       index1 = 0
                                mdp.goal_locs = [positions1[index1]] #to fix, make this non-hard coded 
                                index1 = index1+1
                            elif c == 1:                            
                                c = 0
                                if index2 >= len(positions2): #if gone over, then cyce through list again
                                       index2 = 0
                                mdp.goal_locs = [positions2[index2]] #to fix, make this non-hard coded 
                                index2 = index2+1
                
                
                # Compute the agent's policy.
                if counterfactual_regret:
                    action,happiness = agent.act(state, reward, counterfactual_state, counterfactual_reward, cumulative_episodic_reward, step, agent2=agent2, episode_number = episode) #this only works for happy-q-learning agent
                elif dynamic_aspiration: #this also only works for happy-q-learning
                    action,happiness = agent.act(state, reward, counterfactual_state, counterfactual_reward, cumulative_episodic_reward, step, agent2=agent2, episode_number = episode)
                else:
                    action, happiness = agent.act(state, reward, cumulative_episodic_reward, step)#for all other agents, although note that the last term is only return by happy-Q agent
                    
                if counterfactual_regret:
                    counter_action = agent.counterfactual_policy(action) #obtain the counterfactual action by a random agent
                
                #this will serve as counterfactual state and reward in next round
                counterfactual_reward, counterfactual_state  = mdp.simulate_agent_action(counter_action) 
                # Execute in MDP.
                reward, next_state = mdp.execute_agent_action(action)
                
                if printer == 1:
                    print('intrinsic reward after reaching goal is:', happiness, 'step size is ', step
                    , 'value is', agent.get_max_q_value(state))
                    printer = 0

                # Track value.
                value_per_episode[episode - 1] += reward * gamma ** step
                cumulative_episodic_reward += reward            
                cumulative_happiness += happiness
                
                # Record the experience.
                if experiment is not None:
                    reward_to_track = mdp.get_gamma()**(step + 1 + episode*steps) * reward if track_disc_reward else reward
                    reward_to_track = round(reward_to_track, 5)
                    experiment.add_experience(agent, state, action, reward_to_track, happiness, next_state, 
                        time_taken=1)#time.clock() - step_start)

                if mdp.is_goal_state(state): #for diagnosis, let's see rewards after reaching goal
                    printer = 0 #1 for diagnosis    
                
                #for diagnosis
                '''
                if state[0] == 12 and state[1] == 12: #getting values only for a specific state
                    ind_counts = ind_counts+1
                    print(step, ind_counts, agent.get_max_q_value(state)) 
                '''
                  
                #if step % 1 == 0:
                #    list1 = [step, state, reward, 
                #    happiness, agent.get_max_q_value(state), agent.get_max_q_action(state), 
                #    action, next_state]
                #    list2 = [step, reward]
                #    writer.writerow(list2)
                        
                # Update pointer.
                state = next_state
                
                visit_counts[state[0]-1][state[1]-1] = visit_counts[state[0]-1][state[1]-1] + 1 #increase the visit count of that state   

                if plot_counts and step % plot_steps == 0:
                    mdp.visualize_counts(agent, visit_counts, step, folder) #call the visualize count method every n^th step

                if plot_value and step % plot_steps == 0:
                    mdp.visualize_value(agent, step, folder) #call the visualize count method every n^th step   

                if plot_entropy and step % plot_steps == 0:
                    entropy1, entropy2, entropy3, entropy4 = calculate_entropy(agent)
                    entropy_bottom_left.append(entropy1)
                    entropy_upper_left.append(entropy2)
                    entropy_bottom_right.append(entropy3)    
                    entropy_upper_right.append(entropy4)

                if plot_policy and step % 12500 == 0:
                    mdp.visualize_policy(agent, step, folder) #call the visualize count method every n^th step                  

                #for episodic RL    
                if next_state.is_terminal():
                    if reset_at_terminal:
                        # Reset the MDP and tell the agent the episode is over.
                        mdp.reset()
                        agent.end_of_episode()
                        break       
                       
            # Process experiment info at end of episode.
            if experiment is not None:
                experiment.end_of_episode(agent)
                print

            # Reset the MDP, tell the agent the episode is over.
            mdp.reset()        
            print(cumulative_episodic_reward, step, cumulative_happiness)

            #store in pickle
            pickle.dump([episode, step], f)
            
            agent.end_of_episode()
    
    # Process that learning instance's info at end of learning.
    if experiment is not None:
        experiment.end_of_instance(agent)

    if plot_entropy:
        plot_entropies(entropy_bottom_left,entropy_upper_left, 
            entropy_bottom_right,entropy_upper_right, folder, plot_steps)

    return False, steps, value_per_episode, cumulative_episodic_reward, cumulative_happiness, visit_counts

def choose_mdp(mdp_name, env_name="Asteroids-v0"):
    '''
    Args:
        mdp_name (str): one of {gym, grid, chain, taxi, ...}
        gym_env_name (str): gym environment name, like 'CartPole-v0'

    Returns:
        (MDP)
    '''

    # Other imports
    from simple_rl.tasks import ChainMDP, GridWorldMDP, FourRoomMDP, TaxiOOMDP, RandomMDP, PrisonersDilemmaMDP, RockPaperScissorsMDP, GridGameMDP

    # Taxi MDP.
    agent = {"x":1, "y":1, "has_passenger":0}
    passengers = [{"x":4, "y":3, "dest_x":2, "dest_y":2, "in_taxi":0}]
    walls = []
    if mdp_name == "gym":
        # OpenAI Gym MDP.
        try:
            from simple_rl.tasks.gym.GymMDPClass import GymMDP
        except:
            raise ValueError("(simple_rl) Error: OpenAI gym not installed.")
        return GymMDP(env_name, render=True)
    else:
        return {"grid":GridWorldMDP(5, 5, (1, 1), goal_locs=[(5, 3), (4,1)]),
                "four_room":FourRoomMDP(),
                "chain":ChainMDP(5),
                "taxi":TaxiOOMDP(10, 10, slip_prob=0.0, agent=agent, walls=walls, passengers=passengers),
                "random":RandomMDP(num_states=40, num_rand_trans=20),
                "prison":PrisonersDilemmaMDP(),
                "rps":RockPaperScissorsMDP(),
                "grid_game":GridGameMDP(),
                "multi":{0.5:RandomMDP(num_states=40, num_rand_trans=20), 0.5:RandomMDP(num_states=40, num_rand_trans=5)}}[mdp_name]

def parse_args():
    # Add all arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-mdp", type = str, nargs = '?', help = "Select the mdp. Options: {atari, grid, chain, taxi}")
    parser.add_argument("-env", type = str, nargs = '?', help = "Select the Gym environment.")
    args = parser.parse_args()

    # Fix variables based on options.
    task = args.mdp if args.mdp else "grid"
    env_name = args.env if args.env else "CartPole-v0"

    return task, env_name

def calculate_entropy(ql_agent): 
#computes average entropy of the policy at one particular timestep; for ease of understanding compute entropy at various parts of the mdp
    entropy_bottom_left = []
    entropy_upper_left = []
    entropy_bottom_right = []
    entropy_upper_right = []

    for s in list(ql_agent.q_func.keys()):
        p_data, all_values = ql_agent.get_action_distr(s)
        e = scipy.stats.entropy(p_data)
        if np.isnan(e):
            print(s,p_data,all_values)
        if s.x <= 6 and s.y <= 6:        
            entropy_bottom_left.append(e)    
        elif s.x <= 6 and s.y > 6:            
            entropy_upper_left.append(e)
        elif s.x > 6 and s.y <= 6:            
            entropy_bottom_right.append(e)
        elif s.x > 6 and s.y > 6: 
            entropy_upper_right.append(e)        

    return np.mean(entropy_bottom_left), np.mean(entropy_upper_left), np.mean(entropy_bottom_right), np.mean(entropy_upper_right)

def plot_entropies(entropy1,entropy2,entropy3,entropy4,folder,plot_steps=1000):
    #plots entropy of the policy for the various parts of the MDP
    a = 12500/plot_steps

    fig, ax = plt.subplots()
    x = np.arange(1,13, 12.5/a)
    ax.set_xticks(x)
    plt.bar(x, entropy1)
    plt.title('Entropy of policy on bottom left states')
    plt.xlabel('steps (in thousand)')
    plt.ylabel('entropy')
    plt.tight_layout()
    plt.savefig(folder+'entropy_bottom_left.png')

    fig, ax = plt.subplots()
    ax.set_xticks(x)
    plt.bar(x, entropy2)
    plt.title('Entropy of policy on upper left states')
    plt.xlabel('steps (in thousand)')
    plt.ylabel('entropy')
    plt.tight_layout()
    plt.savefig(folder+'entropy_upper_left.png')

    fig, ax = plt.subplots()
    
    ax.set_xticks(x)
    plt.bar(x, entropy3)
    plt.title('Entropy of policy on bottom right states')
    plt.xlabel('steps (in thousand)')
    plt.ylabel('entropy')
    plt.tight_layout()
    plt.savefig(folder+'entropy_bottom_right.png')

    fig, ax = plt.subplots()
    ax.set_xticks(x)
    plt.bar(x, entropy4)
    plt.title('Entropy of policy on upper right states')
    plt.xlabel('steps (in thousand)')
    plt.ylabel('entropy')
    plt.tight_layout()
    plt.savefig(folder+'entropy_upper_right.png')

def plan_single_agent_on_mdp(agent_original, mdp_original, episodes, 
    steps, experiment=None, verbose=False, track_disc_reward=False, reset_at_terminal=False, 
    resample_at_terminal=False, counterfactual_regret=False,  
    dynamic_aspiration=False, non_stationary = False, frequency_change = 4):
    '''
    Summary:
        Run a single MDP experiment on an already pre-trained agent. 
        counterfactual_regret (bool): If yes, then compute alternative counterfactual reward by a random agent. 
        
        non_stationary is boolean that says whether the MDP is non_stationary
        frequency_change tells how many times the MDP changes in the agent's lifetime
        
    Returns:
        (tuple): (bool:reached terminal, int: num steps taken, list: cumulative discounted reward per episode)
    '''
    mdp = copy.deepcopy(mdp_original)
    agent = copy.deepcopy(agent_original)
    
    #to put agent randomly anywhere on the grid and allow teleportation anywhere
    #a = random.sample(mdp.possible_locations,1)
    #mdp.set_init_state(GridWorldState(a[0][0], a[0][1]))
    #mdp.diagnosis = False
    
    #else use this for diagnosis
    mdp.diagnosis = True
    print(mdp.init_state)


    gamma = mdp.get_gamma()
    c = 0 #c is a counter to oscillate the goal location from start to end back and forth -- we need to make this more flexible
    
    policy = agent.policy #obtain policy of the learnt agent
    policy_dict = defaultdict(lambda: defaultdict(str))

    value_iter = ValueIteration(mdp, sample_rate=5, max_iterations=500) #initialize value iteration just to obtain states

    for s in value_iter.get_states():#go thru the states
        policy_dict[s.x][s.y] = policy(s)[0] #store the policy in a dictionary
    # For each episode.
    for episode in range(1, episodes + 1):

        cumulative_episodic_reward = 0
        cumulative_happiness = 0

        # Compute initial state/reward.
        state = mdp.get_init_state()
        reward = 0
        happiness = 0
        #episode_start_time = time.clock()
        
        index1 = 0 #for changing goal locations, as of now, goal locations fluctate between the 4 corners
        index2 = 0
        positions1 = [(1,1), (7,1)]
        positions2 = [(1,7), (7,7)]
        
        #print(policy_dict)

        for step in range(1, steps + 1):
            
            # step time
            #step_start = time.clock()

            #if non-stationary, and steps matches frequency_change, then change the goal location of the mdp  
            a = list(range(int(steps/frequency_change), int(steps+1), int(steps/frequency_change))) 
            
            if non_stationary:
                if step in a:
                        if c == 0:
                            c = 1
                            if index1 >= len(positions1): #if gone over, then cyce through list again
                                   index1 = 0
                            mdp.goal_locs = [positions1[index1]] #to fix, make this non-hard coded 
                            index1 = index1+1
                        elif c == 1:                            
                            c = 0
                            if index2 >= len(positions2): #if gone over, then cyce through list again
                                   index2 = 0
                            mdp.goal_locs = [positions2[index2]] #to fix, make this non-hard coded 
                            index2 = index2+1
            
            if np.random.random() > 0.1:
            # Exploit.
                action = policy_dict[state.x][state.y] #obtain the action
            else:
                # Explore
                action = np.random.choice(agent.actions)                

            reward, next_state = mdp.execute_agent_action(action) #take the action and get resultant reward and next state
            if reward == 1: #reached the goal state
                print('goal reached in steps:', step)
            cumulative_episodic_reward += reward            
            #print(state, action, next_state)
            state = next_state #update pointer
            
        # Reset the MDP, tell the agent the episode is over.
        mdp.reset()        
        print(cumulative_episodic_reward)
        agent.end_of_episode()

    return steps, cumulative_episodic_reward, cumulative_happiness

def main():
    # Command line args.
    task, rom = parse_args()

    # Setup the MDP.
    mdp = choose_mdp(task, rom)
    actions = mdp.get_actions()
    gamma = mdp.get_gamma()

    # Setup agents.
    from simple_rl.agents import RandomAgent, QLearningAgent
    
    random_agent = RandomAgent(actions)
    qlearner_agent = QLearningAgent(actions, gamma=gamma, explore="uniform")
    agents = [qlearner_agent, random_agent]

    # Run Agents.
    if isinstance(mdp, MarkovGameMDP):
        # Markov Game.
        agents = {qlearner_agent.name: qlearner_agent, random_agent.name:random_agent}
        play_markov_game(agents, mdp, instances=100, episodes=1, steps=500)
    else:
        # Regular experiment.
        run_agents_on_mdp(agents, mdp, instances=50, episodes=1, steps=2000)

if __name__ == "__main__":
    main()
