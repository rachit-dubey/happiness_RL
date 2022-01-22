'''
make_custom_mdp.py

Utility for making my custom MDP instance
'''

# Python imports.
import itertools
import random
import math
from collections import defaultdict

# Other imports.
from simple_rl.tasks import GridWorldMDP

def make_custom_mdp(size = 7, make_walls = True, num_goals = 1, num_lavas = 2, num_sinks = 2,
                    gamma=0.99,slip_prob=0.0,sink_prob=0.95,step_cost=0.05,lava_cost=1.0, is_goal_terminal=False,
                    is_teleportation=True):
    '''
    #size can be 7,9,11,13 -- smaller the size, easier the environment -- also serves a way to control sparsity
    make_walls #boolean variable
    num_goals, num_lavas, num_sinks all specifiy how much food, poison, sinkholes are in the world
    
    Returns:
        (MDP)
    '''
    half_width = math.ceil(size / 2.0)
    half_height = math.ceil(size / 2.0)
    
    #make walls first
    
    if make_walls:
        walls =  compute_walls(size, size)   

    #initial locations    
    init_y_locs = [1,2] #for diagnosis, make it (1,1) else make it (1,2)
    init_x_locs = [1,2]    
    init_loc = (random.choice(init_x_locs),random.choice(init_y_locs))
    #print(init_loc)
    #goal locations
    for i in range(num_goals): #for now we assume only 1 goal 
        goal_y_locs = [size-1,size]
        goal_x_locs = [size-1,size]
        goal_loc = (random.choice(goal_x_locs),random.choice(goal_y_locs))
        #goal_loc = (13,13) #for diagnosis
    #print(init_loc, goal_loc)
    #possible locations for all bad things    
    bad_things = []
    
    #put bad things in center and avoid blocking the room entrances
    for i in range(2,half_width):
        for j in range(1,math.ceil(half_width/4)+1):
            bad_things.append((i, half_width+j))
            bad_things.append((i, half_width-j))
    
    for i in range(half_width+1, size):
        for j in range(1,math.ceil(half_width/4)+1):
            bad_things.append((i, half_width+j))
            bad_things.append((i, half_width-j))
            
    #put bad things in first and last row without colliding with the init and goal loc
    for i in range(2,half_width):
        bad_things.append((i, size))
    
    for i in range(half_width+1, size):
        bad_things.append((i, 1))
        
    #put bad things in second and second last row without colliding with the init and goal loc or blocking entrance
    for i in range(2,half_width-1):
        bad_things.append((i, size-1))
    
    for i in range(half_width+2, size):
        bad_things.append((i, 2))    
    
    #sample 'n' bad thing locations where n = n1+n2
    rand_bad_things = random.sample(bad_things, num_lavas+num_sinks)
    
    #sink locations
    #sink_locs = [(5, 1), (2, 7)] #for diagnosis, else use below line
    sink_locs = rand_bad_things[0:num_sinks]
    
    
    #lava locations
    #lava_locs = [(5, 3), (6, 5)]
    lava_locs = rand_bad_things[num_sinks:num_lavas+num_sinks]

    #possible locations for all other things
    possible_locations = []
    for i in range(1,size+1):
        for j in range(1,size+1):
            if (i,j) not in rand_bad_things and (i,j) not in walls: #only append if location is not in bad things/walls
                possible_locations.append((i,j))
    
    mdp = GridWorldMDP(size, size, init_loc = init_loc, walls = walls, goal_locs=[goal_loc], sink_locs=sink_locs, 
        lava_locs = lava_locs, slip_prob=slip_prob, sink_prob=sink_prob, gamma=gamma, step_cost=step_cost, 
        is_goal_terminal=is_goal_terminal, is_teleportation = is_teleportation, 
        possible_locations = possible_locations) 
    
    return mdp

def compute_walls(width, height):
        '''
        Args:
            width (int)
            height (int)

        Returns:
            (list): Contains (x,y) pairs that define wall locations.
        '''
        walls = []

        half_width = math.ceil(width / 2.0)
        half_height = math.ceil(height / 2.0)
        
        #put 1 brick on center bottom and center up        
        walls.append((half_width, 1))
        walls.append((half_width, height))
        
        #put 1 brick on center
        walls.append((half_width, half_height))
        
        #put bricks in center row
        for i in range(1,math.ceil(half_width/4)+1):
            walls.append((half_height,half_width+i))
            walls.append((half_height,half_width-i))

        #put bricks in center column
        for i in range(1,math.ceil(half_width/4)+1):
            walls.append((half_height+i, half_width))
            walls.append((half_height-i, half_width))
    
        return walls

