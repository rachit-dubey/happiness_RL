# Python imports.
from __future__ import print_function
from collections import defaultdict
try:
    import pygame
except ImportError:
    print("Warning: pygame not installed (needed for visuals).")
import random
import sys

# Other imports.
from simple_rl.planning import ValueIteration
from simple_rl.utils import mdp_visualizer as mdpv


def _draw_state(screen,
                grid_mdp,
                state,
                policy=None,
                action_char_dict={},
                show_value=True,
                agent=None,
                draw_statics=False,
                agent_shape=None):
    '''
    Args:
        screen (pygame.Surface)
        grid_mdp (MDP)
        state (State)
        show_value (bool)
        agent (Agent): Used to show value, by default uses VI.
        draw_statics (bool)
        agent_shape (pygame.rect)

    Returns:
        (pygame.Shape)
    '''
    # Make value dict.
    action_char_dict = {
            "up":"^",       #u"\u2191",
            "down":"v",     #u"\u2193",
            "left":"<",     #u"\u2190",
            "right":">",    #u"\u2192"
            "stay":"+",    #u"\u219."
        }

    #make value dictionary        
    val_text_dict = defaultdict(lambda : defaultdict(float))
    action_text_dict = defaultdict(lambda : defaultdict(str))
    if show_value:
        if agent is not None:
            # Use agent value estimates.
            for s in list(agent.q_func.keys()):
                val_text_dict[s.x][s.y] = agent.get_value(s)
                action_text_dict[s.x][s.y] = agent.get_max_q_action(s)
        else:
            # Use Value Iteration to compute value.
            vi = ValueIteration(grid_mdp, sample_rate=10)
            vi.run_vi()
            for s in vi.get_states():
                val_text_dict[s.x][s.y] = vi.get_value(s)

    # Make policy dict.
    policy_dict = defaultdict(lambda : defaultdict(str))
    
    if policy:
        vi = ValueIteration(grid_mdp)
        for s in vi.get_states(): #keys:
            policy_dict[s.x][s.y] = policy(s)

    # Prep some dimensions to make drawing easier.
    scr_width, scr_height = screen.get_width(), screen.get_height()
    width_buffer = scr_width / 10.0
    height_buffer = 30 + (scr_height / 10.0) # Add 30 for title.
    cell_width = (scr_width - width_buffer * 2) / grid_mdp.width 
    cell_height = (scr_height - height_buffer * 2) / grid_mdp.height
    goal_locs = grid_mdp.get_goal_locs()
    lava_locs = grid_mdp.get_lava_locs()
    sink_locs = grid_mdp.get_sink_locs()
    font_size = int(min(cell_width, cell_height) / 6.0)
    reg_font = pygame.font.SysFont("CMU Serif", font_size+2 + 4)
    cc_font = pygame.font.SysFont("Courier", font_size*2 + 10, bold = True)

    # Draw the static entities.
    if draw_statics:
        # For each row:
        for i in range(grid_mdp.width):
            # For each column:
            for j in range(grid_mdp.height):

                top_left_point = width_buffer + cell_width*i, height_buffer + cell_height*j
                r = pygame.draw.rect(screen, (46, 49, 49), top_left_point + (cell_width, cell_height), 3)

                if policy and not grid_mdp.is_wall(i+1, grid_mdp.height - j):
                    a = policy_dict[i+1][grid_mdp.height - j]                    
                    if a[0] not in action_char_dict:                        
                        text_a = a[0]
                    else:
                        text_a = action_char_dict[a[0]]

                    text_center_point = int(top_left_point[0] + cell_width/2.0 - 10), int(top_left_point[1] + cell_height/3.0)
                    text_rendered_a = cc_font.render(text_a, True, (46, 49, 49))
                    screen.blit(text_rendered_a, text_center_point)

                if show_value and not grid_mdp.is_wall(i+1, grid_mdp.height - j):
                    # Draw the value.
                    val = val_text_dict[i+1][grid_mdp.height - j]
                    act = action_text_dict[i+1][grid_mdp.height - j]

                    if act not in action_char_dict:                        
                        text_a = act
                    else:
                        text_a = action_char_dict[act]

                    #color = mdpv.val_to_color(val)
                    #pygame.draw.rect(screen, color, top_left_point + (cell_width, cell_height), 0)

                if grid_mdp.is_wall(i+1, grid_mdp.height - j):
                    # Draw the walls.
                    top_left_point = width_buffer + cell_width*i + 5, height_buffer + cell_height*j + 5
                    r = pygame.draw.rect(screen, (94, 99, 99), top_left_point + (cell_width-10, cell_height-10), 0)

                if (i+1,grid_mdp.height - j) in goal_locs:
                    # Draw goal.
                    circle_center = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] + cell_height/2.0)
                    circler_color = (154, 195, 157)
                    #pygame.draw.circle(screen, circler_color, circle_center, int(min(cell_width, cell_height) / 3.0))

                if (i+1,grid_mdp.height - j) in lava_locs:
                    # Draw lava.
                    circle_center = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] + cell_height/2.0)
                    circler_color = (224, 145, 157)
                    pygame.draw.circle(screen, circler_color, circle_center, int(min(cell_width, cell_height) / 4.0))
                    
                if (i+1,grid_mdp.height - j) in sink_locs:
                    # Draw sink.
                    circle_center = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] + cell_height/2.0)
                    circler_color = (204, 165, 107)
                    pygame.draw.circle(screen, circler_color, circle_center, int(min(cell_width, cell_height) / 4.0))

                if show_value and not grid_mdp.is_wall(i+1, grid_mdp.height - j):
                    # Write value text to each state.
                    value_text = reg_font.render(str(round(val, 4)), True, (46, 49, 49))
                    text_rendered_a = cc_font.render(text_a, True, (46, 49, 49))
                    text_center_point = int(top_left_point[0] + cell_width/2.0 - 10), int(top_left_point[1] + cell_height/2.0)
                    screen.blit(text_rendered_a, text_center_point)
                    text_center_point = int(top_left_point[0] + cell_width/2.0 - 10), int(top_left_point[1] + cell_height/4.0)
                    screen.blit(value_text, text_center_point)

                # Current state.
                if (i+1,grid_mdp.height - j) == (state.x, state.y) and agent_shape is None:                    
                    tri_center = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] + cell_height/2.0)
                    #agent_shape = _draw_agent(tri_center, screen, base_size=min(cell_width, cell_height)/2.5 - 8)

    if agent_shape is not None:
        # Clear the old shape.
        pygame.draw.rect(screen, (255,255,255), agent_shape)
        top_left_point = width_buffer + cell_width*(state.x - 1), height_buffer + cell_height*(grid_mdp.height - state.y)
        tri_center = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] + cell_height/2.0)

        # Draw new.
        #agent_shape = _draw_agent(tri_center, screen, base_size=min(cell_width, cell_height)/2.5 - 8)

    pygame.display.flip()

    return agent_shape


def _draw_agent(center_point, screen, base_size=20):
    '''
    Args:
        center_point (tuple): (x,y)
        screen (pygame.Surface)

    Returns:
        (pygame.rect)
    '''
    tri_bot_left = center_point[0] - base_size, center_point[1] + base_size
    tri_bot_right = center_point[0] + base_size, center_point[1] + base_size
    tri_top = center_point[0], center_point[1] - base_size
    tri = [tri_bot_left, tri_top, tri_bot_right]
    tri_color = (98, 140, 190)
    return pygame.draw.polygon(screen, tri_color, tri)

def _draw_state_2(screen,
                grid_mdp,
                state,
                visit_counts,
                policy=None,
                action_char_dict={},
                show_value=True,
                agent=None,
                draw_statics=False,
                agent_shape=None):
    '''
    Args:
        screen (pygame.Surface)
        grid_mdp (MDP)
        state (State)
        show_value (bool)
        agent (Agent): Used to show value, by default uses VI.
        draw_statics (bool)
        agent_shape (pygame.rect)

    Returns:
        (pygame.Shape)
    '''
    # Make value dict.
    action_char_dict = {
            "up":"^",       #u"\u2191",
            "down":"v",     #u"\u2193",
            "left":"<",     #u"\u2190",
            "right":">",    #u"\u2192"
            "stay":"+",    #u"\u219."
        }
    val_text_dict = defaultdict(lambda : defaultdict(int))
    if show_value:
        #print(agent.q_func.keys())
        if agent is not None:
            # Use agent value estimates.
            for s in agent.q_func.keys():
                val_text_dict[s.x][s.y] = int(visit_counts[s.x-1][s.y-1])
        else:
            # Use Value Iteration to compute value.
            vi = ValueIteration(grid_mdp, sample_rate=10)
            vi.run_vi()
            for s in vi.get_states():
                val_text_dict[s.x][s.y] = vi.get_value(s)

    # Make policy dict.
    policy_dict = defaultdict(lambda : defaultdict(str))
    if policy:
        vi = ValueIteration(grid_mdp)
        vi.run_vi()
        for s in vi.get_states():
            policy_dict[s.x][s.y] = policy(s)
    #print(policy_dict)

    # Prep some dimensions to make drawing easier.
    scr_width, scr_height = screen.get_width(), screen.get_height()
    width_buffer = scr_width / 10.0
    height_buffer = 30 + (scr_height / 10.0) # Add 30 for title.
    cell_width = (scr_width - width_buffer * 2) / grid_mdp.width 
    cell_height = (scr_height - height_buffer * 2) / grid_mdp.height
    goal_locs = grid_mdp.get_goal_locs()
    lava_locs = grid_mdp.get_lava_locs()
    sink_locs = grid_mdp.get_sink_locs()
    font_size = int(min(cell_width, cell_height) / 3.0)
    reg_font = pygame.font.SysFont("CMU Serif", font_size+12, bold = False)
    cc_font = pygame.font.SysFont("Courier", font_size*2 + 2, bold = True)

    # Draw the static entities.
    if draw_statics:
        # For each row:
        for i in range(grid_mdp.width):
            # For each column:
            for j in range(grid_mdp.height):

                top_left_point = width_buffer + cell_width*i, height_buffer + cell_height*j
                r = pygame.draw.rect(screen, (46, 49, 49), top_left_point + (cell_width, cell_height), 3)

                if policy and not grid_mdp.is_wall(i+1, grid_mdp.height - j):
                    a = policy_dict[i+1][grid_mdp.height - j]                    
                    if a[0] not in action_char_dict:
                        text_a = a[0]
                    else:
                        text_a = action_char_dict[a[0]]
                    text_center_point = int(top_left_point[0] + cell_width/2.0 - 10), int(top_left_point[1] + cell_height/3.0)
                    text_rendered_a = cc_font.render(text_a, True, (46, 49, 49))
                    screen.blit(text_rendered_a, text_center_point)

                if show_value and not grid_mdp.is_wall(i+1, grid_mdp.height - j):
                    # Draw the value.
                    val = val_text_dict[i+1][grid_mdp.height - j]
                    color = mdpv.count_to_color(val)
                    pygame.draw.rect(screen, color, top_left_point + (cell_width, cell_height), 0)

                if grid_mdp.is_wall(i+1, grid_mdp.height - j):
                    # Draw the walls.
                    top_left_point = width_buffer + cell_width*i + 5, height_buffer + cell_height*j + 5
                    r = pygame.draw.rect(screen, (94, 99, 99), top_left_point + (cell_width-10, cell_height-10), 0)

                if (i+1,grid_mdp.height - j) in goal_locs:
                    # Draw goal.
                    circle_center = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] + cell_height/2.0)
                    circler_color = (154, 195, 157)
                    #pygame.draw.circle(screen, circler_color, circle_center, int(min(cell_width, cell_height) / 3.0))

                if (i+1,grid_mdp.height - j) in lava_locs:
                    # Draw lava.
                    circle_center = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] + cell_height/2.0)
                    circler_color = (224, 145, 157)
                    #pygame.draw.circle(screen, circler_color, circle_center, int(min(cell_width, cell_height) / 4.0))
                    
                if (i+1,grid_mdp.height - j) in sink_locs:
                    # Draw sink.
                    circle_center = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] + cell_height/2.0)
                    circler_color = (204, 165, 107)
                    #pygame.draw.circle(screen, circler_color, circle_center, int(min(cell_width, cell_height) / 4.0))

                if show_value and not grid_mdp.is_wall(i+1, grid_mdp.height - j):
                    # Write value text to each state.
                    a = 1
                    value_text = reg_font.render(str(round(val, 3)), True, (46, 49, 49))
                    text_center_point = int(top_left_point[0] + cell_width/2.0 - 10), int(top_left_point[1] + cell_height/3.0)
                    screen.blit(value_text, text_center_point)


                # Current state.
                if show_value and (i+1,grid_mdp.height - j) == (state.x, state.y) and agent_shape is None:
                    tri_center = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] + cell_height/2.0)
                    #agent_shape = _draw_agent(tri_center, screen, base_size=min(cell_width, cell_height)/2.5 - 8)

    if agent_shape is not None:
        # Clear the old shape.
        pygame.draw.rect(screen, (255,255,255), agent_shape)
        top_left_point = width_buffer + cell_width*(state.x - 1), height_buffer + cell_height*(grid_mdp.height - state.y)
        tri_center = int(top_left_point[0] + cell_width/2.0), int(top_left_point[1] + cell_height/2.0)

        # Draw new.
        #agent_shape = _draw_agent(tri_center, screen, base_size=min(cell_width, cell_height)/2.5 - 8)

    pygame.display.flip()

    return agent_shape