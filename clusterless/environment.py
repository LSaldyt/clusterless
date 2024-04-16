import numpy as np

from . import utils
from .map import Map
from .memory import init_memory, sense_environment

def transition(map, actions, s):
    ''' Note: This function MODIFIES map intentionally, for efficiency.

        Move all agents at once according to their actions:
        Agent actions can be stay, up, down, left, and right
        If an agent touches a goal, that goal is achieved. 
        Agents cannot step into obstacles.. or they stay in place at a penalty 
        Agents cannot collide.. or they die? ☠'''
    a_info = map.agents_info
    # Progress actions, (optionally) enforce map boundaries
    next_coords    = (a_info.coords + actions) 
    next_coords    = next_coords % (map.grid.shape[0])
    next_locations = map.grid[next_coords[:, 0], next_coords[:, 1]]
    # Enforce obstacles (and dead agents as obstacles)
    allowed_move   = ((next_locations != s.codes['obstacle']) & (next_locations != s.codes['dead']))
    final_coords   = np.where(utils.broadcast(allowed_move, 2), next_coords, a_info.coords)
    # Check for inter-agent collisions
    unique_coords, unique_counts = np.unique(final_coords, axis=0, return_counts=True)
    collision_mask   = unique_counts > 1
    collision_coords = unique_coords[collision_mask]
    if (collision_mask).any():
        map.set_at(collision_coords, s.codes['dead'])
    non_collision_coords = unique_coords[collision_mask == False]
    reached_locations    = map.grid[non_collision_coords[:, 0], non_collision_coords[:, 1]]
    # Move agents to (filtered) locations
    map.set_at(a_info.coords, s.codes['empty'])
    map.set_at(final_coords,  a_info.codes)

    # Count goals and collision types
    goal_mask = reached_locations == s.codes['goal']
    return dict(
        n_goals_achieved      = np.sum(goal_mask),
        n_collisions_obstacle = np.sum(1 - allowed_move),
        n_collisions_agents   = np.sum(unique_counts) - unique_counts.shape[0],
    )

def simulate(map, policy, base_policy, s, do_render=False): 
    score   = 0 # Number of goals achieved
    n_goals = map.count('goal')
    memory  = init_memory(map, s)

    step_count = s.timesteps
    for t in range(s.timesteps):
        sense_input = list(sense_environment(map, memory, s, t))

        if do_render:
            map.full_render(sense_input, s)

        actions = policy(map, sense_input, base_policy, s)
        info    = transition(map, actions, s) # Important: Do transition at the end of the loop

        score += info['n_goals_achieved']
        info.update(score=score, n_goals=n_goals)
        remaining_goals = map.count('goal')
        if remaining_goals == 0 or map.agents_info.n_agents == 0:
            sense_input = list(sense_environment(map, memory, s, t))
            step_count = t if remaining_goals==0 else s.timesteps
            break
    assert score <= n_goals
    return dict(score=score, percent=score/n_goals, step_count=step_count)
