import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from . import utils

from .map import Map

@dataclass
class Memory():
    grid : npt.ArrayLike
    time : npt.ArrayLike

def transition(map, actions, agent_coords, agent_codes, codes):
    ''' Note: This function MODIFIES map intentionally, for efficiency.

        Move all agents at once according to their actions:
        Agent actions can be stay, up, down, left, and right
        If an agent touches a goal, that goal is achieved. 
        Agents cannot step into obstacles.. or they stay in place at a penalty 
        Agents cannot collide.. or they die? ☠'''
    # Progress actions, (optionally) enforce map boundaries
    next_coords    = (agent_coords + actions) 
    next_coords    = next_coords % (map.grid.shape[0])
    next_locations = map.grid[next_coords[:, 0], next_coords[:, 1]]
    # Enforce obstacles (and dead agents as obstacles)
    allowed_move   = ((next_locations != codes['obstacle']) & (next_locations != codes['dead']))
    final_coords   = np.where(utils.broadcast(allowed_move, 2), next_coords, agent_coords)
    # Check for inter-agent collisions
    unique_coords, unique_counts = np.unique(final_coords, axis=0, return_counts=True)
    collision_mask   = unique_counts > 1
    collision_coords = unique_coords[collision_mask]
    if (collision_mask).any():
        map.set_at(collision_coords, codes['dead'])
    non_collision_coords = unique_coords[collision_mask == False]
    reached_locations    = map.grid[non_collision_coords[:, 0], non_collision_coords[:, 1]]
    # Move agents to (filtered) locations
    map.set_at(agent_coords, codes['empty'])
    map.set_at(final_coords, agent_codes)

    # Count goals and collision types
    goal_mask = reached_locations == codes['goal']
    return dict(
        n_goals_achieved      = np.sum(goal_mask),
        n_collisions_obstacle = np.sum(1 - allowed_move),
        n_collisions_agents   = np.sum(unique_counts) - unique_counts.shape[0],
    )

def views(map, agent_coords, s):
    ''' Produce local views for all agents.
        A view is a local n x n map around an agent.
        View that go beyond the border of the map are padded. '''
    n_agents = agent_coords.shape[0]
    view_box = utils.box(s.view_size)
    for i in range(n_agents): # Seemed intuitive, faster and more memory efficient than a vectorized operation
        view_coords    = view_box + agent_coords[i, :]
        view_coords    = view_coords % map.grid.shape[0]
        view           = map.grid[view_coords[:, 0], view_coords[:, 1]]
        yield agent_coords[i, :], view_coords, view.reshape((s.view_size, s.view_size))

def get_agents(map, s):
    mask     = map.grid >= s.codes['agent']
    codes    = map.grid[mask]
    coords   = map.coords_of(mask)
    n_agents = np.sum(mask)
    return codes, coords, n_agents

''' Memory is a series of views over time 
    Agents will move, goals may be taken, 
    otherwise empty/obstacle are certain, 
    agents/goals in current time step are certain '''

def init_memory(map, agent_codes, codes):
    return {k : Memory(np.full(map.grid.shape, codes['unseen']), 
                       np.full(map.grid.shape, 0)) 
            for k in agent_codes}

def sense_environment(map, memory, agent_codes, agent_coords, s, timestep):
    for c, (ac, view_coords, view) in zip(agent_codes, views(map, agent_coords, s)): # Important: views should be taken before transitions for consistency
        memory[c].grid[view_coords[:, 0], view_coords[:, 1]] = view.ravel() 
        memory[c].time[view_coords[:, 0], view_coords[:, 1]] = timestep
        yield c, view, memory[c], ac

def merge_memory(mem_a, mem_b):
    return np.where(mem_a.time < mem_b.time, mem_a.grid, mem_b.grid)

def simulate(map, policy, s, do_render=False): 
    score   = 0 # Number of goals achieved
    n_goals = map.count('goal')

    agent_codes, agent_coords, n_agents = get_agents(map, s)
    memory = init_memory(map, agent_codes, s.codes)

    step_count = s.timesteps
    for t in range(s.timesteps):
        agent_codes, agent_coords, n_agents = get_agents(map, s)
        sense_input = list(sense_environment(map, memory, agent_codes, agent_coords, s, t))

        if do_render:
            map.full_render(sense_input, s)

        actions = policy(s, n_agents, sense_input, map.coordinates)
        info    = transition(map, actions, agent_coords, agent_codes, s.codes) # Important: Do transition at the end of the loop

        score += info['n_goals_achieved']
        info.update(score=score, n_goals=n_goals)
        remaining_goals = map.count('goal')
        if remaining_goals == 0 or n_agents == 0:
            sense_input = list(sense_environment(map, memory, agent_codes, agent_coords, s, t))
            step_count = t if remaining_goals==0 else s.timesteps
            break
    assert score <= n_goals
    return dict(score=score, percent=score/n_goals, step_count=step_count)
