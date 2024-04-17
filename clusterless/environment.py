import numpy as np
from collections import Counter
from functools import reduce
import operator

from . import utils
from .map import Map
from .memory import init_memory, sense_environment

class CircularBehaviorException(RuntimeError):
    pass

def transition(map, actions, s):
    ''' Note: This function MODIFIES map intentionally, for efficiency.

        Move all agents at once according to their actions:
        Agent actions can choose: stay, up, down, left, and right
        If an agent touches a goal, that goal is achieved. 
        Agents cannot step into obstacles.. or they stay in place at a penalty 
        Agents cannot collide.. or they die? ☠'''
    a_info = map.agents_info
    # Progress actions, (optionally) enforce map boundaries
    assert (np.sum(np.abs(actions),axis=1)<=1).all()
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

def simulate(env_map, policy, base_policy, timesteps, env_index, s, do_render=False, check_goals=True): 
    score   = 0 # Number of goals achieved
    n_goals = env_map.count('goal')
    memory  = init_memory(env_map, s)

    unique_maps = set()
    trace       = list()

    cumulative = Counter(n_goals_achieved=0, n_collisions_obstacle=0, n_collisions_agents=0)

    step_count = timesteps
    for t in range(timesteps):
        sense_input = list(sense_environment(env_map, memory, s, t))

        if do_render:
            env_map.full_render(sense_input)

        actions = policy(env_map, sense_input, base_policy, t, s)
        info    = transition(env_map, actions, s) # Important: Do transition at the end of the loop

        if s.debug:
            trace.append((actions, env_map.clone()))

        if s.detect_cycles:
            env_hash = env_map.hash()
            if env_hash not in unique_maps:
                unique_maps.add(env_hash)
            else:
                for actions, old_map in trace[-s.debug_trace_depth:]:
                    old_map.render_grid()
                raise CircularBehaviorException(f'Circular behavior detected!!')

        cumulative = {k : cumulative[k] + vn for k, vn in info.items()}

        score += info['n_goals_achieved'] * (s.discount)**(t)
        remaining_goals = env_map.count('goal')
        if do_render:
            print(f'Step {t} {info} env = {env_index}')
        if ((check_goals and remaining_goals == 0) 
            or env_map.agents_info.n_agents == 0):
            sense_input = list(sense_environment(env_map, memory, s, t))
            step_count  = t + 1 if remaining_goals == 0 else timesteps
            break
    assert score <= n_goals
    return dict(score=score, percent=score/n_goals, step_count=step_count, **dict(cumulative))
