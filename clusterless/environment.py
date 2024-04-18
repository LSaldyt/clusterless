import numpy as np
from collections import Counter

from rich.progress import track

from . import utils
from .utils import at_xy
from .map import Map
from .memory import init_memory, sense_environment

class CircularBehaviorException(RuntimeError):
    pass

def transition(env_map, actions, s):
    ''' Note: This function MODIFIES map intentionally, for efficiency.

        Move all agents at once according to their actions:
        Agent actions can choose: stay, up, down, left, and right
        If an agent touches a goal, that goal is achieved. 
        Agents cannot step into obstacles.. or they stay in place at a penalty 
        Agents cannot collide.. or they die? ☠'''
    a_info = env_map.agents_info
    # Progress actions, (optionally) enforce map boundaries
    assert (np.sum(np.abs(actions),axis=1)<=1).all()
    assert actions.shape == a_info.coords.shape, f'Future actions must match existing agent count, {a_info.coords.shape}, {actions.shape}'
    next_coords    = (a_info.coords + actions) 
    next_coords    = next_coords % (env_map.grid.shape[0])
    next_locations = env_map.grid[*at_xy(next_coords)]
    # Enforce obstacles (and dead agents as obstacles)
    allowed_move   = ((next_locations != s.codes['obstacle']) & (next_locations != s.codes['dead']))
    final_coords   = np.where(utils.broadcast(allowed_move, 2), next_coords, a_info.coords)
    # Check for inter-agent collisions
    unique_coords, unique_counts = np.unique(final_coords, axis=0, return_counts=True)
    collision_mask   = unique_counts > 1
    collision_coords = unique_coords[collision_mask]
    if (collision_mask).any():
        env_map.set_at(collision_coords, s.codes['dead'])
    non_collision_coords = unique_coords[collision_mask == False]
    reached_locations    = env_map.grid[*at_xy(non_collision_coords)]
    # Move agents to (filtered) locations
    env_map.set_at(a_info.coords, s.codes['empty'])
    env_map.set_at(final_coords,  a_info.codes)

    # Count goals and collision types
    goal_mask = reached_locations == s.codes['goal']
    return dict(
        n_goals_achieved      = np.sum(goal_mask),
        n_collisions_obstacle = np.sum(1 - allowed_move),
        n_collisions_agents   = np.sum(unique_counts) - unique_counts.shape[0],
    )

def detect_cycles(env_hash, unique_maps, trace, do_render, policy, s):
    if env_hash not in unique_maps:
        unique_maps.add(env_hash)
    else:
        print(f'Repeated hash: {env_hash}!!!')
        print(policy)
        for actions, old_map in trace[-s.debug_trace_depth:]:
            print(old_map.hash())
            old_map.render_grid()
        exit()
        raise CircularBehaviorException(f'Circular behavior detected!!')

def simulate(env_map, policy, base_policy, timesteps, env_index, s, do_render=False, check_goals=True, check_cycles=True, progress=None, task=None): 
    score   = 0 # Number of goals achieved
    score_d = 0 # Discounted score
    n_goals = env_map.count('goal')
    memory  = init_memory(env_map, s)

    unique_maps = set()
    trace       = list()

    cumulative = Counter(n_goals_achieved=0, n_collisions_obstacle=0, n_collisions_agents=0)

    scores = []

    step_count = timesteps
    for t in range(timesteps):
        sense_input = list(sense_environment(env_map, memory, s, t))

        env_hash = env_map.hash()
        if s.detect_cycles and check_cycles: # So that cycles can be disabled globally AND locally
            detect_cycles(env_hash, unique_maps, trace, do_render, policy, s)

        if do_render:
            print(f'Environment hash: {env_hash}')
            env_map.full_render(sense_input)

        try:
            actions = policy(env_map, sense_input, base_policy, t, s)
            info    = transition(env_map, actions, s) # Important: Do transition at the end of the loop
            if progress is not None and task is not None:
                progress.update(task, advance=1) # type: ignore
        except utils.UnsolvableException:
            break

        if s.debug:
            trace.append((actions, env_map.clone()))

        cumulative = {k : cumulative[k] + vn for k, vn in info.items()}

        scores.append(info['n_goals_achieved'])

        score   += info['n_goals_achieved'] 
        score_d += info['n_goals_achieved'] * (s.discount)**(t)
        remaining_goals = env_map.count('goal')
        if do_render:
            print(f'Step {t} {info} env = {env_index}')
        if ((check_goals and remaining_goals == 0) 
            or env_map.agents_info.n_agents == 0):
            sense_input = list(sense_environment(env_map, memory, s, t))
            step_count  = t + 1 if remaining_goals == 0 else timesteps
            break
    print(scores)
    assert score <= n_goals
    return dict(score=score, score_d=score_d, percent=score/n_goals, step_count=step_count, **dict(cumulative))
