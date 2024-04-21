import numpy as np
from collections import Counter
from rich.pretty import pprint

from . import utils
from .utils import at_xy
from .map import Map
from .memory import init_memory, sense_environment
from .belief import init_beliefs, update_belief_from_ground_truth

# TODO introduce a simulation settings type to consolidate this garbage :)
# TODO Simplify this function and maybe remove about 10-20 lines
def simulate(env_map, policy, base_policy, timesteps, s, 
             do_render=False, check_goals=True, check_cycles=True, 
             progress=None, task=None, log_fn=None, extra=None,
             track_beliefs=True,
             memory=None, start_t=0): 
    score      = 0 # Number of goals achieved
    score_d    = 0 # Discounted score
    n_goals    = env_map.count('goal')
    memory     = init_memory(env_map, s)  if memory is None else memory 
    beliefs    = init_beliefs(env_map, s) if track_beliefs else None
    map_hashes = dict()
    cumulative = Counter(n_goals_achieved=0, n_collisions_obstacle=0, n_collisions_agents=0)
    log_fn     = log_fn if log_fn is not None else lambda *x, **k : None
    extra      = extra  if extra  is not None else dict()

    step_count = timesteps
    for t in range(start_t, start_t + timesteps):
        sense_input = list(sense_environment(env_map, memory, s, t))
        env_hash    = env_map.hash()

        if track_beliefs:
            for sense in sense_input:
                belief = beliefs[sense.code] # type: ignore
                update_belief_from_ground_truth(s, belief, sense) # Woo! We have belief states!

        if do_render:
            print(f'Hash: {env_hash}')
            env_map.full_render(sense_input)

        try:
            actions = policy(env_map, sense_input, memory, base_policy, t, s)
            info    = transition(env_map, actions, s) # Important: Do transition at the end of the loop
            if progress is not None and task is not None:
                progress.update(task, advance=1) # type: ignore
        except utils.UnsolvableException:
            log_fn('unsolvable', dict(**extra))
            break

        cumulative = {k : cumulative[k] + vn for k, vn in info.items()}

        score   += info['n_goals_achieved'] 
        score_d += info['n_goals_achieved'] * (s.discount)**(t)
        remaining_goals = env_map.count('goal')

        if do_render:
            print(f'Step: {t} {score} {info} env = {extra.get("env", "imagined")}')
            action_repr = ' '.join(s.action_lookup[str(tuple(a))] for a in actions)
            print(f'Acts: {action_repr}')
            print('-' * 80)
            if s.detect_cycles and check_cycles: # So that cycles can be disabled globally AND locally
                detect_cycles(env_hash, map_hashes, action_repr, policy, s)

        # Log all information to parent experiment in simulation.csv
        log_fn('simulation', dict(timestep=t, score_d=score_d, score=score,
            wait=(actions == 0).all(), 
            n_moves=np.sum(np.abs(actions)), **info, **extra))

        if ((check_goals and remaining_goals == 0) 
            or env_map.agents_info.n_agents == 0):
            step_count  = t + 1 # if remaining_goals == 0 else timesteps
            break

    if do_render:
        pprint(map_hashes)

    assert score <= n_goals
    metrics = dict(score=score, score_d=score_d, percent=score/n_goals, step_count=step_count, **dict(cumulative))
    return metrics

def transition(env_map, actions, s):
    ''' Note: This function MODIFIES map intentionally, for efficiency.

        Move all agents at once according to their actions:
        Agent actions can choose: stay, up, down, left, and right
        If an agent touches a goal, that goal is achieved. 
        Agents cannot step into obstacles.. or they stay in place at a penalty 
        Agents cannot collide.. or they die? ☠'''
    a_info = env_map.agents_info
    # Progress actions, (optionally) enforce map boundaries
    assert (np.sum(np.abs(actions), axis=1) <= 1).all()
    assert actions.shape == a_info.coords.shape, f'coords={a_info.coords.shape} != actions={actions.shape}'
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
    non_collision_coords = unique_coords[collision_mask == False]
    reached_locations    = env_map.grid[*at_xy(non_collision_coords)]
    # Move agents to (filtered) locations
    env_map.set_at(a_info.coords, s.codes['empty'])
    env_map.set_at(final_coords,  a_info.codes)
    # Set dead agents last (overwriting agent codes at collision coords)
    if (collision_mask).any():
        env_map.set_at(collision_coords, s.codes['dead'])

    # Count goals and collision types
    goal_mask = reached_locations == s.codes['goal']
    return dict(
        n_goals_achieved      = np.sum(goal_mask),
        n_collisions_obstacle = np.sum(1 - allowed_move),
        n_collisions_agents   = np.sum(unique_counts) - unique_counts.shape[0],
    )

class CircularBehaviorException(RuntimeError):
    pass

def detect_cycles(env_hash, map_hashes, acts, policy, s):
    if env_hash not in map_hashes:
        map_hashes[env_hash] = acts
    else:
        print(f'Repeated hash: {env_hash}!!!')
        print(f'Last action: {map_hashes[env_hash]} ? {acts}')
        print(policy)
        pprint(map_hashes)
        raise CircularBehaviorException(f'Circular behavior detected!!')
