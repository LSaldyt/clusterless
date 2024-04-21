from clusterless.memory import map_for_simulate, sense_environment
from ..environment import transition, simulate
import rich

import numpy as np

from .utils import empty_actions

def rollout(map, sense_info, memory, base_policy, t, s):
    ''' Rollout for all agents. 
    Technically not MAR. This version assumes NO ORDERING, and each agent 
    pretends that *everyone* else is using the base policy'''
    # First calculate base policy for all agents
    base_policy_actions = base_policy(map, sense_info, memory, base_policy, t, s)
    assert not (base_policy_actions == 0).all(), f'Base policy returned all stay actions, reject!!'
    codes = [sense.code for sense in sense_info]

    actions = empty_actions(len(sense_info))
    for i, sense in enumerate(sense_info):
        actions[i, :] = egocentric_rollout(sense.memory, codes, memory,
                                           base_policy_actions, base_policy, 
                                           sense.code, t, s)
    return actions

def cost_to_go(env_map, memory, policy, horizon, s, do_render=False, start_t=0):
    results = simulate(env_map, policy, policy, horizon, s, check_goals=True, check_cycles=False, do_render=do_render, memory=memory, start_t=start_t)
    return results

def egocentric_rollout(mem, codes, memory, given_actions, base_policy, agent_code, t, s, mask_unseen=False):    
    ''' Egocentric 1-step lookahead with truncated rollout
        Requires s to define a base policy '''
    values  = np.zeros(s.action_space.shape[0], dtype=np.float32)
    mem_map = map_for_simulate(mem, s, duplicates_only=s.rollout_duplicates_only, mask_unseen=mask_unseen) # This line will potentially delete agents we can't see!!
    a_info  = mem_map.agents_info
    if a_info.n_agents == 0:
        return s.action_space[-1, :]
    sense_codes    = a_info.codes
    matching_codes = [c in sense_codes for c in codes]
    # Prune future actions for agents we don't know about!!
    print('Given future actions')
    print(given_actions)
    future_actions = given_actions[np.arange(len(codes))[matching_codes]]
    print(future_actions)
    assert future_actions.shape[0] == a_info.n_agents, f'actions = {future_actions.shape}[0] != {a_info.n_agents}'

    ego_i = list(sense_codes).index(agent_code)

    for j, action in enumerate(s.action_space):
        next_map  = mem_map.clone()
        print('*' * 80)
        print(f'Imagined ROLLOUT for ', s.action_words[j], action)
        future_actions[ego_i, :] = action
        code_mask = np.array([c in next_map.agents_info.codes for c in a_info.codes]) # So we remove unseen agents
        acts      = future_actions[np.arange(a_info.n_agents)[code_mask]]
        # next_map.color_render()
        info      = transition(next_map, acts, s) # Modifies next_map
        # next_map.color_render()
        horizon   = np.minimum(s.timesteps - t, s.truncated_timesteps)
        remain    = cost_to_go(next_map, memory, base_policy, horizon, s, do_render=False, start_t=t)

        immediate_coll = info['n_collisions_obstacle'] + info['n_collisions_agents']
        if immediate_coll > 0:
            values[j] = -np.infty
        else:
            values[j] = info['n_goals_achieved'] + s.discount * remain['score_d']
        print(s.action_words[j])
        print(info)
        print(remain)
        print('*' * 80)
    rich.print(values)
    if np.max(values) == 0: # All goals in seen region are explored..
        print(f'ROLLOUT defaulted to BASE POLICY')
        ego_i = list(codes).index(agent_code)
        print(given_actions[ego_i, :])
        return given_actions[ego_i, :]
    return s.action_space[np.argmax(values)]
