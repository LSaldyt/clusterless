from copy import deepcopy
import numpy as np
import rich

from ..environment import transition, simulate
from ..memory      import Memory, map_for_simulate, sense_environment
from ..utils       import PolicyInputs

from .utils import empty_actions

def rollout(p, s):
    ''' Rollout for all agents. 
    Technically not MAR. This version assumes NO ORDERING, and each agent 
    pretends that *everyone* else is using the base policy'''
    # First calculate base policy for all agents
    base_policy_actions = p.base_policy(p, s)
    assert not (base_policy_actions == 0).all(), f'Base policy returned all stay actions, reject!!'
    codes = [sense.code for sense in p.sense_info]

    actions = empty_actions(len(p.sense_info))
    for i, sense in enumerate(p.sense_info):
        actions[i, :] = egocentric_rollout(map, sense.memory, codes, p.memory,
                                           base_policy_actions, p.base_policy, 
                                           sense.code, p.t + 1, s, reveal=False)[0]
    return actions

def cost_to_go(env_map, memory, policy, horizon, s, do_render=False, start_t=0, mask_unseen=False):
    memory  = deepcopy(memory)
    memory  = {c : Memory(map_for_simulate(mem, s, duplicates_only=s.rollout_duplicates_only, mask_unseen=mask_unseen),
                      mem.time) if c != '__last_updated' else mem
               for c, mem in memory.items() }
    results = simulate(env_map, policy, policy, horizon, s, 
                       check_goals=True, check_cycles=False, 
                       do_render=do_render, memory=memory, 
                       start_t=start_t,
                       track_beliefs=False) # Do NOT track beliefs within rollout!
    return results

def egocentric_rollout(ground_truth_map, mem, codes, in_memory, given_actions, base_policy, agent_code, t, s, mask_unseen=False, reveal=False):    
    ''' Egocentric 1-step lookahead with truncated rollout
        Requires s to define a base policy '''
    memory  = deepcopy(in_memory)
    values  = np.zeros(s.action_space.shape[0], dtype=np.float32)
    if reveal:
        mem_map = ground_truth_map.clone()
    else:
        mem_map = map_for_simulate(mem, s, duplicates_only=s.rollout_duplicates_only, mask_unseen=mask_unseen) # This line will potentially delete agents we can't see!!
    a_info  = mem_map.agents_info
    if a_info.n_agents == 0:
        return s.action_space[-1, :]
    sense_codes    = a_info.codes
    matching_codes = [c in sense_codes for c in codes]
    # Prune future actions for agents we don't know about!!
    future_actions = given_actions[np.arange(len(codes))[matching_codes]]
    if future_actions.shape[0] != a_info.n_agents:
        print(f'WARNING: actions = {future_actions.shape}[0] != {a_info.n_agents}')
        future_actions = given_actions

    ego_i = list(sense_codes).index(agent_code)

    # print(f'Rollout map!')
    # mem_map.color_render()

    for j, action in enumerate(s.action_space):
        next_map  = mem_map.clone()
        # print('*' * 80)
        future_actions[ego_i, :] = action
        code_mask = np.array([c in next_map.agents_info.codes for c in a_info.codes]) # So we remove unseen agents
        acts      = future_actions[np.arange(a_info.n_agents)[code_mask]]
        info      = transition(next_map, acts, s) # Modifies next_map
        horizon   = np.minimum(s.timesteps - t, s.truncated_timesteps)
        next_mem  = deepcopy(memory)
        senses    = list(sense_environment(next_map, next_mem, s, t + 1))
        remain    = cost_to_go(next_map, next_mem, base_policy, horizon, s, do_render=False, start_t=t+2, mask_unseen=mask_unseen)

        # print(s.action_words[j])
        # print(f'Rollout end map!')
        # next_map.color_render()
        # print('score', remain['score'])
        # print('*' * 80)

        immediate_coll = info['n_collisions_obstacle'] + info['n_collisions_agents']
        if immediate_coll > 0:
            values[j] = -np.infty
        else:
            values[j] = info['n_goals_achieved'] + s.discount * remain['score_d']

    # rich.print(values)
    if np.max(values) == 0: # All goals in seen region are explored..
        # print(f'ROLLOUT defaulted to BASE POLICY') # Should only happen when we exhaust goals in a view
        ego_i = list(codes).index(agent_code)
        act = given_actions[ego_i, :]
        val = 0 # Bogus 
    else:
        act, val = s.action_space[np.argmax(values)], np.max(values)
    if act.shape != (1, 2): # Not sure why this is needed
        act = np.expand_dims(np.array([act[0], act[1]]), 0)
    # print(act, val)
    assert act.shape == (1, 2)
    return act, val
