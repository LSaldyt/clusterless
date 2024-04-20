from clusterless.memory import map_for_simulate
from ..environment import transition, simulate

import numpy as np

from .utils import empty_actions

def rollout(map, sense_info, memory, base_policy, t, s):
    ''' Rollout for all agents. 
    Technically not MAR. This version assumes NO ORDERING, and each agent 
    pretends that *everyone* else is using the base policy'''
    # First calculate base policy for all agents
    base_policy_actions = base_policy(map, sense_info, base_policy, t, s)

    codes = [sense.code for sense in sense_info]

    actions = empty_actions(len(sense_info))
    for i, sense in enumerate(sense_info):
        actions[i, :] = egocentric_rollout(sense.memory, codes,
                                           base_policy_actions, base_policy, 
                                           sense.code, t, s)
    return actions

def cost_to_go(env_map, policy, horizon, s, do_render=False):
    results = simulate(env_map, policy, policy, horizon,
                       0, s, do_render=do_render, check_goals=True, check_cycles=False)
    return results

def egocentric_rollout(mem, codes, future_actions, base_policy, agent_code, t, s, do_render=False):
    ''' Egocentric 1-step lookahead with truncated rollout
        Requires s to define a base policy '''
    values  = np.zeros(s.action_space.shape[0], dtype=np.float32)
    ego_i   = codes.index(agent_code)
    mem_map = map_for_simulate(mem, s)
    a_info  = mem_map.agents_info

    for j, action in enumerate(s.action_space):
        future_actions[ego_i, :] = action
        next_map  = map_for_simulate(mem, s) # This line will potentially delete agents we can't see!!
        code_mask = np.array([c in next_map.agents_info.codes for c in a_info.codes]) # So we remove unseen agents
        acts      = future_actions[np.arange(a_info.n_agents)[code_mask]]
        info      = transition(next_map, acts, s) # Modifies next_map
        if do_render:
            next_map.color_render()
        horizon   = np.minimum(s.timesteps - t, s.truncated_timesteps)
        score_d   = cost_to_go(next_map, base_policy, horizon, s, do_render=do_render)['score_d']

        immediate_coll = info['n_collisions_obstacle'] + info['n_collisions_agents']
        if immediate_coll > 0:
            values[j] = -np.infty
        else:
            values[j] = info['n_goals_achieved'] + s.discount * score_d

    return s.action_space[np.argmax(values)]
