from ..environment import transition, simulate

import numpy as np

from .utils import empty_actions

def rollout(map, sense_info, base_policy, t, s):
    ''' Rollout for all agents. 
    Technically not MAR. This version assumes NO ORDERING, and each agent 
    pretends that *everyone* else is using the base policy'''
    # First calculate base policy for all agents
    base_policy_actions = base_policy(map, sense_info, base_policy, t, s)

    codes = [sense.code for sense in sense_info]

    actions = empty_actions(len(sense_info))
    for i, sense in enumerate(sense_info):
        actions[i, :] = egocentric_rollout(sense.memory, map.agents_info, codes,
                                           base_policy_actions, base_policy, 
                                           sense.code, t, s)
    return actions

def cost_to_go(env_map, policy, horizon, s):
    results = simulate(env_map, policy, policy, horizon,
                       0, s, do_render=False, check_goals=True, check_cycles=False)
    return results

def make_actions(acts, map, codes):
    code_l     = list(map.agents_info.codes)
    ind        = np.array([code_l.index(c) for c in codes])
    empty      = empty_actions(map.agents_info.n_agents)
    empty[ind] = acts
    return empty

def egocentric_rollout(mem, perfect_a_info, codes, given_actions, base_policy, agent_code, t, s):
    ''' Egocentric 1-step lookahead with truncated rollout
        Requires s to define a base policy '''
    mem.map._inc_purity()
    values = np.zeros(s.action_space.shape[0], dtype=np.float32)

    future_actions = given_actions
    ego_i          = codes.index(agent_code)

    for j, action in enumerate(s.action_space):
        future_actions[ego_i, :] = action

        next_map  = mem.map.clone()
        actions   = make_actions(future_actions, mem.map, codes)
        info      = transition(next_map, actions, s) # Modifies next_map
        horizon   = np.minimum(s.timesteps - t, s.truncated_timesteps)
        score_d   = cost_to_go(next_map, base_policy, horizon, s)['score_d']

        immediate_coll = info['n_collisions_obstacle'] + info['n_collisions_agents']
        if immediate_coll > 0:
            values[j] = -np.infty
        else:
            values[j] = info['n_goals_achieved'] + s.discount * score_d

    return s.action_space[np.argmax(values)]
