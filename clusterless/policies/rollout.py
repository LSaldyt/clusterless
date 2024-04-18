from ..environment import transition, simulate

import numpy as np

empty_actions = lambda n : np.zeros(shape=(n, 2), dtype=np.int32)

def rollout(map, sense_info, base_policy, t, s):
    ''' Rollout for all agents. 
    Technically not MAR. This version assumes NO ORDERING, and each agent 
    pretends that *everyone* else is using the base policy'''
    # First calculate base policy for all agents
    base_policy_actions = base_policy(map, sense_info, base_policy, t, s)

    actions = empty_actions(map.agents_info.n_agents)
    for i, sense in enumerate(sense_info):
        actions[i, :] = egocentric_rollout(sense.memory, map.agents_info, 
                                           base_policy_actions, base_policy, 
                                           sense.code, t, s)
    return actions

def cost_to_go(env_map, policy, horizon, s):
    results = simulate(env_map, policy, policy, horizon,
                       0, s, do_render=False, check_goals=True, check_cycles=False)
    return results['score_d'] # Discounted score

def egocentric_rollout(mem, perfect_a_info, given_actions, base_policy, agent_code, t, s):
    ''' Egocentric 1-step lookahead with truncated rollout
        Requires s to define a base policy '''
    mem.map._inc_purity()
    a_info = mem.map.agents_info
    values = np.zeros(s.action_space.shape[0], dtype=np.float32)

    given_indices = [np.argmax(perfect_a_info.codes == code)
                     for code in a_info.codes]
    future_actions = given_actions[given_indices]
    ego_i = np.argmax(a_info.codes == agent_code) # Our index

    for j, action in enumerate(s.action_space):
        future_actions[ego_i, :] = action

        next_map  = mem.map.clone()
        info      = transition(next_map, future_actions, s) # Modifies next_map
        horizon   = np.minimum(s.timesteps - t, s.truncated_timesteps)
        score_d   = cost_to_go(next_map, base_policy, horizon, s)

        immediate_coll = info['n_collisions_obstacle'] + info['n_collisions_agents']
        if immediate_coll > 0:
            values[j] = -np.infty
        else:
            values[j] = info['n_goals_achieved'] + s.discount * score_d

    return s.action_space[np.argmax(values)]
