from ..environment import transition, simulate

import numpy as np

empty_actions = lambda n : np.zeros(shape=(n, 2), dtype=np.int32)

def rollout(map, sense_info, base_policy, s):
    ''' Rollout for all agents '''
    # First calculate base policy for all agents
    base_policy_actions = base_policy(map, sense_info, base_policy, s)

    actions = empty_actions(map.agents_info.n_agents)
    for i, (c, mem, _) in enumerate(sense_info):
        actions[i, :] = rollout_egocentric(mem, map.agents_info, 
                                           base_policy_actions, base_policy, c, s)
    return actions

def rollout_egocentric(mem, perfect_a_info, base_policy_actions, base_policy, agent_code, s):
    ''' Egocentric 1-step lookahead with truncated rollout
        Requires s to define a base policy '''
    a_info = mem.map.agents_info
    values = np.zeros(s.action_space.shape[0], dtype=np.float32)

    bp_indices = [np.argmax(perfect_a_info.codes == code)
                  for code in a_info.codes]
    future_actions = base_policy_actions[bp_indices]
    ego_i = np.argmax(a_info.codes == agent_code) # Our index

    # print('future', future_actions)
    for j, action in enumerate(s.action_space):
        future_actions[ego_i, :] = action

        print('-' * 80)
        print('future', future_actions)
        next_map  = mem.map.clone()
        info      = transition(next_map, future_actions, s) # Modifies next_map
        info['score'] = info['n_goals_achieved']
        info['step_count'] = 1 # type: ignore
        next_map.render_grid()
        print(info)
        lookahead_miracle = next_map.count('goal') == 0
        results   = simulate(next_map, base_policy, base_policy, 
                             s.truncated_timesteps, s, do_render=False, check_goals=True)
        results   = {k : results.get(k, 0) + info.get(k, 0) for k in results}
        results['percent'] = results['score'] / mem.map.count('goal')
        if lookahead_miracle:
            results['step_count'] = results['step_count'] - 1
        print(results)
        immediate_coll = info['n_collisions_obstacle'] + info['n_collisions_agents']
        if immediate_coll > 0:
            values[j] = -np.infty
        else:
            values[j] = results['percent'] / results['step_count']
        print(values[j], action)
        print('-' * 80)
    # print(s.action_space)
    print(values)
    print(s.action_space[np.argmax(values)])

    return s.action_space[np.argmax(values)]
