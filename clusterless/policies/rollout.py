from ..environment import transition, simulate

import numpy as np

empty_actions = lambda n : np.zeros(shape=(n, 2), dtype=np.int32)

def rollout(map, sense_info, base_policy, t, s):
    ''' Rollout for all agents '''
    # First calculate base policy for all agents
    base_policy_actions = base_policy(map, sense_info, base_policy, t, s)

    actions = empty_actions(map.agents_info.n_agents)
    for i, (c, mem, _) in enumerate(sense_info):
        actions[i, :] = rollout_egocentric(mem, map.agents_info, 
                                           base_policy_actions, base_policy, c, t, s)
    return actions

def rollout_egocentric(mem, perfect_a_info, base_policy_actions, base_policy, agent_code, t, s):
    ''' Egocentric 1-step lookahead with truncated rollout
        Requires s to define a base policy '''
    mem.map._inc_purity()
    a_info = mem.map.agents_info
    values = np.zeros(s.action_space.shape[0], dtype=np.float32)

    bp_indices = [np.argmax(perfect_a_info.codes == code)
                  for code in a_info.codes]
    future_actions = base_policy_actions[bp_indices]
    ego_i = np.argmax(a_info.codes == agent_code) # Our index

    for j, action in enumerate(s.action_space):
        future_actions[ego_i, :] = action
        next_map  = mem.map.clone()
        info      = transition(next_map, future_actions, s) # Modifies next_map
        info['score']   = info['n_goals_achieved']
        info['score_d'] = info['n_goals_achieved']
        info['step_count'] = 1 # type: ignore
        lookahead_miracle = next_map.count('goal') == 0
        rollout_timesteps = np.minimum(s.timesteps - t, s.truncated_timesteps)
        results   = simulate(next_map, base_policy, base_policy, rollout_timesteps,
                             0, s, do_render=False, check_goals=True, check_cycles=False)
        results   = {k : results.get(k, 0) + info.get(k, 0) for k in results}
        results['percent'] = results['score'] / mem.map.count('goal')
        if lookahead_miracle:
            results['step_count'] = results['step_count'] - 1
        immediate_coll = info['n_collisions_obstacle'] + info['n_collisions_agents']
        if immediate_coll > 0:
            values[j] = -np.infty
        else:
            values[j] = results['score_d'] 
        a = s.action_words[j]
        v = values[j]
        print(f'{a:10} {v:4.4f} {results["score_d"]:3} {results["step_count"]:<3}')
    chosen = np.argmax(values)
    print(f'Rollout action: {s.action_words[chosen]}')
    return s.action_space[np.argmax(values)]
