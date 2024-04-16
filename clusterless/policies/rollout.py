from ..environment import transition, simulate
from rich.pretty import pprint

import numpy as np

empty_actions = lambda n : np.zeros(shape=(n, 2), dtype=np.int32)

def rollout(map, sense_info, base_policy, s):
    ''' Rollout for all agents '''

    base_policy_actions = base_policy(map, sense_info, base_policy, s)

    actions = empty_actions(map.agents_info.n_agents)
    for i, (c, _, mem, _) in enumerate(sense_info):
        actions[i, :] = rollout_egocentric(mem, map.agents_info, base_policy_actions, base_policy, c, s)
    return actions

def rollout_egocentric(mem, perfect_a_info, base_policy_actions, base_policy, agent_code, s):
    ''' Egocentric 1-step lookahead with truncated rollout
        Requires s to define a base policy '''
    a_info = mem.map.agents_info
    values = np.zeros(s.action_space.shape[0])

    # TODO: Separate into agents in our memory only
    in_map_mask    = perfect_a_info.codes
    future_actions = base_policy_actions 

    for j, action in enumerate(s.action_space):
        i = np.argmax(a_info.codes == agent_code)
        
        future_actions = empty_actions(a_info.n_agents)
        # TODO Calculate base policies
        future_actions[i, :] = action

        next_map  = mem.map.clone()
        transition(next_map, future_actions, s) # Modifies next_map

        results = simulate(next_map, base_policy, base_policy, s.truncated_timesteps, s, check_goals=False)
        values[j] = results['score']

    return s.action_space[np.argmax(values)]
