from ..environment import transition

import numpy as np

empty_actions = lambda n : np.zeros(shape=(n, 2), dtype=np.int32)

def rollout(map, sense_info, base_policy, s):
    ''' Rollout for all agents '''
    actions = empty_actions(map.agents_info.n_agents)
    for i, (_, _, mem, _) in enumerate(sense_info):
        actions[i, :] = rollout_egocentric(mem, i, base_policy, s)
    return actions

def rollout_egocentric(mem, i, base_policy, s):
    ''' Egocentric 1-step lookahead with truncated rollout
        Requires s to define a base policy '''
    print(mem, base_policy)
    a_info = mem.map.agents_info
    for action in s.action_space:
        future_actions       = empty_actions(a_info.n_agents)
        future_actions[i, :] = action

        transition(mem.map, future_actions, s)
    exit()
