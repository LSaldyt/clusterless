# from ..policies import available_policies

import numpy as np

def rollout(s, n_agents, sense_info, coordinates):
    ''' Egocentric 1-step lookahead with truncated rollout
        Requires s to define a base policy '''

    # base_policy = available_policies[s.base_policy]
    1/0

    maps = np.zeros((n_agents, s.size, s.size))
    s.action_space
