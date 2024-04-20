import numpy as np

from clusterless.utils import at_xy

def brownian(map, sense_info, memory, base_policy, t, s):
    ''' Semi-intelligent brownian motion that doesn't move into obstacles '''
    a_info         = map.agents_info
    obs_mask       = np.full((a_info.n_agents,), True)
    actions        = np.zeros_like(a_info.coords)
    while obs_mask.any(): 
        act_inds          = s.gen.integers(low=0, high=s.action_space.shape[0], size=(a_info.n_agents,)) 
        actions[obs_mask] = s.action_space[act_inds][obs_mask] # Only replace invalid actions
        obs_mask          = map.grid[*at_xy((actions + a_info.coords) % s.size)] == s.codes['obstacle']
    return actions
