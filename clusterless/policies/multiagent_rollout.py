import numpy as np

from .rollout import empty_actions, egocentric_rollout

def multiagent_rollout(map, sense_info, base_policy, t, s):
    ''' Rollout for all agents, each getting to do one-step lookahead '''
    # First calculate base policy for all agents
    given_actions = base_policy(map, sense_info, base_policy, t, s)
    for i, (c, mem, _) in enumerate(sense_info):
        given_actions[i, :] = egocentric_rollout(mem, map.agents_info, 
                                                 given_actions, base_policy, c, t, s)
    return given_actions

