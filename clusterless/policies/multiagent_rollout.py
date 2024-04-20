import numpy as np

from .rollout import empty_actions, egocentric_rollout

def multiagent_rollout(map, sense_info, memory, base_policy, t, s, do_render=False):
    ''' Rollout for all agents, each getting to do one-step lookahead '''
    codes = [sense.code for sense in sense_info]
    # First calculate base policy for all agents
    given_actions = base_policy(map, sense_info, memory, base_policy, t, s)
    for i, sense in enumerate(sense_info):
        given_actions[i, :] = egocentric_rollout(sense.memory, codes,
                                                 given_actions, base_policy, sense.code, t, s)
    return given_actions
