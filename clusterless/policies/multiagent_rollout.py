from .rollout import egocentric_rollout

def multiagent_rollout(map, sense_info, memory, base_policy, t, s, do_render=False, mask_unseen=False):
    ''' Rollout for all agents, each getting to do one-step lookahead '''
    codes  = [sense.code for sense in sense_info]
    # First calculate base policy for all agents
    given_actions = base_policy(map, sense_info, memory, base_policy, t, s)
    assert not (given_actions == 0).all(), f'Base policy returned all stay actions, reject!!'
    for i, sense in enumerate(sense_info):
        given_actions[i, :] = egocentric_rollout(sense.memory, codes, memory,
                                                 given_actions, base_policy, sense.code, t, s, mask_unseen=mask_unseen)
    return given_actions
