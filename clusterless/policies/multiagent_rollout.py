from .rollout import egocentric_rollout

def multiagent_rollout(p, s, do_render=False, mask_unseen=True):
    ''' Rollout for all agents, each getting to do one-step lookahead '''
    codes  = [sense.code for sense in p.sense_info]
    # First calculate base policy for all agents
    given_actions = p.base_policy(p, s)
    assert not (given_actions == 0).all(), f'Base policy returned all stay actions, reject!!'
    for i, sense in enumerate(p.sense_info):
        given_actions[i, :] = egocentric_rollout(map, sense.memory, codes, p.memory,
                                                 given_actions, p.base_policy, sense.code, p.t, s, mask_unseen=mask_unseen,
                                                 reveal=False)[0]
    return given_actions
