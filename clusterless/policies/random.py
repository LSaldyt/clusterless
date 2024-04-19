
def random(map, sense_info, base_policy, t, s):
    a_info         = map.agents_info
    action_indices = s.gen.integers(low=0, high=s.action_space.shape[0], size=(len(sense_info),)) 
    actions        = s.action_space[action_indices]
    return actions
