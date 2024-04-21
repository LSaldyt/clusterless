
def random(p, s):
    action_indices = s.gen.integers(low=0, high=s.action_space.shape[0], size=(len(p.sense_info),)) 
    actions        = s.action_space[action_indices]
    return actions
