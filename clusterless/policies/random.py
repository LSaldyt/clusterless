
def random(s, n_agents, sense_info):
    action_indices = s.gen.integers(low=0, high=s.action_space.shape[0], size=(n_agents,)) 
    actions        = s.action_space[action_indices]
    return actions
