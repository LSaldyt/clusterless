import string
import numpy as np
from clusterless.environment import create_grid, render, transition, views

def cartesian_product(*arrays):
    ''' Convenience. From https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points'''
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def run(*args):
    ''' First three symbols are reserved (empty, obstacle, goal)
        Last two symbols are reserved (unseen, dead agent)
        All symbols inbetween are used for agent codes '''
    symbols = '·□★' + 'ζξΞѯƔȣ☭' + '♔♕♖♗♘♙♚♛♜♝♞♟' + string.ascii_lowercase + string.ascii_uppercase + '☺☆?☠'
    size    = 8
    gen     = np.random.default_rng(2024)
    probs   = dict(empty=0.5, obstacle=0.35, goal=0.1, agent=0.05) # Order matters! Agents come last
    codes   = {k : i for i, k in enumerate(probs.keys())}
    codes['dead']      = -1
    codes['unseen']    = -2
    codes['old_goal']  = -3
    codes['old_agent'] = -4

    assert sum(probs.values()) < 1.0 + 1e-6

    grid = create_grid(gen, (size, size), probs)

    action_space   = np.array([[0, 0], [0, 1], [1, 0], [-1, 0], [0, -1]])
    coordinates    = cartesian_product(np.arange(size), np.arange(size))

    memory = dict() # Memory tensors for all agents

    for t in range(128):
        agent_mask     = grid >= codes['agent']
        agent_codes    = grid[agent_mask]
        agent_coords   = coordinates[agent_mask.reshape(size * size,)]

        n_agents       = np.sum(agent_mask)
        
        if not memory: # Memory is cheap. Fill it :)
            memory = {k : np.full(grid.shape, codes['unseen']) # type: ignore
                      for k in agent_codes}

        print(render(grid, symbols))
        for c, (view_coords, view) in zip(agent_codes, views(grid, agent_coords)): # Important: views should be taken before transitions for consistency
            mem = memory[c]
            print(f'View of agent {symbols[c]}')
            print(render(view, symbols))
            was_agent_mask = mem >= codes['agent']
            was_goal_mask  = mem == codes['goal']
            mem[was_agent_mask] = codes['old_agent'] # Old agent locations (uniform for now)
            mem[was_goal_mask]  = codes['old_goal']  # Old goals are inverted
            mem[view_coords[:, 0], view_coords[:, 1]] = view.ravel() # New information (overwrites)
            print(f'Memory of agent {symbols[c]}')
            print(render(mem, symbols))

        action_indices = gen.integers(low=0, high=action_space.shape[0], size=(n_agents,)) # type: ignore (Silence! My code is right!)
        actions        = action_space[action_indices]

        info = transition(grid, actions, agent_coords, agent_codes, codes) # Important: Do transition at the end of the loop
        print(f'Step {t}: ' + ('goals acheived: {n_goals_achieved}, '
                               'collisions: {n_collisions_obstacle} (obstacle) {n_collisions_agents} (agent)'
                   ).format(**info))
        print('-' * 80)
