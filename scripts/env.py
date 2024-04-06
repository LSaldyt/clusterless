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
    symbols = '·□★' + 'ζξΞѯƔȣ☭' + '♔♕♖♗♘♙♚♛♜♝♞♟' + string.ascii_lowercase + string.ascii_uppercase + '?☠'
    size    = 8
    gen     = np.random.default_rng(2024)
    probs   = dict(empty=0.5, obstacle=0.35, goal=0.1, agent=0.05) # Order matters! Agents come last
    codes   = {k : i for i, k in enumerate(probs.keys())}
    codes['dead']   = -1
    codes['unseen'] = -2

    assert sum(probs.values()) < 1.0 + 1e-6

    grid_shape = (size, size)
    grid = create_grid(gen, grid_shape, probs)

    action_space   = np.array([[0, 0], [0, 1], [1, 0], [-1, 0], [0, -1]])
    coordinates    = cartesian_product(np.arange(size), np.arange(size))

    for i in range(10):
        agent_mask     = grid >= codes['agent']
        agent_codes    = grid[agent_mask]
        agent_coords   = coordinates[agent_mask.reshape(size * size,)]

        n_agents = np.sum(agent_mask)

        action_indices = gen.integers(low=0, high=action_space.shape[0], size=(n_agents,)) # type: ignore (Silence! My code is right!)
        actions        = action_space[action_indices]

        print(render(grid, symbols))
        view_arr = views(grid, agent_coords, codes, wrap=True)
        for j in range(n_agents): # type: ignore
            print(f'View of agent {symbols[agent_codes[j]]}')
            print(render(view_arr[j, :, :], symbols))
        info = transition(grid, actions, agent_coords, agent_codes, codes)
        print(f'Step {i}: ' + ('goals acheived: {n_goals_achieved}, '
                               'collisions: {n_collisions_obstacle} (obstacle) {n_collisions_agents} (agent)'
                   ).format(**info))
        print('-' * 80)
