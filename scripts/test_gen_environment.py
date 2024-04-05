import numpy as np

from clusterless.environment import create_grid, render, transition

def cartesian_product(*arrays):
    ''' Convenience. From https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points'''
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def run(*args):
    size  = 10
    gen   = np.random.default_rng(2024)
    probs = dict(empty=0.5, obstacle=0.35, goal=0.1, agent=0.05) # Order matters! Agents come last
    codes = {k : i for i, k in enumerate(probs.keys())}
    codes['dead'] = -1

    assert sum(probs.values()) < 1.0 + 1e-6

    grid_shape = (size, size)
    grid = create_grid(gen, grid_shape, probs)
    print(render(grid))

    action_space   = np.array([[0, 0], [0, 1], [1, 0], [-1, 0], [0, -1]])
    coordinates    = cartesian_product(np.arange(size), np.arange(size))

    for i in range(1000):
        agent_indices  = grid >= codes['agent']
        agent_codes    = grid[agent_indices]
        agent_coords   = coordinates[agent_indices.reshape(size * size,)]

        n_agents = np.sum(agent_indices)

        action_indices = gen.integers(low=0, high=action_space.shape[0], size=(n_agents,)) # type: ignore (Silence! My code is right!)
        actions        = action_space[action_indices]

        info = transition(grid, actions, agent_coords, agent_codes, codes)
        print(f'Step {i}: ' + ('goals acheived: {n_goals_achieved}, '
                               'collisions: {n_collisions_obstacle} (obstacle) {n_collisions_agents} (agent)'
                   ).format(**info))
        print(render(grid))
