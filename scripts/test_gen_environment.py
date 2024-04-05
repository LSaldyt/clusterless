import numpy as np
import string

chess = '♔♕♖♗♘♙♚♛♜♝♞♟'
def render(grid, syms='·□★' + 'ζξΞѯƔȣ☭' + chess + string.ascii_lowercase + string.ascii_uppercase + '☠'):
    ''' Render the environment in questionable unicode '''
    h, w = grid.shape
    def gen_syms():
        for i in range(h):
            for j in range(w):
                yield syms[grid[i, j]]
            yield '\n'
    return ''.join(gen_syms())

def create_grid(gen, grid_shape, probs, agent_index=3):
    ''' Create an initial environment from a multinomial distribution '''
    grid = gen.choice(np.arange(len(probs)), size=grid_shape, p=list(probs.values()))
    # Re-assign agents to unique numbers, e.g. [3, 3, 3] becomes [3, 4, 5]
    mask = grid == agent_index
    grid[mask] = (np.arange(np.sum(mask)) + agent_index)
    return grid

def set_at(grid, coords, values):
    ''' Set grid by vectorized coordinates to new values '''
    grid[coords[:, 0], coords[:, 1]] = values

def views():
    ''' Produce local views for all agents '''
    pass

def transition(grid, actions, agent_coords, agent_codes, codes):
    ''' Move all agents at once according to their actions:
        Agent actions can be stay, up, down, left, and right
        If an agent touches a goal, that goal is achieved. 
        Agents cannot collide.. or they die?
        Agents cannot step into obstacles.. or they die? '''
    
    next_coords    = agent_coords + actions
    next_coords    = np.minimum(np.maximum(next_coords, 0), 9)

    next_locations = grid[next_coords[:, 0], next_coords[:, 1]]
    allowed_move   = next_locations != codes['obstacle']
    final_coords   = np.where(np.repeat(np.expand_dims(allowed_move, -1), 2, -1), 
                              next_coords, agent_coords)
    next_locations = grid[final_coords[:, 0], final_coords[:, 1]]
    set_at(grid, agent_coords, codes['empty'])
    set_at(grid, final_coords, agent_codes)
    unique_coords, unique_counts = np.unique(final_coords, axis=0, return_counts=True)
    collision_mask   = unique_counts > 1
    collision_coords = unique_coords[collision_mask]

    if (collision_mask).any():
        set_at(grid, collision_coords, codes['dead'])
        coll_coord_mask   = np.all(final_coords == collision_coords, axis=-1)
        reached_locations = next_locations[coll_coord_mask]
    else:
        reached_locations = next_locations

    goal_mask = reached_locations == codes['goal']

    n_goals    = np.sum(goal_mask)
    n_collides = np.sum(unique_counts) - unique_counts.shape[0]


    return n_goals, n_collides

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

        n_goals, n_collides = transition(grid, actions, agent_coords, agent_codes, codes)
        print(f'Step {i}: goals acheived {n_goals}, collisions {n_collides}')
        print(render(grid))

