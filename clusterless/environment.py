import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

@dataclass
class Memory():
    grid : npt.ArrayLike
    time : npt.ArrayLike

from . import utils
def create_grid(gen, grid_shape, probs, agent_index=3):
    ''' Create an initial environment from a multinomial distribution '''
    grid = gen.choice(np.arange(len(probs)), size=grid_shape, p=list(probs.values()))
    # Re-assign agents to unique numbers, e.g. [3, 3, 3] becomes [3, 4, 5]
    mask = grid == agent_index
    size = grid_shape[0]

    grid[mask] = (np.arange(np.sum(mask)) + agent_index)
    coordinates = utils.cartesian_product(np.arange(size), np.arange(size))
    return grid, coordinates

def render(grid, syms):
    ''' Render the environment in questionable unicode '''
    h, w = grid.shape
    def gen_syms():
        for i in range(h):
            for j in range(w):
                yield syms[grid[i, j]]
            yield '\n'
    return ''.join(gen_syms())

def full_render(grid, sense_input, s):
    rendered_views = [(' ' * s.view_size + '\n') * s.view_size]
    rendered_grids = [render(grid, s.symbols)]
    codes          = []
    for c, view, mem, coords in sense_input:
        rendered_views.append(render(view,     s.symbols))
        rendered_grids.append(render(mem.grid, s.symbols))
        codes.append(f'agent {s.symbols[c]}')
    descriptions   = [f'{name:<{s.size}}\n' for name in ['full'] + codes]
    print(utils.horizontal_join(rendered_grids))
    print(utils.horizontal_join(descriptions))
    print(utils.horizontal_join(rendered_views, join=' ' * (s.size - s.view_size + 1)))

def set_at(grid, coords, values):
    ''' Set grid by vectorized coordinates to new values '''
    grid[coords[:, 0], coords[:, 1]] = values

def transition(grid, actions, agent_coords, agent_codes, codes):
    ''' Note: This function MODIFIES grid intentionally, for efficiency.

        Move all agents at once according to their actions:
        Agent actions can be stay, up, down, left, and right
        If an agent touches a goal, that goal is achieved. 
        Agents cannot step into obstacles.. or they stay in place at a penalty 
        Agents cannot collide.. or they die? ☠'''
    # Progress actions, (optionally) enforce grid boundaries
    next_coords    = (agent_coords + actions) 
    next_coords    = next_coords % (grid.shape[0])
    next_locations = grid[next_coords[:, 0], next_coords[:, 1]]
    # Enforce obstacles (and dead agents as obstacles)
    allowed_move   = ((next_locations != codes['obstacle']) & (next_locations != codes['dead']))
    final_coords   = np.where(utils.broadcast(allowed_move, 2), next_coords, agent_coords)
    # Move agents to (filtered) locations
    set_at(grid, agent_coords, codes['empty'])
    set_at(grid, final_coords, agent_codes)
    # Check for inter-agent collisions
    unique_coords, unique_counts = np.unique(final_coords, axis=0, return_counts=True)
    collision_mask   = unique_counts > 1
    collision_coords = unique_coords[collision_mask]
    if (collision_mask).any():
        set_at(grid, collision_coords, codes['dead'])
    non_collision_coords = unique_coords[collision_mask == False]
    reached_locations    = grid[non_collision_coords[:, 0], non_collision_coords[:, 1]]

    # Count goals and collision types
    goal_mask = reached_locations == codes['goal']
    return dict(
        n_goals_achieved      = np.sum(goal_mask),
        n_collisions_obstacle = np.sum(1 - allowed_move),
        n_collisions_agents   = np.sum(unique_counts) - unique_counts.shape[0],
    )

def views(grid, agent_coords, s):
    ''' Produce local views for all agents.
        A view is a local n x n grid around an agent.
        View that go beyond the border of the grid are padded. '''
    n_agents = agent_coords.shape[0]
    view_box = utils.box(s.view_size)
    for i in range(n_agents): # Seemed intuitive, faster and more memory efficient than a vectorized operation
        view_coords    = view_box + agent_coords[i, :]
        view_coords    = view_coords % grid.shape[0]
        view           = grid[view_coords[:, 0], view_coords[:, 1]]
        yield agent_coords[i, :], view_coords, view.reshape((s.view_size, s.view_size))

def get_agents(grid, coordinates, codes):
    mask     = grid >= codes['agent']
    codes    = grid[mask]
    coords   = coordinates[mask.reshape((np.prod(grid.shape),))]
    n_agents = np.sum(mask)
    return codes, coords, n_agents

''' Memory is a series of views over time 
    Agents will move, goals may be taken, 
    otherwise empty/obstacle are certain, 
    agents/goals in current time step are certain '''

def init_memory(grid, agent_codes, codes):
    return {k : Memory(np.full(grid.shape, codes['unseen']), 
                       np.full(grid.shape, 0)) 
            for k in agent_codes}

def sense_environment(grid, memory, agent_codes, agent_coords, s, timestep):
    for c, (ac, view_coords, view) in zip(agent_codes, views(grid, agent_coords, s)): # Important: views should be taken before transitions for consistency
        memory[c].grid[view_coords[:, 0], view_coords[:, 1]] = view.ravel() 
        memory[c].time[view_coords[:, 0], view_coords[:, 1]] = timestep
        yield c, view, memory[c], ac
