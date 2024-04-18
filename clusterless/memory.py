import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

from . import utils
from .utils import at_xy
from .map import Map

@dataclass
class Memory():
    map  : Map
    time : npt.ArrayLike

''' Memory is a series of views over time 
    Agents will move, goals may be taken, 
    otherwise empty/obstacle are certain, 
    agents/goals in current time step are certain '''

def views(env_map, s):
    ''' Produce local views for all agents.
        A view is a local n x n map around an agent.
        View that go beyond the border of the map are padded. '''
    a_info       = env_map.agents_info
    view_fn      = getattr(utils, s.view_type)
    view_offsets = view_fn(s.view_size)
    for i in range(a_info.n_agents): # Seemed intuitive, faster and more memory efficient than a vectorized operation
        view_coords    = view_offsets + a_info.coords[i, :]
        view_coords    = view_coords % env_map.grid.shape[0]
        view           = env_map.grid[view_coords[:, 0], view_coords[:, 1]]
        yield a_info.coords[i, :], view_coords, view

def init_memory(env_map, s):
    return {k : Memory(Map(s, np.full(env_map.grid.shape, s.codes['unseen'])), 
                       np.full(env_map.grid.shape, 0)) 
            for k in env_map.agents_info.codes}

def sense_environment(env_map, memory, s, timestep):
    a_info = env_map.agents_info
    for c, (ac, view_coords, view) in zip(a_info.codes, views(env_map, s)): 
        memory[c].map.grid[*at_xy(view_coords)] = view
        memory[c].time[    *at_xy(view_coords)] = timestep
        yield c, memory[c], ac

def merge_memory(mem_a, mem_b):
    recent   = mem_a.time < mem_b.time
    new_grid = np.where(recent, mem_a.map.grid, mem_b.map.grid)
    new_time = np.where(recent, mem_a.time,     mem_b.time)
    return Memory(Map(new_grid), new_time)
