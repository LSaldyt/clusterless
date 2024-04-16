import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

from . import utils
from .map import Map

@dataclass
class Memory():
    map  : Map
    time : npt.ArrayLike

''' Memory is a series of views over time 
    Agents will move, goals may be taken, 
    otherwise empty/obstacle are certain, 
    agents/goals in current time step are certain '''

def views(map, s):
    ''' Produce local views for all agents.
        A view is a local n x n map around an agent.
        View that go beyond the border of the map are padded. '''
    a_info       = map.agents_info
    view_fn      = getattr(utils, s.view_type)
    view_offsets = view_fn(s.view_size)
    for i in range(a_info.n_agents): # Seemed intuitive, faster and more memory efficient than a vectorized operation
        view_coords    = view_offsets + a_info.coords[i, :]
        view_coords    = view_coords % map.grid.shape[0]
        view           = map.grid[view_coords[:, 0], view_coords[:, 1]]
        yield a_info.coords[i, :], view_coords, view

def init_memory(map, s):
    return {k : Memory(Map(s, np.full(map.grid.shape, s.codes['unseen'])), 
                       np.full(map.grid.shape, 0)) 
            for k in map.agents_info.codes}

def sense_environment(map, memory, s, timestep):
    a_info = map.agents_info
    for c, (ac, view_coords, view) in zip(a_info.codes, views(map, s)): 
        memory[c].map.grid[view_coords[:, 0], view_coords[:, 1]] = view
        memory[c].time[    view_coords[:, 0], view_coords[:, 1]] = timestep
        yield c, memory[c], ac

def merge_memory(mem_a, mem_b):
    recent   = mem_a.time < mem_b.time
    new_grid = np.where(recent, mem_a.map.grid, mem_b.map.grid)
    new_time = np.where(recent, mem_a.time,     mem_b.time)
    return Memory(Map(new_grid), new_time)
