import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from collections import namedtuple

from . import utils
from .utils import at_xy, mask
from .map import Map

@dataclass
class Memory():
    map  : Map
    time : npt.ArrayLike

AgentSense = namedtuple('AgentSense', ['memory', 'view', 'code', 'xy'])

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

def map_for_simulate(mem, s, duplicates_only=False):
    ''' Remove old agents when maps are being used for simulation. Not compatible with beliefs '''
    # print('BEFORE')
    # mem.map.color_render()
    clone  = mem.map.clone()
    latest = np.max(mem.time)
    agents = clone.grid >= s.codes['agent']
    if duplicates_only:
        for agent_code in mem.map.agents_info.codes:
            copy_mask    = clone.grid == agent_code
            agent_latest = np.max(mem.time[copy_mask])
            clone.grid[copy_mask] = np.where(mem.time[copy_mask] == agent_latest, agent_code, s.codes['empty'])
    else:
        clone.grid = np.where(agents, np.where(mem.time == latest, clone.grid, s.codes['empty']), clone.grid)
    clone.set_at(clone.at('unseen'), s.codes['obstacle'])
    clone._inc_purity()
    # print('AFTER')
    # clone.color_render()
    return clone

def sense_environment(env_map, memory, s, timestep):
    a_info = env_map.agents_info
    for c, (xy, view_coords, view) in zip(a_info.codes, views(env_map, s)): 
        memory[c].map.grid[*at_xy(view_coords)] = view
        memory[c].time[    *at_xy(view_coords)] = timestep
        yield AgentSense(memory[c], view, c, xy)

def merge_memory(mem_a, mem_b, s):
    recent   = mem_a.time > mem_b.time
    new_grid = np.where(recent, mem_a.map.grid, mem_b.map.grid)
    new_time = np.where(recent, mem_a.time,     mem_b.time)
    return Memory(Map(s, new_grid), new_time)

def get_neighbors(sense, s):
    neighbor_mask = ((sense.view >= s.codes['agent']) & (sense.view != sense.code))
    neighbors     = sense.view[neighbor_mask]
    return neighbors

def update_memories(senses, memory, s):
    for sense in senses:
        new_memory = sense.memory
        for n in get_neighbors(sense, s):
            new_memory = merge_memory(new_memory, memory[n], s)
        yield sense.code, new_memory

def communicate(memory, senses, s):
    for c, mem in list(update_memories(senses, memory, s)):
        memory[c] = mem
