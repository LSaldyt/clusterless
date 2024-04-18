from .map import Map

import numpy as np

def emplace(memory, world, s):
    ''' Emplace a random world in the periphery of an agent's world 
    We simply replace unseen tiles with world tiles, and reveal them as-necessary
    However we need to reconcile existing agents in our world with the imagined world '''
    # First, reconcile agents between observed map and imagined world
    seen = memory.map.agents_info # Agents in the active memory
    new  = world.agents_info

    duplicate_mask   = np.array([c in seen.codes for c in new.codes], dtype=np.int32)
    if duplicate_mask.shape[0] != 0:
        duplicate_coords = new.coords[duplicate_mask]
        unique_codes     = new.codes[duplicate_mask] + np.max(new.codes)
        world_filtered   = world.clone()
        world_filtered.set_at(duplicate_coords, unique_codes)
    else:
        world_filtered = world

    # Then, replace periphery with the imagined world tiles
    new_grid = np.where(memory.map.grid == s.codes['unseen'],
                        world_filtered.grid, memory.map.grid)
    return Map(s, new_grid)

def generate_worlds(s):
    ''' Simply generate n_worlds random worlds using probabilities in settings '''
    for _ in range(s.n_worlds):
        yield Map(s)
