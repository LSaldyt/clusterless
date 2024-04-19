import numpy as np
from collections import deque
from ..map   import render
from ..utils import UnsolvableException, at_xy, broadcast

from .utils import empty_actions

def wave(map, sense_info, base_policy, t, s):
    actions = empty_actions(len(sense_info))
    for i, sense in enumerate(sense_info):
        actions[i, :] = wave_egocentric(sense.memory, sense.xy, s)
    if (actions == 0).all(): # Absolutely no achievable goals or unexplored regions
        raise UnsolvableException('Wavefront algorithm definitively found no achievable goals for any agent (promise..)')
    return actions

def wave_egocentric(mem, xy, s):
    targets   = mem.map.mask('goal', 'unseen')
    obstacles = mem.map.mask('obstacle', 'dead', 'unseen')

    def show(w):
        if s.debug_alt_nearest:
            print(render(working, '□.↑↓←→∙'))
            print(np.sum(working == 1))

    target_coords   = mem.map.coords_of(targets)
    obstacle_coords = mem.map.coords_of(obstacles)

    # obstacle: 0 | up, down, left, right : 2, 3, 4, 5 | goal : 6
    working = np.ones_like(mem.map.grid, dtype=np.int32)
    working[*at_xy(obstacle_coords)] = 0
    working[*at_xy(target_coords)]   = 6 
    last = np.copy(working)
    while np.sum(working == 1) > 0:
        # show(working)
        for act_i in np.arange(2, 5+1):
            # print(s.action_words[act_i - 2])
            action      = s.action_space[act_i - 2]
            next_coords = ((target_coords - action) % s.size)
            existing = working[*at_xy(next_coords)]
            working[*at_xy(next_coords)] = np.where(existing == 1, existing * act_i, existing) 
        # show(working)
        if (working == last).all(): # There are other ways to check the stopping condition but this was robust
            break
        last = np.copy(working)
        target_coords = mem.map.coords_of((working > 1) & (working < 6))
    # show(working)
    act_i = working[xy[0], xy[1]]
    return s.action_space[act_i - 2]
