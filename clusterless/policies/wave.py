import numpy as np
from collections import deque
from ..map   import render
from ..utils import UnsolvableException

action_space = np.array([[0, 1], [1, 0], [-1, 0], [0, -1]])

def wave(map, sense_info, base_policy, t, s):
    a_info  = map.agents_info
    actions = np.zeros(shape=(a_info.n_agents, 2), dtype=np.int32)
    for i, (c, mem, ac) in enumerate(sense_info):
        actions[i, :] = wave_egocentric(mem, ac, s)
    if (actions == 0).all(): # Absolutely no achievable goals or unexplored regions
        raise UnsolvableException('Wavefront algorithm definitively found no achievable goals for any agent (promise..)')
    return actions

def wave_egocentric(mem, ac, s):
    targets   = mem.map.mask('goal', 'unseen')
    obstacles = mem.map.mask('obstacle', 'dead', 'unseen')
    coords    = mem.map.coords

    def show(w):
        if s.debug_alt_nearest:
            print(render(working, '□.↑↓←→∙'))
            print(np.sum(working == 1))

    target_coords   = mem.map.coords_of(targets)
    obstacle_coords = mem.map.coords_of(obstacles)

    # obstacle: 0
    # up, down, left, right : 2, 3, 4, 5
    working = np.ones_like(mem.map.grid, dtype=np.int32)
    working[obstacle_coords[:, 0], obstacle_coords[:, 1]] = 0
    working[target_coords[:, 0], target_coords[:, 1]]     = 6 # Stay in place policy
    do = True
    last = np.copy(working)
    while np.sum(working == 1) > 0:
        if not do: # Only on subsequent iterations
            target_coords = mem.map.coords_of((working > 1) & (working < 6))
        # show(working)
        for act_i in np.arange(2, 5+1):
            # print(s.action_words[act_i - 2])
            action      = s.action_space[act_i - 2]
            next_coords = ((target_coords - action) % s.size)
            existing = working[next_coords[:, 0], next_coords[:, 1]]
            working[next_coords[:, 0], next_coords[:, 1]] = np.where(existing == 1, existing * act_i, existing) 
            # show(working)
        # show(working)
        if (working == last).all(): # There are other ways to check the stopping condition but this was robust
            break
        do = False
        last = np.copy(working)
    # show(working)
    act_i = working[ac[0], ac[1]]
    return s.action_space[act_i - 2]