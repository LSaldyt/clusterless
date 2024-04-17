import numpy as np
from collections import deque
from ..map import render

action_space = np.array([[0, 1], [1, 0], [-1, 0], [0, -1]])

def alt_nearest(map, sense_info, base_policy, t, s):
    a_info  = map.agents_info
    actions = np.zeros(shape=(a_info.n_agents, 2), dtype=np.int32)
    for i, (c, mem, ac) in enumerate(sense_info):
        actions[i, :] = alt_nearest_single_agent(mem, ac, s)
    return actions

def alt_nearest_single_agent(mem, ac, s):
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
    while np.sum(working == 1) > 0:
        no_possible = np.zeros(4)
        if not do: # Only on subsequent iterations
            print('Getting new target coords from ')
            print(target_coords)
            print((working > 0) & (working < 6))
            target_coords = mem.map.coords_of((working > 1) & (working < 6))
            print(target_coords)
        show(working)
        for act_i in np.arange(2, 5+1):
            print(s.action_words[act_i - 2])
            action      = s.action_space[act_i - 2]
            next_coords = ((target_coords - action) % s.size)
            existing = working[next_coords[:, 0], next_coords[:, 1]]
            working[next_coords[:, 0], next_coords[:, 1]] = np.where(existing == 1, existing * act_i, existing) 
            no_possible[act_i - 2] = (existing == 0).all()
            show(working)
        show(working)
        print(no_possible)
        if not do:
            break
        if (no_possible == 0).all():
            break
        do = False
    exit()
    1/0
