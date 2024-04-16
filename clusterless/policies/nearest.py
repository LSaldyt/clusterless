import numpy as np
from collections import deque

action_space = np.array([[0, 1], [1, 0], [-1, 0], [0, -1]])

def nearest(map, sense_info, base_policy, s):
    a_info  = map.agents_info
    actions = np.zeros(shape=(a_info.n_agents, 2), dtype=np.int32)
    for i, (c, mem, coords) in enumerate(sense_info):
        targets       = mem.map.mask('goal', 'unseen')
        target_coords = mem.map.coords_of(targets)
        move          = shortest_path(s, target_coords, coords, mem)
        actions[i,:]  = move
    return actions

def shortest_path(s, choices, coords, mem):
    d = deque()
    visited = set() 
    obstacles = mem.map.mask('obstacle', 'dead')
    illegal   = mem.map.coords_of(obstacles)
    illegal = {tuple(illegal[i,:]) for i in range(illegal.shape[0])}
    possible_first_moves = {}
    for action in action_space:
        poss_first_move = (coords+action)%s.size
        if tuple(poss_first_move) not in illegal:
            d.append((poss_first_move,poss_first_move))
            possible_first_moves[tuple(poss_first_move)] = 1
    visited.add(tuple(coords))
    choices = {tuple(choices[i,:]) for i in range(choices.shape[0])}
    while len(d)>0:
        cell, first_cell = d.popleft()
        possible_first_moves[tuple(first_cell)]-=1
        cell_tuple = tuple(cell)
        first_cell_tuple = tuple(first_cell)
        visited.add(cell_tuple)
        if cell_tuple in choices:
            # print(f"going towards {cell_tuple}")
            # print(f"p0: {coords}, p1: {first_cell}")
            # print(f"move: {first_cell-coords}")
            # print(f"og set: {og_first_moves}")
            # print(d)
            proposed_move = first_cell-coords
            return np.where(proposed_move == 1-s.size, 1, (np.where(proposed_move == s.size-1, -1, proposed_move)))
        neighbors = (cell + action_space) % s.size
        for neighbor in neighbors:
            n = tuple(neighbor)
            if n not in visited and n not in illegal:
                d.append((neighbor, first_cell))
                possible_first_moves[first_cell_tuple]+=1
        if possible_first_moves[first_cell_tuple] == 0: 
            del possible_first_moves[first_cell_tuple]
        if len(possible_first_moves) == 1:
            proposed_move = list(possible_first_moves.keys())[0]-coords 
            return np.where(proposed_move == 1-s.size, 1, (np.where(proposed_move == s.size-1, -1, proposed_move)))
    return np.array([0,0],dtype=np.int32)
