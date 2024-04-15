import numpy as np
from collections import deque

action_space = np.array([[0, 1], [1, 0], [-1, 0], [0, -1]])

def nearest(s, n_agents, sense_info, coordinates):
    actions = np.zeros(shape=(n_agents,2),dtype=np.int32)
    for i, (c, view, mem, coords) in enumerate(sense_info):
        goals = mem.grid == s.codes['goal']
        unexplored = mem.grid == s.codes['unseen']
        possible_targets = goals | unexplored
        possible_coordinates = coordinates[possible_targets.reshape((np.prod((s.size,s.size)),))]
        move = shortest_path(s, possible_coordinates, coords, coordinates, mem)
        actions[i,:]=move
    return actions

def shortest_path(s, choices, coords, coordinates, mem):
    d = deque()
    visited = set() 
    obstacles = np.logical_or(mem.grid == s.codes['obstacle'],mem.grid == s.codes['dead'])
    illegal = coordinates[obstacles.reshape((np.prod((s.size,s.size)),))]
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
            return first_cell-coords
        neighbors = (cell + action_space) % s.size
        for neighbor in neighbors:
            n = tuple(neighbor)
            if n not in visited and n not in illegal:
                d.append((neighbor, first_cell))
                possible_first_moves[first_cell_tuple]+=1
        if possible_first_moves[first_cell_tuple] == 0: 
            del possible_first_moves[first_cell_tuple]
        if len(possible_first_moves) == 1:
            return list(possible_first_moves.keys())[0]-coords 
    return np.array([0,0],dtype=np.int32)