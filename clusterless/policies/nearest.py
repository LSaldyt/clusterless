import numpy as np
from collections import deque
from copy import deepcopy

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
    # print(obstacles)
    illegal = coordinates[obstacles.reshape((np.prod((s.size,s.size)),))]
    illegal = {tuple(illegal[i,:]) for i in range(illegal.shape[0])}
    d.append((coords,[]))
    choices = {tuple(choices[i,:]) for i in range(choices.shape[0])}
    while len(d)>0:
        cell, path = d.popleft()
        path = deepcopy(path)
        path.append(cell)
        # print(f"popping {cell} with path {path}")
        cell_tuple = tuple(cell)
        visited.add(cell_tuple)
        if cell_tuple in choices:
            print(f"going towards {cell_tuple}")
            print(f"p0: {path[0]}, p1: {path[1]}")
            print(f"move: {path[1]-path[0]}")
            return path[1]-path[0]
        neighbors = (cell + action_space) % s.size
        # print(f"neighbors: {neighbors}")
        for neighbor in neighbors:
            # neighbor = neighbor %s.size
            # print(neighbor)
            n = tuple(neighbor)
            # n = (n[0]%s.size,n[1]%s.size)
            if n not in visited and n not in illegal:
                d.append((neighbor, path)) 
    return np.array([0,0],dtype=np.int32)