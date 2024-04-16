import numpy as np
from collections import deque
from functools import reduce

action_space = np.array([[0, 1], [1, 0], [-1, 0], [0, -1]])


# def alpha_nearest(s, n_agents, sense_info, coordinates):
def alpha_nearest(map, sense_info, base_policy, s):
    ''' This policy tries to balance exploration and exploitation with a tunable parameter alpha.
        Ideal: take the shortest path to the cell v which minimizes -log P(G(v)) + alpha*D(v)
            P(G(v)) is the probability that v is a goal. We write I(v) := -log P(G(v))
            MD(v) is the manhattan distance of the shortest path to v from the current cell
        In short, we prefer to go to v rather than to a closer w if alpha times the extra number
            of steps is less than the number of extra bits of surprisal w having a goal would have.
            alpha*(D(v)-D(w))<I(w)-I(v)
        First Order Simplifications:
            We assume P(G(v)) is either 1 (goal seen at cell v) or the goal generation probability aka s.probs['goal']'''
    a_info = map.agents_info
    actions = np.zeros(shape=(a_info.n_agents,2),dtype=np.int32)
    for i, (c, view, mem, coords) in enumerate(sense_info):
        goals = mem.map.grid == s.codes['goal']
        goals = goals.astype(float)
        unexplored = mem.map.grid == s.codes['unseen']
        unexplored = s.probs['goal']* unexplored.astype(float)
        possible_targets = (unexplored+goals).reshape((np.prod((s.size,s.size))),)
        # print(f"checking for {c}")
        move = shortest_path_alpha(s, possible_targets, coords, map.coords, mem)
        actions[i,:]=move
    return actions

def shortest_path_alpha(s, probabilities, coords, coordinates, mem):
    alpha = s.alpha
    d = deque()
    visited = set()
    obstacles = np.logical_or(mem.map.grid == s.codes['obstacle'],mem.map.grid == s.codes['dead'])
    illegal = coordinates[obstacles.reshape((np.prod((s.size,s.size)),))]
    illegal = {tuple(illegal[i,:]) for i in range(illegal.shape[0])}
    possible_first_moves = {}
    best = [np.array([0,0],dtype=np.int32), np.inf,np.inf,np.inf]
    for action in action_space:
        poss_first_move = (coords+action)%s.size
        if tuple(poss_first_move) not in illegal:
            d.append((poss_first_move,poss_first_move,1))
            possible_first_moves[tuple(poss_first_move)] = 1
    visited.add(tuple(coords))
    choices = {tuple(coordinates[i,:]): probabilities[i] for i in range(coordinates.shape[0])}
    max_probability = reduce(lambda x,v: max(x,v), choices.values(), 0)
    while len(d)>0:
        cell, first_cell, path_length = d.popleft()
        path_length+=1
        possible_first_moves[tuple(first_cell)]-=1
        cell_tuple = tuple(cell)
        if cell_tuple in visited: continue
        first_cell_tuple = tuple(first_cell)
        probability = choices[cell_tuple]
        if probability > 0:
            surprisal = -np.log10(probability)
            heuristic = surprisal+alpha*path_length
            if heuristic < best[1]:
                proposed_move = first_cell-coords
                best = [np.where(proposed_move == 1-s.size, 1, (np.where(proposed_move == s.size-1, -1, proposed_move))),heuristic,path_length, surprisal]
                # quick quit if there will be no higher prob later 
                if probability >= max_probability: break
        # if we know we can never possibly find anything better than the current v, stop and return v's first move
        # we know this when alpha*current extra search distance is equal or greater to I(v)
        if alpha*(path_length-best[2])>=best[3]:
            break
        neighbors = (cell + action_space) % s.size
        for neighbor in neighbors:
            n = tuple(neighbor)
            if n not in visited and n not in illegal:
                d.append((neighbor, first_cell, path_length))
                possible_first_moves[first_cell_tuple]+=1
                visited.add(cell_tuple)
        if possible_first_moves[first_cell_tuple] == 0: 
            del possible_first_moves[first_cell_tuple]
        if len(possible_first_moves) == 1:
            proposed_move = list(possible_first_moves.keys())[0]-coords 
            return np.where(proposed_move == 1-s.size, 1, (np.where(proposed_move == s.size-1, -1, proposed_move)))
    return best[0]
