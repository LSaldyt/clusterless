from experiments.base import defaults

from clusterless.memory   import init_memory, sense_environment, communicate
from clusterless.scenario import from_unicode
from experiments.base     import defaults
from clusterless.environment import transition
from clusterless.utils import broadcast

import numpy as np

world_str = '''
·········
·········
····□····
····ζ····
·········
····ξ····
·········
·········
·········
'''

threshold_value = 0.1

def run():
    s   = defaults().derive(size=9, view_size=2)
    map = from_unicode(world_str, s)

    # Initialize belief from probabilities dictionary
    # NOTE all other agents look the same from the pov of level 0
    b0 = np.full((s.size,s.size,4),list(s.probs.values()))
    # Sanity check that it adds to 1 at every cell
    assert (np.sum(b0, axis=2)==1).all()
    # Depending on threshold, be a solipsist or believe in someday friendship
    dump_thresholded(b0, 0)

    memory = init_memory(map, s)
    senses = list(sense_environment(map, memory, s, 0))
    communicate(memory, senses, s)
    senses = list(sense_environment(map, memory, s, 0))
    print(f'Map/Memories before movement')
    map.full_render(senses)
    
    update_belief_from_ground_truth(b0, senses[0])
    
    print("Level 0 Belief:")
    print(b0)

    a_info = map.agents_info
    actions = np.array([[0,1],[1,0]])
    transition(map, actions, s)
    senses = list(sense_environment(map, memory, s, 1))
    print(f'Map/Memories after movement')
    map.full_render(senses)

    update_belief_from_ground_truth(b0, senses[0])

    print('Level 0 Belief (t=1):')
    print(b0)

def dump_thresholded(b0, b1):
    b0[:,:,3] = np.where(b0[:,:,3]<threshold_value, 0, b0[:,:,3])
    b0[:,:,0] += 1-np.sum(b0, axis=2)
    assert (np.sum(b0, axis=2)==1).all()

def update_belief_from_ground_truth(b0, sense):
    # NOTE that this does not include ALL belief updating.
    #      Some may be done by rollout
    # In particular, agents are not moved around at all
    #      In fact, they're not distinguished from each other at all
    max_t = np.max(sense.memory.time)
    update_grid = np.where(sense.memory.time==max_t,sense.memory.map.grid,-2)
    # -2 is unseen. Don't update based on this!
    # LAYERS
    # 0: empty
    #    codes: 0
    # 1: obstacle
    #    codes: -1, 1
    update_grid = np.where(update_grid==-1,1,update_grid)
    # 2: goal
    #    codes: 2
    # 3: agent 
    #    codes: 3,4,5,...
    update_grid = np.min(update_grid,3)
    
    belief_update = broadcast(update_grid,4) == np.arange(4)
    belief_update_mask = broadcast(np.max(belief_update,axis=2),4)
    b0[belief_update_mask] = belief_update[belief_update_mask]
    assert (np.sum(b0, axis=2)==1).all()