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
memory_bound = 7

def run():
    s   = defaults().derive(size=9, view_size=2)
    map = from_unicode(world_str, s)

    # Initialize belief1 as my b0 + a bunch of empty b0s (they assume everything is empty with prob 1),
    #       together with the array of labels w/ probs all at 0
    # Maybe it's 'more principled' to keep a probability over own location/action history too?? (always 1 for us)
    # b0 = np.full((s.size,s.size,4,memory_bound+1),list(s.probs.values()),)

    b1_probabilities = np.zeros((memory_bound,2))
    #np.arange(2*memory_bound).reshape((memory_bound,2))
    b1_b0s = np.full((s.size,s.size,4,memory_bound+1),np.full((memory_bound+1,4),[1,0,0,0]).transpose(),dtype=float)
    b1_b0s[:,:,:,0] = init_belief0(s)
    assert (np.sum(b1_b0s, axis=2)==1).all()

    memory = init_memory(map, s)
    senses = list(sense_environment(map, memory, s, 0))
    communicate(memory, senses, s)
    senses = list(sense_environment(map, memory, s, 0))
    print(f'Map/Memories before movement')
    map.full_render(senses)
    
    update_belief_from_ground_truth(b1_b0s, b1_probabilities, senses[0])
    
    print("Time t=0")
    print("Level 0 Belief:")
    # print(b1_b0s[:,:,:,0])
    print("Level 1 Belief:")
    for x in range(memory_bound):
        if b1_probabilities[x,1]:
            print(f"I believe in {int(b1_probabilities[x,0])} (P={b1_probabilities[x,1]})")
            # print(b1_b0s[:,:,:,x+1])

    a_info = map.agents_info
    actions = np.array([[0,1],[1,0]])
    transition(map, actions, s)
    senses = list(sense_environment(map, memory, s, 1))
    print(f'Map/Memories after movement')
    map.full_render(senses)

    update_belief_from_ground_truth(b1_b0s, b1_probabilities, senses[0])

    print("Time t=0")
    print("Level 0 Belief:")
    # print(b1_b0s[:,:,:,0])
    print("Level 1 Belief:")
    for x in range(memory_bound):
        if b1_probabilities[x,1]:
            print(f"I believe in {int(b1_probabilities[x,0])} (P={b1_probabilities[x,1]})")

def dump_thresholded(b0):
    b0[:,:,3] = np.where(b0[:,:,3]<threshold_value, 0, b0[:,:,3])
    b0[:,:,0] += 1-np.sum(b0, axis=2)
    assert (np.sum(b0, axis=2)==1).all()

def init_belief0(s):
    # Initialize belief from probabilities dictionary
    # NOTE all other agents look the same from the pov of level 0
    b0 = np.full((s.size,s.size,4),list(s.probs.values()))
    # Sanity check that it adds to 1 at every cell
    assert (np.sum(b0, axis=2)==1).all()
    # Depending on threshold, be a solipsist or believe in someday friendship
    dump_thresholded(b0)
    return b0

def update_belief_from_ground_truth(b1_b0s, b1_probabilties, sense):
    # NOTE that this does not include ALL belief updating.
    #      Some may be done by rollout
    # In particular, agents are not moved around at all
    #      In fact, they're not distinguished from each other at all
    # Also, we just assume agents don't exist after they move out of view
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
    b0_grid = np.where(update_grid>=3,3,update_grid)

    belief_update = broadcast(b0_grid,4) == np.arange(4)
    belief_update_mask = broadcast(np.max(belief_update,axis=2),4)
    b1_b0s[:,:,:,0][belief_update_mask] = belief_update[belief_update_mask]
    assert (np.sum(b1_b0s, axis=2)==1).all()

    # Deleting thresholded agents as if they never existed at all
    dump_thresholded(b1_b0s[:,:,:,0])
    b1_probabilties[:,1] = np.where(b1_probabilties[:,1]<threshold_value,0, b1_probabilties[:,1])

    # Agents we can see, we believe in absolutely
    agents = np.unique(update_grid)  #& update_grid !=sense.code).any()
    agents = agents[np.logical_and(agents!=sense.code,agents>=3)]
    for agent in agents:
        # NOTE if a view is crowded with other agents, and the memory_bound is too low,
        #      the reasoner WILL get overwhelmed and start to forget people after others 
        #      introduce themselves 

        # Find the lowest P entry in b1_probabilities
        argmin_probability = np.argmin(b1_probabilties[:,1])
        # Replace it with a new entry
        b1_probabilties[argmin_probability,:] = np.array([agent,1]) 
        # Then replace the corresponding view of b1_b0s with the reasoning agent's b0
        #      that is, with a copy of our newly updated b1_b0s[:,:,:,0]
        b1_b0s[:,:,:,argmin_probability+1] = b1_b0s[:,:,:,0]
    assert (np.sum(b1_b0s, axis=2)==1).all()

    # The dumbest version: if I didn't see someone this round, I forget them completely
    b1_probabilties[:,1][np.isin(b1_probabilties[:,0], agents, invert=True)]=0

    # TODO make sure that updating actually makes sense.
    # IN particular, the sum of a single agent id's probabilities must add up to 1
    # So, when we update 