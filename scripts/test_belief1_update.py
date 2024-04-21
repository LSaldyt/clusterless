from experiments.base import defaults

from clusterless.memory   import init_memory, sense_environment, communicate
from clusterless.scenario import from_unicode
from experiments.base     import defaults
from clusterless.environment import transition
from clusterless.utils import broadcast

from clusterless.utils import at_xy
from clusterless import utils

from clusterless.monte_carlo import generate_worlds

import numpy as np

world_str = '''
·········
·········
····□····
····ζ····
·········
····ξΞ···
·········
·········
·········
'''

threshold_value = 0.1
memory_bound = 7

other_agent = 4

def run():
    s   = defaults().derive(size=9, view_size=2, n_worlds=3)
    map = from_unicode(world_str, s)

    # Initialize belief1 as my b0 + a bunch of empty b0s (they assume everything is empty with prob 1),
    #       together with the array of labels w/ probs all at 0
    # Maybe it's 'more principled' to keep a probability over own location/action history too?? (always 1 for us)
    # b0 = np.full((s.size,s.size,4,memory_bound+1),list(s.probs.values()),)

    b1_probabilities = np.zeros((memory_bound,4))
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
    
    update_belief_from_ground_truth(s,b1_b0s, b1_probabilities, senses[0])
    
    print("Time t=0")
    print("Level 0 Belief:")
    # print(b1_b0s[:,:,:,0])
    print("Level 1 Belief:")
    for x in range(memory_bound):
        if b1_probabilities[x,1]:
            print(f"I believe in {int(b1_probabilities[x,0])} (P={b1_probabilities[x,1]}) at location {b1_probabilities[x,2:]}")
            # print(b1_b0s[:,:,:,x+1])

    
    actions = np.array([[0,1],[1,0],[-1,0]])
    transition(map, actions, s)
    senses = list(sense_environment(map, memory, s, 1))
    print(f'Map/Memories after movement')
    map.full_render(senses)

    previous_b1_probabilities = b1_probabilities.copy()

    update_belief_from_ground_truth(s,b1_b0s, b1_probabilities, senses[0])

    max_t = np.max(senses[0].memory.time)
    update_grid = np.where(senses[0].memory.time==max_t,senses[0].memory.map.grid,-2)
    agents = np.unique(update_grid)
    agents = agents[np.logical_and(agents!=senses[0].code,agents>=3)]
    previous_b1_probabilities[:,1][~np.isin(b1_probabilities[:,0], agents, invert=True)]=0


    print(previous_b1_probabilities)
    print(b1_probabilities)
    print(np.isin(b1_probabilities[:,1],1))
    previous_b1_probabilities[:,1][np.isin(b1_probabilities[:,1],1)]=0
    print(previous_b1_probabilities)
    exit()

    for x in range(memory_bound):
        if b1_probabilities[x,1]:
            print(f"I believe in {int(b1_probabilities[x,0])} (P={b1_probabilities[x,1]}) at location {b1_probabilities[x,2:]}")

    # Now that we've taken into account solid, absolute reality, we have to do some 
    #     kind of calculation to update our uncertain cells.
    # VERY IMPORTANT: do not update cells we are certain of during this part! 

    # # The dumbest version: if I didn't see someone this round, I forget them completely
    # # In the future, this portion will be replaced with actual updating via rollout\
    # max_t = np.max(senses[0].memory.time)
    # update_grid = np.where(senses[0].memory.time==max_t,senses[0].memory.map.grid,-2)
    # agents = np.unique(update_grid)
    # agents = agents[np.logical_and(agents!=senses[0].code,agents>=3)]
    # b1_probabilities[:,1][np.isin(b1_probabilities[:,0], agents, invert=True)]=0

    # # Deleting thresholded agents as if they never existed at all
    # dump_thresholded(b1_b0s[:,:,:,0])
    # b1_probabilities[:,1] = np.where(b1_probabilities[:,1]<threshold_value,0, b1_probabilities[:,1])

    # TODO assert that for each agent, the b1_probs sum to at most 1 


    # The rollout version:
    int_action_probs, int_b1_b0s = initialize_intermediate_belief(s)

    # TODO keep track of the probabilities from last time and then multiply by that in the update, so that we aren't getting probabilities from nowhere
    #      e.g. 50% should split into 25%, 25%, 0%, 0%, 0%; not grow into 100%....

    a_info = map.agents_info
    view_fn      = getattr(utils, s.view_type)
    view_offsets = view_fn(s.view_size)
    view_coords    = view_offsets + b1_probabilities[other_agent,2:].astype(int)
    view_coords    = view_coords % map.grid.shape[0]
    # Get my certainties from my b0. We'll remove these from phi
    certainties = np.logical_and.reduce(b1_b0s[...,0] != 1,axis=2)

    phi_mask = at_xy(view_coords)

    phis = []
    for world in generate_worlds(s):
        # will there be any in-place issues?
        phi = np.full((s.size,s.size),-2)
        phi[phi_mask] = world.grid[phi_mask]
        phi = np.where(certainties, phi, -2)
        phis.append(phi)

    test_samples = [(phis[0], np.array([0,1])), (phis[1],np.array([0,1])),(phis[2],np.array([0,0]))]
    for sample in test_samples:
        add_sample_to_intermediate_belief(s,sample,int_action_probs, int_b1_b0s)

    normalize_sampled_belief(int_action_probs, int_b1_b0s)

    # Now we create the new b1 sub states
    # NOTE that we pass in the INDEX of the agent-location, rather than merely the agent code.
    #      These are meaningfully different, as we are keeping track of a belief-history, not just a belief 
    update_belief_from_simulation(s,b1_b0s, b1_probabilities, int_b1_b0s, int_action_probs, 1, other_agent)

    print("Time t=0")
    print("Level 0 Belief:")
    # print(b1_b0s[:,:,:,0])
    print("Level 1 Belief:")
    for x in range(memory_bound):
        if b1_probabilities[x,1]:
            print(f"I believe in {int(b1_probabilities[x,0])} (P={b1_probabilities[x,1]}) at location {b1_probabilities[x,2:]}")

    

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

def update_belief_from_ground_truth(s, b1_b0s, b1_probabilities, sense):
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
    assert(np.sum(b1_b0s, axis=2)==1).all()

    # Agents we can see, we believe in absolutely
    agents = np.unique(update_grid)  #& update_grid !=sense.code).any()
    agents = agents[np.logical_and(agents!=sense.code,agents>=3)]

    # Before we update agents with P=1 locations, we zero out any existing entries for them
    b1_probabilities[:,1][np.isin(b1_probabilities[:,0], agents)]=0

    for agent in agents:
        # NOTE if a view is crowded with other agents, and the memory_bound is too low,
        #      the reasoner WILL get overwhelmed and start to forget people after others 
        #      introduce themselves 

        # Find the lowest P entry in b1_probabilities
        argmin_probability = np.argmin(b1_probabilities[:,1])
        # Replace it with a new entry

        # TODO get the x, y of the agent by code in the update grid
        x, y = sense.memory.map.coords[(update_grid == agent).reshape((np.prod((s.size,s.size)),))][0]
        b1_probabilities[argmin_probability,:] = np.array([agent,1,x,y]) 
        # Then replace the corresponding view of b1_b0s with the reasoning agent's b0
        #      that is, with a copy of our newly updated b1_b0s[:,:,:,0]
        b1_b0s[:,:,:,argmin_probability+1] = b1_b0s[:,:,:,0]
    assert (np.sum(b1_b0s, axis=2)==1).all()

# ---Notes on the full version---
    # 
    # We consider a particular agent-location B-l in the prior.
    #   Call phi the part of the world (in some sample) that is different for B-l than for A, which we (agent A) are uncertain over
    # Then, we wish to determine two things: the probability distribution over actions a and the 
    #   corresponding B-l_a posteriors given those actions
    # Sample worlds according to the agent-location component of the prior
    #   For a given phi, we can calculate MAR and determine where B would go.
    #   Add this to a new 5-vector that is action-labeled. e.g. [0,1]+=1
    #   We also generate 5 nxnx4 tensors, one for each possible action.
    #   After each phi, add to each cell (of the action-specified tensor) what was true about that phi. Leave non-phi cells alone
    # After sampling is done, normalize the action probability vector, associate each action with a (also normalized)
    #   3D tensor, and add them to A's b1 belief state.

def initialize_intermediate_belief(s):
    intermediate_action_probabilities = np.zeros(5)
    intermediate_b1_b0s = np.zeros((s.size, s.size, 4, 5))
    return intermediate_action_probabilities, intermediate_b1_b0s

def normalize_sampled_belief(intermediate_action_probabilities, intermediate_b1_b0s):
    # NOTE: the order of these tensors is the same as the order of actions in s.action_space
    #       up, down, left, right, stay
    intermediate_action_probabilities /= np.sum(intermediate_action_probabilities)
    # Note that this is normalized relative to the action taken! So we only divide by a subset of worlds
    sum_b0_b1s = broadcast(np.sum(intermediate_b1_b0s,2),4,axis=2)
    with np.errstate(divide='ignore',invalid='ignore'):
        intermediate_b1_b0s[...] = np.where(sum_b0_b1s>0,intermediate_b1_b0s/sum_b0_b1s,0)

def add_sample_to_intermediate_belief(s, sample, intermediate_action_probabilities, intermediate_b1_b0s):
    action_number = s.action_space_lookup[tuple(sample[1])]
    intermediate_action_probabilities[action_number] +=1

    #TODO clean up the sample first 
    # TODO split off the cleaning up function so it can be shared with ground truth update
    update = broadcast(sample[0],4) == np.arange(4)
    intermediate_b1_b0s[...,action_number] += update

def update_belief_from_simulation(s,b1_b0s, b1_probabilities, int_b1_b0s, int_action_probs, agent_index, agent_code):
    old_loc = b1_probabilities[agent_index,2:]
    print(old_loc)
    print(agent_code)
    print(s.symbols[agent_code])
    print(b1_probabilities)
    update_belief_for_agent_location(s,b1_b0s[:,:,:,agent_index], int_b1_b0s[...,4],b1_probabilities[agent_index,:],int_action_probs[4],agent_code,4,old_loc)
    print(b1_probabilities)
    # Only overwrite the lowest probability slices, and only do so if the new data is higher probability
    # THIS MAY HAVE PROBLEMS. In particular, if the order of the agents is weird, one high prob could get smeared out and dumped even though another (earlier updated)
    #      had a higher probability agent-location.
    # For now, oh well... Later: TODO fix or find a better way to structure this??
    for action_num in range(4):
        worst = np.min(b1_probabilities[:,1])
        worst_index = np.argmin(b1_probabilities[:,1])
        if int_action_probs[action_num] > worst:
            update_belief_for_agent_location(s,b1_b0s[...,worst_index],int_b1_b0s[...,action_num],b1_probabilities[worst_index,:],int_action_probs[action_num],agent_code,action_num,old_loc)

def update_belief_for_agent_location(s,b1_slice, int_b1_slice, action_probs_slice, int_action_probs_slice,agent_code,action_num,old_location):
    action_probs_slice[0] = agent_code
    action_probs_slice[1] = int_action_probs_slice
    x, y = s.action_space[action_num]
    print(f"{x} and {y}")
    action_probs_slice[2] = (old_location[0] + x)%s.size
    action_probs_slice[3] = (old_location[1] + y)%s.size
    update_mask = broadcast(np.sum(int_b1_slice,2),4,axis=2) ==1
    b1_slice[update_mask] = int_b1_slice[update_mask]

def b0_to_map(b0):
    m0 = np.argmin(b0, axis=2)
    print(m0)
