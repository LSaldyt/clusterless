from experiments.base import defaults

from clusterless.memory   import init_memory, sense_environment, communicate
from clusterless.scenario import from_unicode
from experiments.base     import defaults
from clusterless.environment import transition

from clusterless.utils import at_xy
from clusterless import utils

from clusterless.monte_carlo import generate_worlds

from clusterless.map import Map

from clusterless.monte_carlo import emplace

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

other_agent_code  = 4
other_agent_index = 1 # HARDCODED for now, TODO In general run for all indices in arbitrary order

from clusterless.belief import *

def run():
    s   = defaults().derive(size=9, view_size=2, n_worlds=3, belief_threshold= 0.1, belief_max_friends=7)
    map = from_unicode(world_str, s)

    # This section is the environment setup stuff
    memory = init_memory(map, s)
    senses = list(sense_environment(map, memory, s, 0))
    communicate(memory, senses, s)
    senses = list(sense_environment(map, memory, s, 0))
    print(f'Map/Memories before movement')
    map.full_render(senses)

    belief = Belief(s)
    update_belief_from_ground_truth(s, belief, senses[0])
    belief.show(s)

    # This section is just running 3 hardcoded actions 
    actions = np.array([[0,1],[1,0],[-1,0]])
    transition(map, actions, s)
    senses = list(sense_environment(map, memory, s, 1))
    print(f'Map/Memories after movement')
    map.full_render(senses)

    # Save previous probabilities, bc we will be updating from these with rollout
    #      (Except in cases where we know exactly what the ground truth is now)
    old_friends = belief.friends_dist.copy() # type: ignore

    # TODO Loop over agent indices
    # for agent_index in range(belief.friends_dist[0])

    # This will need to save before the whole run so that we remember previous probs
    previous_probability = old_friends[other_agent_index][1]

    # Update everything that's certain. No hallucinations here
    update_belief_from_ground_truth(s, belief, senses[0])

    # Zero out anything in previous probabilities that we're certain about rn
    # That way we won't touch it when we update from data.
    mem         = senses[0].memory
    max_t       = np.max(mem.time)
    update_grid = np.where(mem.time == max_t, mem.map.grid, -2)
    # agents      = np.unique(update_grid) # All unique codes in mem.map.grid (in current time), and -2
    # agents      = agents[np.logical_and(agents != senses[0].code, agents >= 3)]
    agents = update_grid[(update_grid >= 3) & (update_grid != senses[0].code)]
    old_friends[:, 1][~np.isin(belief.friends_dist[:,0], agents, invert=True)] = 0  # type: ignore

    # Now that we've taken into account solid, absolute reality, we have to do some 
    #     kind of calculation to update our uncertain cells.
    # VERY IMPORTANT: do not update cells we are certain of during this part! 

    # # The dumbest version: if I didn't see someone this round, I forget them completely
    # # In the future, this portion will be replaced with actual updating via rollout\
    # if s.do_dumb_beliefs:
    # max_t = np.max(senses[0].memory.time)
    # update_grid = np.where(senses[0].memory.time==max_t,senses[0].memory.map.grid,-2)
    # agents = np.unique(update_grid)
    # agents = agents[np.logical_and(agents!=senses[0].code,agents>=3)]
    # belief.friends_dist[:,1][np.isin(belief.friends_dist[:,0], agents, invert=True)]=0

    # # Deleting thresholded agents as if they never existed at all
    # dump_thresholded(b1_b0s[:,:,:,0])
    # belief.friends_dist[:,1] = np.where(belief.friends_dist[:,1]<threshold_value,0, belief.friends_dist[:,1])

    # TODO assert that for each agent, the b1_probs sum to at most 1 

    # The rollout version:
    int_action_probs, int_b1_b0s = initialize_intermediate_belief(s)

    # TODO keep track of the probabilities from last time and then multiply by that in the update, so that we aren't getting probabilities from nowhere
    #      e.g. 50% should split into 25%, 25%, 0%, 0%, 0%; not grow into 100%....

    view_fn      = getattr(utils, s.view_type)
    view_offsets = view_fn(s.view_size)
    view_coords  = view_offsets + belief.friends_dist[other_agent_code,2:].astype(int) # type: ignore
    view_coords  = view_coords % map.grid.shape[0] # type:ignore
    # Get my certainties from my b0. We'll remove these from phi
    certainties  = np.logical_and.reduce(belief.beliefs[...,0] != 1,axis=2) # type: ignore

    phi_mask = at_xy(view_coords)

    phis = []
    for world in generate_worlds(s):
        # will there be any in-place issues?
        # TODO: Sample phis randomly from our belief state
        # TODO: Use correct belief state for phi sampling based on our theory of mind level
        # phi           = sample(belief, tom_level)
        # emplaced      = emplace(world, phi)
        # act, eventual = rollout(policy, emplaced)
        # transition(emplaced, act)
        # data.append((act, emplaced))
        phi = np.full((s.size,s.size),-2)
        phi[phi_mask] = world.grid[phi_mask] #type: ignore
        phi = np.where(certainties, phi, -2)
        phis.append(phi)

    test_samples = [(phis[0], np.array([0,1])), (phis[1],np.array([0,1])),(phis[2],np.array([0,0]))]
    for sample in test_samples:
        add_sample_to_intermediate_belief(s, sample, int_action_probs, int_b1_b0s)

    normalize_sampled_belief(int_action_probs, int_b1_b0s, previous_probability)

    # Now we create the new b1 sub states
    # NOTE that we pass in the INDEX of the agent-location, rather than merely the agent code.
    #      These are meaningfully different, as we are keeping track of a belief-history, not just a belief 
    update_belief_from_simulation(s, belief, int_b1_b0s, int_action_probs, other_agent_index, other_agent_code)

    belief.show(s)

    generate_phis(belief, 1, s)
    
