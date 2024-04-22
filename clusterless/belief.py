import rich
import numpy as np
import numpy.typing as npt

from . import utils

class Belief():
    friends_dist : npt.ArrayLike # (f, (code, existence_prob, x, y))
    beliefs      : npt.ArrayLike # (n, n, 4, f + 1)

    def __init__(self, s):
        # Initialize belief1 as my b0 + a bunch of empty b0s (they assume everything is empty with prob 1),
        #       together with the array of labels w/ probs all at 0
        # Maybe it's 'more principled' to keep a probability over own location/action history too?? (always 1 for us)

        self.friends_dist = np.zeros((s.belief_max_friends+1,4))
        self.beliefs      = np.full((s.size,s.size,4,s.belief_max_friends+1), 
                                    np.full((s.belief_max_friends+1,4),[1,0,0,0]).transpose(), 
                                    dtype=float)
        self.level_0 = self.init_belief0(s)
        assert (np.sum(self.beliefs, axis=2) == 1).all()

    # These are tedious to define, but let us abstract the underlying numpy array implementation
    @property
    def level_0(self):
        return self.beliefs[:,:,:,0]  # type: ignore

    @level_0.setter
    def level_0(self, value):
        self.beliefs[:,:,:,0] = value # type: ignore

    @property
    def level_1(self):
        return self.beliefs[:,:,:,1:]  # type: ignore

    @level_1.setter
    def level_1(self, value):
        self.beliefs[:,:,:,1:] = value # type: ignore

    def show(self, s):
        for x in range(s.belief_max_friends):
            i = x + 1
            if self.friends_dist[i, 1]: # type: ignore
                print(f"I believe (P={self.friends_dist[i,1]}) I have a friend {s.symbols[int(self.friends_dist[i,0])]} at location {self.friends_dist[i,2:]}") # type: ignore

    def to_grid(self):
        return np.argmax(self.level_0, axis=-1) # type: ignore

    def init_belief0(self, s):
        # Initialize belief from probabilities dictionary
        # NOTE all other agents look the same from the pov of level 0
        b0 = np.full((s.size,s.size,4), list(s.probs.values()))
        # Sanity check that it adds to 1 at every cell
        assert (np.sum(b0, axis=2)==1).all()
        # Depending on threshold, be a solipsist or believe in someday friendship :)
        self.dump_thresholded(b0, s)
        return b0

    def dump_thresholded(self, b0, s):
        b0[:,:,3] = np.where(b0[:,:,3] < s.belief_threshold, 0, b0[:,:,3]) # KILL!
        b0[:,:,0] += 1-np.sum(b0, axis=2)
        assert (np.sum(b0, axis=2)==1).all()

def init_beliefs(env_map, s):
    return {k : Belief(s) for k in env_map.agents_info.codes}

def render_belief_dists(b0, s):
    from .map import render, Map
    if (np.argmax(b0, axis=-1) == 0).all():
        return # Belief is all zeros
    map          = Map(s, np.argmax(b0, axis=-1)) # type: ignore
    rendered     = [map.color_render(show=False)]
    descriptions = [f'{d:<{s.size}}' for d in ('argmax', 'empty', 'obs', 'goal', 'agent')]
    for c in range(4):
        p       = b0[:, :, c]
        choices = np.arange(8)
        conds   = [p < 0.05, p < 0.15, p < 0.3, p < 0.5, p < 0.65, p < 0.8, p < 0.99, p < 1.01]
        codes   = np.select(conds, choices, default=0) # type: ignore
        rendered.append(render(codes, s.dist_symbols))
    rich.print(utils.horizontal_join(rendered))
    rich.print(utils.horizontal_join(descriptions))
    print()

def self_friend(sense):
    return np.array([sense.code, 1.0, *sense.xy])

def spare_friend(friends_dist):
    argmin_probability = np.argmin(friends_dist[:, 1])
    return np.minimum(argmin_probability, 1)

def update_belief_from_ground_truth(s, belief, sense):
    # NOTE that this does not include ALL belief updating.
    #      Some may be done by rollout
    # In particular, agents are not moved around at all
    #      In fact, they're not distinguished from each other at all
    # Also, we just assume agents don't exist after they move out of view
    max_t       = np.max(sense.memory.time)
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

    belief_update      = utils.broadcast(b0_grid,4) == np.arange(4)
    belief_update_mask = utils.broadcast(np.max(belief_update,axis=2),4)
    belief.level_0[belief_update_mask] = belief_update[belief_update_mask]
    # assert(np.sum(belief.beliefs, axis=2)==1).all()

    # Agents we can see, we believe in absolutely
    agents = np.unique(update_grid)  #& update_grid !=sense.code).any()
    agents = agents[np.logical_and(agents!=sense.code,agents>=3)]

    # Before we update agents with P=1 locations, we zero out any existing entries for them
    belief.friends_dist[:,1][np.isin(belief.friends_dist[:,0], agents)]=0

    for agent in agents:
        # NOTE if a view is crowded with other agents, and the memory_bound is too low,
        #      the reasoner WILL get overwhelmed and start to forget people after others 
        #      introduce themselves 

        spare = spare_friend(belief.friends_dist)

        x, y = sense.memory.map.coords[(update_grid == agent).reshape((np.prod((s.size,s.size)),))][0]
        belief.friends_dist[spare,:] = np.array([agent,1,x,y]) 
        # Then replace the corresponding view of belief.beliefs with the reasoning agent's b0
        #      that is, with a copy of our newly updated belief.beliefs[:,:,:,0]
        belief.beliefs[:,:,:,spare+1] = belief.level_0

    belief.friends_dist[0, :] = self_friend(sense) # Ensure ourselves

    # assert (np.sum(belief.beliefs, axis=2)==1).all()

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

def initialize_intermediate_belief(s, d=5):
    intermediate_action_probabilities = np.zeros(d)
    intermediate_b1_b0s = np.zeros((s.size, s.size, 4, d))
    return intermediate_action_probabilities, intermediate_b1_b0s

def normalize_sampled_belief(intermediate_action_probabilities, intermediate_b1_b0s, previous_probability):
    # NOTE: the order of these tensors is the same as the order of actions in s.action_space
    #       up, down, left, right, stay
    intermediate_action_probabilities /= np.sum(intermediate_action_probabilities)
    intermediate_action_probabilities *= previous_probability
    normalize_sampled_belief_ego(intermediate_b1_b0s, previous_probability)

def normalize_sampled_belief_ego(intermediate_b1_b0s, previous_probability):
    # Note that this is normalized relative to the action taken! So we only divide by a subset of worlds
    sum_b0_b1s = utils.broadcast(np.sum(intermediate_b1_b0s,2),4,axis=2)
    with np.errstate(divide='ignore',invalid='ignore'):
        intermediate_b1_b0s[...] = np.where(sum_b0_b1s>0,intermediate_b1_b0s/sum_b0_b1s,0)*previous_probability

def add_sample_to_intermediate_belief(s, sample, intermediate_action_probabilities, intermediate_b1_b0s, weight=1.0):
    action_number = s.action_number_lookup[str(tuple(np.squeeze(sample[1])))] 
    intermediate_action_probabilities[action_number] += weight

    #TODO clean up the sample first 
    # TODO split off the cleaning up function so it can be shared with ground truth update
    update = utils.broadcast(sample[0], 4) == np.arange(4)
    intermediate_b1_b0s[...,action_number] += update

def add_weighted_sample_to_intermediate_belief_ego(s, grid, intermediate_b1_b0s, weight=1.0):
    update = utils.broadcast(grid, 4) == np.arange(4)
    intermediate_b1_b0s += update * weight

def update_belief_from_simulation(s, belief, int_b1_b0s, int_action_probs, agent_index, agent_code):
    # print(f"updating agent {agent_code} at index {agent_index}")
    old_loc = np.copy(belief.friends_dist[agent_index,2:])
    # print(old_loc)
    # print(agent_code)
    # print(s.symbols[agent_code])
    # print(belief.friends_dist)
    update_belief_for_agent_location(s, belief.beliefs[:,:,:,agent_index], int_b1_b0s[...,4],
                                     belief.friends_dist[agent_index,:], int_action_probs[4], agent_code, 4, old_loc)
    # print(belief.friends_dist)

    # Only overwrite the lowest probability slices, and only do so if the new data is higher probability
    # THIS MAY HAVE PROBLEMS. In particular, if the order of the agents is weird, one high prob could get smeared out and dumped even though another (earlier updated)
    #      had a higher probability agent-location.
    # For now, oh well... Later: TODO fix or find a better way to structure this??
    
    # print(old_loc)
    for action_num in range(4):
        worst = np.min(belief.friends_dist[:,1])
        worst_index = spare_friend(belief.friends_dist)
        if int_action_probs[action_num] > worst:
            update_belief_for_agent_location(s,belief.beliefs[...,worst_index],int_b1_b0s[...,action_num],belief.friends_dist[worst_index,:],int_action_probs[action_num],agent_code,action_num,old_loc)

def update_belief_for_agent_location(s, b1_slice, int_b1_slice, action_probs_slice, 
                                     int_action_probs_slice, agent_code, action_num, old_location):
    action_probs_slice[0] = agent_code
    action_probs_slice[1] = int_action_probs_slice
    x, y = s.action_space[action_num]
    action_probs_slice[2] = (old_location[1] + y)%s.size
    action_probs_slice[3] = (old_location[0] + x)%s.size
    update_mask = utils.broadcast(np.sum(int_b1_slice,2),4,axis=2) ==1
    b1_slice[update_mask] = int_b1_slice[update_mask]

def b0_to_map(b0):
    m0 = np.argmin(b0, axis=2)
    print(m0)

def guaranteed_mask(b0):
    return np.max(np.where(b0[...,:2] == 1, b0[..., :2], 0),axis=2)

def generate_friends(belief, a_i, a_c, s):
    friends_dist = belief.friends_dist
    for fc in np.unique(friends_dist[:, 0]): # Iterate over our friends, e.g. fred, sally, anne
        if fc == a_c:
            yield belief.friends_dist[a_i, -2:], a_c # Coordinates of the current agent
            break
        friend = None
        f_dist = friends_dist[friends_dist[:, 0] == fc] # Distribution of only freds
        opts   = f_dist[:, -2:] # Coordinates of multiple friends
        probs  = f_dist[:, 1]   # Probabilities of multiple friends
        if np.sum(probs) != 1.0:
            probs = np.array(list(probs) + [1 - np.sum(probs)])
            opts  = np.array(list(opts)  + [np.array([np.nan, np.nan])])
        friend = s.gen.choice(opts, p=probs) # Choice of left fred or right fred. Fred is required.
        if not np.isnan(friend).any(): # :( Sad when our friends are None
            yield friend, a_c  # Great joy when our friends are not None. YIELD

def generate_phis(belief, a_i, s):
    b0 = belief.beliefs
    b0 = b0[...,a_i]
    # For a given agent's b0, generate the phis we'll emplace into our worlds

    #yea... this is copy paste and needs to be fixed TODO but belief structure is awk for priors rn
    prior = np.full((s.size,s.size,4), list(s.probs.values()))
    prior[:,:,3] = np.where(prior[:,:,3] < s.belief_threshold, 0, prior[:,:,3]) # KILL!
    prior[:,:,0] += 1-np.sum(prior, axis=2)
    b0_mask = (b0 == prior).all(axis=2)

    guaranteed_objects = guaranteed_mask(b0) 

    for _ in range(s.n_worlds):
        pre_map = np.zeros((s.size,s.size))
        random_from_probs = lambda probs: s.gen.choice(np.arange(len(s.probs)), size=1, p=probs)
        random_map = np.apply_along_axis(random_from_probs, 2,b0)
        random_map = np.squeeze(random_map)
        yield b0_mask, random_map, guaranteed_objects

    # exit()

