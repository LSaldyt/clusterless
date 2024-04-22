from copy import deepcopy
from os.path import normpath
import numpy as np

from ..environment import transition, Map
from ..monte_carlo import generate_worlds, emplace
from ..belief      import *
from ..memory      import init_memory, map_for_simulate, sense_environment, Memory
from ..utils       import PolicyInputs
from .utils        import empty_actions
from .rollout      import egocentric_rollout

def belief_monte_carlo_rollout(p, s):
    actions = empty_actions(len(p.sense_info))
    for i, sense in enumerate(p.sense_info):   # This loop is over agents. Certain agents. That are actually in memory. With certainty.
        world_acts = np.zeros((s.n_worlds, 2))
        world_vals = np.zeros(s.n_worlds)
        belief     = p.beliefs[sense.code]

        ego_int_b0 = belief.level_0 * s.n_worlds * s.belief_max_friends

        for friend_dist in sorted(belief.friends_dist, key=lambda r : r[0]): # I am simulating the people in MY belief state
            a_i  = int(friend_dist[-1])
            phis = generate_phis(belief, a_i, s) # Local worlds from choosing a particular friend

            old_friends                  = belief.friends_dist.copy()
            previous_probability         = old_friends[a_i][1]
            int_action_probs, int_b1_b0s = initialize_intermediate_belief(s)

            for w, ((phi_mask, phi), world) in enumerate(zip(phis, s.worlds)):
                memory   = deepcopy(p.memory)
                emplaced = np.where(phi_mask, world.grid, phi) # phi_mask indicates where belief state has default multinomial per-cell probabilities
                emplaced = map_for_simulate(Memory(Map(s, emplaced), sense.memory.time), s) # type: ignore
                imagined = emplaced.clone()

                emplaced.color_render()

                world_sense_info = list(sense_environment(emplaced, memory, s, p.t + 1))
                p_base = PolicyInputs(emplaced, world_sense_info, memory, p.beliefs, p.base_policy, p.t + 1)

                base_policy_actions = p.base_policy(p_base, s)

                codes = [sense.code for sense in world_sense_info]
                assert not (base_policy_actions == 0).all(), f'Base policy returned all stay actions, reject!!'
                act, val = egocentric_rollout(emplaced, deepcopy(sense.memory), codes, memory,
                                              base_policy_actions, p.base_policy, 
                                              sense.code, p.t + 2, s, reveal=True) # IMPORTANT!!! Now we can reveal the map!!
                ego_i = list(codes).index(sense.code)
                base_policy_actions[ego_i, :] = act
                world_acts[w, :] = act
                world_vals[w]    = val
                print(f'Emplaced world {w} and observations')
                emplaced.full_render(world_sense_info)

                # Transition the imagined world by the action chosen by rollout
                transition(imagined, base_policy_actions, s)
                print(f'Imagined {w} after chosen action')
                imagined.color_render()
                # Partial belief update based on this imagined world
                add_sample_to_intermediate_belief(s, (imagined, act), int_action_probs,     int_b1_b0s)#                                              
                add_weighted_sample_to_intermediate_belief_ego(s, imagined.grid, ego_int_b0, weight=friend_dist[1])
            normalize_sampled_belief(int_action_probs, int_b1_b0s, previous_probability)
            update_belief_from_simulation(s, belief, int_b1_b0s, int_action_probs, a_i, old_friends[a_i, 0])
        normalize_sampled_belief_ego(ego_int_b0, 1.0)
        print(f'Resulting egocentric belief distribution')
        render_belief_dists(ego_int_b0, s)

        best_w        = np.argmax(world_vals)
        actions[i, :] = world_acts[best_w, :]
    return actions
