from copy import deepcopy
from itertools import chain
from os.path import normpath
import numpy as np

from ..environment import transition, Map
from ..monte_carlo import generate_worlds, emplace
from ..belief      import *
from ..memory      import init_memory, map_for_simulate, sense_environment, Memory, communicate
from ..utils       import PolicyInputs, at_xy , broadcast
from .utils        import empty_actions
from .rollout      import egocentric_rollout

def emplace_belief(phi_t, world, belief, a_i, a_c, sense, s):
    phi_mask, phi, guar_mask = phi_t
    emplaced = np.where(phi_mask, world.grid, phi) # phi_mask indicates where belief state has default multinomial per-cell probabilities
    b0_amax  = np.argmax(belief.level_0, -1)
    emplaced = np.where(guar_mask, b0_amax, emplaced)
    emplaced = np.where(emplaced >= s.codes['agent'], s.codes['empty'], emplaced) # Remove duplicate agents
    for xy, fc in list(generate_friends(belief, a_i, a_c, s)) + [(sense.xy, sense.code)]:
        emplaced[int(xy[0]), int(xy[1])] = fc
    emplaced = map_for_simulate(Memory(Map(s, emplaced), sense.memory.time), s) # type: ignore
    return emplaced

def belief_monte_carlo_rollout(p, s):
    communicate(p.memory, p.sense_info, s)
    actions = empty_actions(len(p.sense_info))
    for i, sense in enumerate(p.sense_info):   # This loop is over agents. Certain agents. That are actually in memory. With certainty.
        world_acts = np.zeros((s.n_worlds, 2))
        world_vals = np.zeros(s.n_worlds)
        belief     = p.beliefs[sense.code]

        ego_int_b0 = belief.level_0 * s.n_worlds * s.belief_max_friends * s.russian_trust_factor

        friends_iter = sorted(enumerate(belief.friends_dist), key=lambda r : r[1][0])
        friends_iter = chain(friends_iter, [(0, self_friend(sense))])

        for a_i, friend_dist in friends_iter: # I am simulating the people in MY belief state
            a_c  = int(friend_dist[0])
            phis = generate_phis(belief, a_i, s) # Local worlds from choosing a particular friend
            if a_c == 0:
                continue

            old_friends                  = belief.friends_dist.copy()
            if a_i == -1:
                previous_probability     = 1.0
            else:
                previous_probability     = old_friends[a_i, 1]
            int_action_probs, int_b1_b0s = initialize_intermediate_belief(s)

            for w, (phi_t, world) in enumerate(zip(phis, s.worlds)):
                local_bel = deepcopy(belief)
                memory    = deepcopy(p.memory)
                emplaced  = emplace_belief(phi_t, world, local_bel, a_i, a_c, sense, s)
                imagined  = emplaced.clone()

                world_sense_info = list(sense_environment(emplaced, memory, s, p.t + 1))
                p_base = PolicyInputs(emplaced, world_sense_info, memory, local_bel, p.base_policy, p.t + 1)

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
                # print(f'Emplaced world {w} and observations')
                # emplaced.full_render(world_sense_info)

                # Transition the imagined world by the action chosen by rollout
                transition(imagined, base_policy_actions, s)
                # print(f'Imagined {w} after chosen action')
                # imagined.color_render()
                # Partial belief update based on this imagined world
                add_sample_to_intermediate_belief(s, (imagined, act), int_action_probs,     int_b1_b0s)#                                              
                if s.belief_update_egocentric:
                    add_weighted_sample_to_intermediate_belief_ego(s, imagined.grid, ego_int_b0, weight=friend_dist[1])
            if a_i > 0: # Do not update ourselves like we do friends
                normalize_sampled_belief(int_action_probs, int_b1_b0s, previous_probability)
                update_belief_from_simulation(s, belief, int_b1_b0s, int_action_probs, a_i, old_friends[a_i, 0])
        normalize_sampled_belief_ego(ego_int_b0, 1.0)

        print(f'Resulting egocentric belief distribution')
        render_belief_dists(ego_int_b0, s)
        if s.belief_update_egocentric:

            belief.level_0 = np.where(broadcast(guaranteed_mask(belief.level_0), 4), belief.level_0, ego_int_b0)

            print(f'Egocentric belief post guarantees')
            render_belief_dists(belief.level_0, s)

        # print(f'Belief choices')
        # print(world_acts)
        # print(world_vals)
        best_w         = np.argmax(world_vals)
        actions[i, :]  = world_acts[best_w, :]
    return actions
