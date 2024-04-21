from copy import deepcopy
import numpy as np

from ..monte_carlo import generate_worlds, emplace
from ..memory      import init_memory, map_for_simulate, sense_environment, Memory
from .utils   import empty_actions
from .rollout import egocentric_rollout

def monte_carlo_rollout(map, sense_info, input_memory, base_policy, t, s):
    # First calculate base policy for all agents
    actions = empty_actions(len(sense_info))
    for i, sense in enumerate(sense_info):
        world_acts = np.zeros((s.n_worlds, 2))
        world_vals = np.zeros(s.n_worlds)
        eventual_worlds = []

        for w, world in enumerate(generate_worlds(s)):
            memory = deepcopy(input_memory)
            # print(f'Monte Carlo world {w}')
            emplaced = emplace(sense.memory, world, s)
            emplaced = map_for_simulate(Memory(emplaced, sense.memory.time), s)
            # world.color_render()
            # If we hallucinate agents on the new world, recreate the base policy
            world_sense_info    = list(sense_environment(emplaced, memory, s, t + 1))
            base_policy_actions = base_policy(emplaced, world_sense_info, memory, base_policy, t + 1, s)

            # world_memory = init_memory(world, s)
            # world_memory.update(memory)
            # print(world_memory)

            codes = [sense.code for sense in world_sense_info]
            assert not (base_policy_actions == 0).all(), f'Base policy returned all stay actions, reject!!'
            try:
                act, val = egocentric_rollout(emplaced, deepcopy(sense.memory), codes, memory,
                                              base_policy_actions, base_policy, 
                                              sense.code, t + 2, s, reveal=True) # IMPORTANT!!! Now we can reveal the map!!
            except AssertionError:
                print('Underlying world')
                world.color_render()
                print('Emplaced world and observations')
                emplaced.full_render(world_sense_info)
                exit()
                # world.color_render()
            # print(f'Rollout action, value: {act} {val}')
            world_acts[w, :] = act
            world_vals[w]    = val
            # print(world_acts)
            # eventual_worlds.append(last_world_from_rollout_simulation)
        # print(world_acts)
        # print(world_vals)

        best_w        = np.argmax(world_vals)
        actions[i, :] = world_acts[best_w, :]
        # print(best_w)
        # new_belief = belief_update(blah, blah, eventual_worlds)
    # print(actions)
    return actions
