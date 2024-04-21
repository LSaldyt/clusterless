from copy import deepcopy
import numpy as np

from ..monte_carlo import generate_worlds, emplace
from ..memory      import init_memory, sense_environment
from .utils   import empty_actions
from .rollout import egocentric_rollout

def monte_carlo_rollout(map, sense_info, input_memory, base_policy, t, s):
    # First calculate base policy for all agents
    actions = empty_actions(len(sense_info))
    for i, sense in enumerate(sense_info):
        world_acts = np.zeros((s.n_worlds, 2))
        world_vals = np.zeros(s.n_worlds)

        for w, world in enumerate(generate_worlds(s)):
            memory = deepcopy(input_memory)
            # print(f'Monte Carlo world {w}')
            world = emplace(sense.memory, world, s)
            # world.color_render()
            # If we hallucinate agents on the new world, recreate the base policy
            world_sense_info    = list(sense_environment(world, memory, s, t + 1))
            base_policy_actions = base_policy(world, world_sense_info, memory, base_policy, t + 1, s)

            world_memory = init_memory(world, s)
            world_memory.update(memory)
            # print(world_memory)

            codes = [sense.code for sense in world_sense_info]
            assert not (base_policy_actions == 0).all(), f'Base policy returned all stay actions, reject!!'
            # print(world.agents_info)
            # print(base_policy_actions.shape)
            act, val = egocentric_rollout(world, deepcopy(sense.memory), codes, world_memory,
                                          base_policy_actions, base_policy, 
                                          sense.code, t + 1, s, reveal=True) # IMPORTANT!!! Now we can reveal the map!!
            # print(f'Rollout action, value: {act} {val}')
            world_acts[w, :] = act
            world_vals[w]    = val
            # print(world_acts)
        # print(world_acts)
        # print(world_vals)

        best_w        = np.argmax(world_vals)
        actions[i, :] = world_acts[best_w, :]
        # print(best_w)
    # print(actions)
    return actions
