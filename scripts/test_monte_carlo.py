import numpy  as np
import pandas as pd

from science import Settings

from experiments.base import defaults

from clusterless.environment      import Map
from clusterless.memory           import init_memory, sense_environment
from clusterless.monte_carlo      import generate_worlds, emplace
from clusterless.policies.rollout import cost_to_go
from clusterless.policies         import available_policies

def run():
    s = defaults(Settings())
    s = s.derive(size=8,
                 probs=dict(empty=0.40, obstacle=0.25, goal=0.2, agent=0.15), 
                 n_worlds=128,
                 )

    horizon     = 32
    base_policy = available_policies['wave']

    print('Original\n')
    env_map = Map(s)
    env_map.color_render()

    a_info = env_map.agents_info

    memory = init_memory(env_map, s)
    senses = sense_environment(env_map, memory, s, 0)

    scores = np.zeros((a_info.n_agents, s.n_worlds))
    for i, world in enumerate(generate_worlds(s)):
        print('World (ϴ)\n')
        world.color_render()
        for j, (c, mem, _) in enumerate(senses):
            agent = s.symbols[c]
            print(f'Memory {agent}\n')
            mem.map.color_render()
            possible = emplace(mem, world, s)
            print(f'Emplaced {agent} ∈ ϴ\n')
            possible.color_render()

            score_d = cost_to_go(possible, base_policy, horizon, s)
            print(score_d)
            scores[i, j] = score_d

    agents = np.array(list(s.symbols))[a_info.codes]
    print(scores.shape)
    df = pd.DataFrame(scores.T, columns=agents)
    print(df)
    print(df.describe())

