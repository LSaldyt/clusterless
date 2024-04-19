import numpy  as np
import pandas as pd

from science import Settings
from rich.progress import track

from experiments.base import defaults

from clusterless.environment      import Map, simulate
from clusterless.memory           import init_memory, sense_environment
from clusterless.monte_carlo      import generate_worlds, emplace
from clusterless.policies.rollout import cost_to_go
from clusterless.policies         import available_policies

def run():
    s = defaults(Settings())
    s = s.derive(size=16,
                 probs=dict(empty=0.54, obstacle=0.25, goal=0.2, agent=0.01), 
                 n_worlds=128)

    horizon     = 256
    base_policy = available_policies['wave']

    print('Original\n')
    env_map = Map(s) # Randomly generated map. Ground truth
    env_map.color_render()

    a_info = env_map.agents_info

    memory = init_memory(env_map, s)
    senses = list(sense_environment(env_map, memory, s, 0))

    scores = np.zeros((a_info.n_agents, s.n_worlds))
    for i, world in track(enumerate(generate_worlds(s)),
                          description='Dreaming of other worlds to do glorious rollout in..',
                          total=s.n_worlds):
        print('World (ϴ)\n')
        world.color_render()
        for j, sense in enumerate(senses):
            agent = s.symbols[sense.code]
            print(f'Memory {agent}\n')
            sense.memory.map.color_render()
            possible = emplace(sense.memory, world, s)
            print(f'Emplaced {agent} ∈ ϴ\n')
            possible.color_render()

            score_d = cost_to_go(possible, base_policy, horizon, s)['score']
            print(score_d)
            scores[j, i] = score_d

    agents = np.array(list(s.symbols))[a_info.codes]
    print(scores.shape)
    df = pd.DataFrame(scores.T, columns=agents)
    print(df)
    print(df.describe())

    print('Upper bound on expectation:')
    print(senses[0].memory.map.count('unseen'), '*', s.probs['goal'])
    print(senses[0].memory.map.count('unseen') * s.probs['goal'])
    print('Ground truth for our world')
    env_map.color_render()
    print(simulate(env_map, base_policy, base_policy, horizon, 0, s, do_render=False))
