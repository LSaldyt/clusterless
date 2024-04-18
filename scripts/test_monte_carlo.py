import numpy as np

from science import Settings

from experiments.base import defaults

from clusterless.environment import Map, transition
from clusterless.memory      import init_memory, sense_environment
from clusterless.monte_carlo import generate_worlds, emplace

def run():
    s = defaults(Settings())
    s = s.derive(size=8,
                 probs=dict(empty=0.40, obstacle=0.25, goal=0.2, agent=0.15), 
                 )

    env_map = Map(s)
    env_map.color_render()

    memory = init_memory(env_map, s)
    senses = sense_environment(env_map, memory, s, 0)

    for world in generate_worlds(s):
        world.color_render()
        for c, mem, _ in senses:
            mem.map.color_render()
            possible = emplace(mem, world, s)
            possible.color_render()
        exit()
        break
