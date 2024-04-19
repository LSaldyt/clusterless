import numpy as np

from science import Settings

from experiments.base import defaults

from experiments.base import defaults

from clusterless.memory   import init_memory, sense_environment, communicate
from clusterless.scenario import from_unicode
from experiments.base     import defaults
from clusterless.environment import Map, transition

world_str = '''
········
····★···
····□···
····ζ···
····ξ···
····□···
········
········
'''
def run():
    s   = defaults().derive(size=8, view_size=2)
    map = from_unicode(world_str, s)

    memory = init_memory(map, s)
    senses = list(sense_environment(map, memory, s, 0))

    print(f'Map/Memories before communication')
    map.full_render(senses)

    for action, desc in zip(s.action_space, ('up', 'down', 'left', 'right', 'stay')):
        if desc != 'down':
            continue
        print('-' * 80)
        print(action, desc, '?')
        copy_env_map = map.clone()
        acts   = np.stack([action, np.array([0, 0])])
        info   = transition(copy_env_map, acts, s)
        copy_env_map.color_render()
        senses = list(sense_environment(map, memory, s, 0))
        print(info)
        print('-' * 80)
