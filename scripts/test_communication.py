from experiments.base import defaults

from clusterless.memory   import init_memory, sense_environment, communicate
from clusterless.scenario import from_unicode
from experiments.base     import defaults

world_str = '''
·········
·········
····□····
····ζ····
·········
····ξ····
····□····
·········
·········
'''

def run():
    s   = defaults().derive(size=9, view_size=2)
    map = from_unicode(world_str, s)

    memory = init_memory(map, s)
    senses = list(sense_environment(map, memory, s, 0))

    print(f'Map/Memories before communication')
    map.full_render(senses)

    communicate(memory, senses, s)
    senses = list(sense_environment(map, memory, s, 0))
    print(f'Map/Memories after communication')
    map.full_render(senses)
