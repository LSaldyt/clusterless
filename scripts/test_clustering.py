from experiments.base import defaults

from clusterless.memory   import init_memory, sense_environment, communicate
from clusterless.scenario import from_unicode
from clusterless.clusters import clustering_baseline

from experiments.base     import defaults

from .test_communication import world_str

def run():
    s   = defaults().derive(size=9, view_size=2)
    map = from_unicode(world_str, s)

    memory = init_memory(map, s)
    senses = list(sense_environment(map, memory, s, 0))

    print(f'Map/Memories')
    map.full_render(senses)

    clustering_baseline(map, senses, memory, s)
    # communicate(memory, senses, s)

