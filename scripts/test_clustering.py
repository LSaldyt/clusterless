from clusterless.policies.wave import wave
from experiments.base import defaults

from clusterless.memory      import init_memory, sense_environment, communicate
from clusterless.scenario    import from_unicode
from clusterless.clusters    import clustered_multiagent_rollout 
from clusterless.environment import transition

from experiments.base     import defaults

from .test_communication import world_str

# ΞѯƔȣ☭

world_str = '''
Ξ·······ѯ
····★····
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

    print(f'Map/Memories')
    map.full_render(senses)

    base_policy = wave

    actions = clustered_multiagent_rollout(map, senses, memory, base_policy, s, 0)
    try:
        for action in actions:
            # print(action)
            transition(map, action, s)
            map.color_render()
    except:
        pass

