from clusterless.scenario import from_unicode
from experiments.base     import defaults

world_str = '''
·········
·········
·········
····□····
····ζ····
·········
····ξ····
····□····
·········
'''

def run():
    print(world_str)

    s   = defaults()
    map = from_unicode(world_str, s)
    map.color_render()

