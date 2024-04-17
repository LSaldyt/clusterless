import numpy as np

from science import Settings

from experiments.base import defaults

from clusterless.environment import Map, transition

def run():
    s = defaults(Settings())
    s = s.derive(size=9)
    shape = (s.size, s.size)

    env_map = Map(s, initial_grid=np.full(shape, s.codes['empty']))
    env_map.render_grid()

    env_map.grid[4, 4] = s.codes['obstacle'] # type: ignore
    env_map.grid[3, 4] = s.codes['agent']    # type: ignore

    env_map.render_grid()

    for action, desc in zip(s.action_space, ('up', 'down', 'left', 'right', 'stay')):
        print('-' * 80)
        print(action, desc, '?')
        copy_env_map = env_map.clone()
        info = transition(copy_env_map, np.expand_dims(action, 0), s)
        copy_env_map.render_grid()
        print(info)
        print('-' * 80)
