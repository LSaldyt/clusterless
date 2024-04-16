import numpy as np
import numpy.typing as npt
from copy import deepcopy

from science import Settings

from . import utils

def render(mx, syms):
    ''' Render the environment in questionable unicode '''
    h, w = mx.shape
    def gen_syms():
        for i in range(h):
            for j in range(w):
                yield syms[mx[i, j]]
            yield '\n'
    return ''.join(gen_syms())

class Map():
    grid     : npt.ArrayLike
    coords   : npt.ArrayLike
    settings : Settings

    def __init__(self, s):
        ''' Create an initial environment from a multinomial distribution '''
        grid_shape = (s.size, s.size)
        self.grid  = s.gen.choice(np.arange(len(s.probs)), size=grid_shape, p=list(s.probs.values()))
        # Re-assign agents to unique numbers, e.g. [3, 3, 3] becomes [3, 4, 5]
        mask = self.grid == s.codes['agent']
        size = grid_shape[0]

        self.grid[mask]     = (np.arange(np.sum(mask)) + s.codes['agent'])
        self.coordinates    = utils.cartesian_product(np.arange(size), np.arange(size))
        self.settings       = s
        self._dont_deepcopy = {'coordinates', 'settings'} # Only deepcopy self.grid!

    def full_render(self, sense_input, s):
        rendered_views = [(' ' * s.view_size + '\n') * s.view_size]
        rendered_grids = [render(self.grid, s.symbols)]
        codes          = []
        for c, view, mem, coords in sense_input:
            rendered_views.append(render(view,     s.symbols))
            rendered_grids.append(render(mem.grid, s.symbols))
            codes.append(f'agent {s.symbols[c]}')
        descriptions   = [f'{name:<{s.size}}\n' for name in ['full'] + codes]
        print(utils.horizontal_join(rendered_grids))
        print(utils.horizontal_join(descriptions))
        print(utils.horizontal_join(rendered_views, join=' ' * (s.size - s.view_size + 1)))

    def set_at(self, coords, values):
        ''' Set grid by vectorized coordinates to new values '''
        self.grid[coords[:, 0], coords[:, 1]] = values # type: ignore

    def coords_of(self, mask):
        return self.coordinates[mask.reshape((np.prod(self.grid.shape),))] # type: ignore

    def mask(self, *keys, kind='or'):
        comb = np.logical_or if kind == 'or' else np.logical_and
        if len(keys) == 1: comb = lambda x : x
        return comb(*(self.grid == self.settings.codes[k] for k in keys))

    def count(self, *keys):
        return np.sum(self.mask(*keys)) # type: ignore

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in self._dont_deepcopy:
                setattr(result, k, deepcopy(v, memo))
        return result

