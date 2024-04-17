import numpy as np
import numpy.typing as npt
from copy import deepcopy
from collections import namedtuple
import hashlib

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

AgentsInfo = namedtuple('AgentsInfo', ['codes', 'coords', 'n_agents'])

class Map():
    grid     : npt.ArrayLike
    coords   : npt.ArrayLike
    settings : Settings
    purity   : int

    def __init__(self, s, initial_grid=None):
        ''' Create an initial environment from a multinomial distribution '''
        if initial_grid is None:
            grid_shape = (s.size, s.size)
            self.grid  = s.gen.choice(np.arange(len(s.probs)), size=grid_shape, p=list(s.probs.values()))
            # Re-assign agents to unique numbers, e.g. [3, 3, 3] becomes [3, 4, 5]
            mask = self.grid == s.codes['agent']
            self.grid[mask] = (np.arange(np.sum(mask)) + s.codes['agent'])
        else:
            self.grid = initial_grid # type: ignore

        self.coords         = utils.cartesian_product(np.arange(s.size), np.arange(s.size))
        self.settings       = s
        self.purity         = 0 # Integer that increments when grid is modified
        self.cache          = dict()
        self._dont_deepcopy = {'coords', 'settings'} # Only deepcopy self.grid!

    def clone(self):
        child = deepcopy(self)
        child._inc_purity()
        return child

    def full_render(self, sense_input):
        s = self.settings
        rendered_views = [(' ' * s.view_size + '\n') * s.view_size]
        rendered_grids = [render(self.grid, s.symbols)]
        codes          = []
        for c, mem, coords in sense_input:
            # rendered_views.append(render(view,         s.symbols))
            rendered_grids.append(render(mem.map.grid, s.symbols))
            codes.append(f'agent {s.symbols[c]}')
        descriptions   = [f'{name:<{s.size}}\n' for name in ['full'] + codes]
        print(utils.horizontal_join(rendered_grids))
        print(utils.horizontal_join(descriptions))
        # print(utils.horizontal_join(rendered_views, join=' ' * (s.size - s.view_size + 1)))

    def render_grid(self):
        print(render(self.grid, self.settings.symbols))

    def set_at(self, coords, values):
        ''' Set grid by vectorized coordinates to new values 
            This function is IMPURE, so it sets a flag accordingly '''
        self.grid[coords[:, 0], coords[:, 1]] = values # type: ignore
        self._inc_purity()

    def coords_of(self, mask):
        return self.coords[mask.reshape((np.prod(self.grid.shape),))] # type: ignore

    def mask(self, *keys, kind='or'):
        comb = np.logical_or if kind == 'or' else np.logical_and
        if len(keys) == 1: comb = lambda x : x
        return comb(*(self.grid == self.settings.codes[k] for k in keys))

    def count(self, *keys):
        mask = self.mask(*keys)
        return np.sum(mask) # type: ignore

    @property
    def agents_info(self):
        key = ('a_info', self.purity)
        if key not in self.cache:
            mask     = self.grid >= self.settings.codes['agent']
            codes    = self.grid[mask] # type: ignore
            coords   = self.coords_of(mask)
            n_agents = np.sum(mask)
            self.cache[key] = AgentsInfo(codes, coords, n_agents)
        return self.cache[key]

    def hash(self):
        return hashlib.md5(self.grid).hexdigest()

    def _inc_purity(self):
        self.purity += 1
        self.cache = dict()

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
            else:
                setattr(result, k, v)
        return result

