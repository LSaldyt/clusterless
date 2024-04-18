import numpy as np
import numpy.typing as npt
from copy import deepcopy
from collections import namedtuple
import hashlib

from science import Settings

import rich

from . import utils

def render(mx, syms, colors=None):
    ''' Render the environment in questionable unicode '''
    h, w = mx.shape
    def gen_syms():
        for i in range(h):
            for j in range(w):
                s = syms[mx[i, j]]
                if colors is not None:
                    color = colors[i * h + j]
                    if color:
                        yield f'[{color}]{s}[/{color}]'
                    else:
                        yield s
                else:
                    yield s
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
        rendered_grids = [self.color_render(show=False)]
        codes          = []
        for sense in sense_input:
            # TODO emplace circular views and render them if desired
            # rendered_views.append(render(sense.view,         s.symbols))
            rendered_grids.append(sense.memory.map.color_render(show=False))
            codes.append(f'agent {s.symbols[sense.code]}')
        descriptions   = [f'{name:<{s.size}}\n' for name in ['full'] + codes]
        rich.print(utils.horizontal_join(rendered_grids))
        rich.print(utils.horizontal_join(descriptions))
        # print(utils.horizontal_join(rendered_views, join=' ' * (s.size - s.view_size + 1)))

    def color_render(self, show=True):
        colors          = np.array(list(self.settings.colors))
        per_cell_colors = colors[self.grid]
        rendered = render(self.grid, self.settings.symbols, per_cell_colors.ravel())
        if show:
            rich.print(rendered)
        else:
            return rendered

    def render_grid(self):
        print(render(self.grid, self.settings.symbols))

    def set_at(self, coords, values):
        ''' Set grid by vectorized coordinates to new values 
            This function is IMPURE, so it sets a flag accordingly '''
        utils.set_at(self.grid, coords, values)
        self._inc_purity()

    def coords_of(self, mask):
        return self.coords[mask.reshape((np.prod(self.grid.shape),))] # type: ignore

    def mask(self, *keys, kind='or'):
        return utils.mask(self.grid, self.settings, *keys, kind=kind)

    def at(self, *keys, **kwargs):
        return self.coords_of(self.mask(*keys, **kwargs))

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

