from clusterless.utils import circle
from clusterless.map   import render

import numpy as np

def run(s = 11, p = 5, r = 1): 
    grid = np.zeros((s, s), dtype=np.int32)
    loc  = np.array([p, p], dtype=np.int32)
    off  = circle(radius=r)
    locs = loc + off

    grid[locs[:, 0], locs[:, 1]] = 1

    print(render(grid, '.*'))
