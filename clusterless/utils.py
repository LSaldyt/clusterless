import functools, itertools
import numpy as np

@functools.cache
def box(size=3):
    assert size % 2 == 1, f'Odd-numbered box size required'
    factor   = (size - 1) // 2
    elements = np.arange(-factor, factor + 1)
    return np.array(list(itertools.product(elements, repeat=2)))

@functools.cache
def circle(radius=3):
    loc    = np.array([radius, radius])
    coords = grid_coordinates((radius + 1) * 2)
    diff   = coords - loc 
    dists  = np.sum(np.abs(diff), axis=-1)
    radius_mask = dists <= radius
    return diff[radius_mask]

def broadcast(a, n, axis=-1):
    return np.repeat(np.expand_dims(a, axis), n, axis)

def cartesian_product(*arrays):
    ''' Convenience. From https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points'''
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def grid_coordinates(n):
    return cartesian_product(np.arange(n), np.arange(n))

def horizontal_join(elements, join=' '):
    element_lines = (el.split('\n') for el in elements)
    rows = (join.join(row_items) for row_items in zip(*element_lines))
    return '\n'.join(rows)
