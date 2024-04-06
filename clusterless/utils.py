import functools, itertools
import numpy as np

@functools.cache
def box(size=3):
    assert size % 2 == 1, f'Odd-numbered box size required'
    factor   = (size - 1) // 2
    elements = np.arange(-factor, factor + 1)
    return np.array(list(itertools.product(elements, repeat=2)))

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
