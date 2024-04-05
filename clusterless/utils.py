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
