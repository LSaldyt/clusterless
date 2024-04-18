import numpy as np

from functools import cache

from .map import Map

@cache
def inverse_symbols(syms):
    return {s : i for i, s in enumerate(syms)}

def encode_unicode(char, s):
    return inverse_symbols(s.symbols)[char]

def from_unicode(map_str, s):
    ''' Assumes that map_str is a newline separated grid of unicode characters '''
    lines = [l for l in map_str.splitlines() if l.strip() != '']
    w, h  = len(lines[0]), len(lines[1])
    grid  = np.zeros((w, h), dtype=np.int32)
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            grid[i, j] = encode_unicode(char, s)
    return Map(s, grid)
