from clusterless.memory import communicate
from .wave import wave

def communication_wave(p, s):
    ''' The wavefront algorithm prefixed by a local communication step '''
    communicate(p.memory, p.sense_info, s)
    return wave(p, s)

