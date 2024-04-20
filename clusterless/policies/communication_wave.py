from clusterless.memory import communicate
from .wave import wave

def communication_wave(map, sense_info, memory, base_policy, t, s):
    ''' The wavefront algorithm prefixed by a local communication step '''
    communicate(memory, sense_info, s)
    return wave(map, sense_info, memory, base_policy, t, s)

