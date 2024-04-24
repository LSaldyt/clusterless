from clusterless.memory import communicate
from .wave import wave
from .multiagent_rollout import multiagent_rollout

def communication_multiagent_rollout(p, s):
    ''' The wavefront algorithm prefixed by a local communication step '''
    communicate(p.memory, p.sense_info, s)
    return multiagent_rollout(p, s)

