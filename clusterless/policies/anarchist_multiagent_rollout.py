from clusterless.memory import communicate
from .multiagent_rollout import multiagent_rollout

def anarchist_multiagent_rollout(p, s):
    ''' Rollout prefixed by a local communication step '''
    communicate(p.memory, p.sense_info, s)
    return multiagent_rollout(p, s)

