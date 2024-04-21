from clusterless.memory import communicate
from .multiagent_rollout import multiagent_rollout

def anarchist_multiagent_rollout(map, sense_info, memory, base_policy, t, s):
    ''' Rollout prefixed by a local communication step '''
    communicate(memory, sense_info, s)
    return multiagent_rollout(map, sense_info, memory, base_policy, t, s)

