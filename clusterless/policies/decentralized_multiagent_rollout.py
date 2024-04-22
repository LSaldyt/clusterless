from ..clusters import clustered_multiagent_rollout
from .wave      import wave

from copy import deepcopy

global_dmar_queue = []

def decentralized_multiagent_rollout(p, s):
    global global_dmar_queue # Yikes!!

    if len(global_dmar_queue) == 0:
        print(global_dmar_queue)
        global_dmar_queue = clustered_multiagent_rollout(deepcopy(p), s)
    if len(global_dmar_queue) == 0:
        print(global_dmar_queue)
        global_dmar_queue = [wave(deepcopy(p), s)]
    assert len(global_dmar_queue) > 0, 'WTF'
    # Only the first plan element (cache/queue plan elements) 
    action = global_dmar_queue.pop(0)
    if action.shape[0] != p.map.agents_info.n_agents: # If an agent has died
        global_dmar_queue = []                      # Reset the global queue
        return decentralized_multiagent_rollout(p, s)
    else:
        return action
