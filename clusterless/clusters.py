
''' Ironic. Intended for implementing a baseline.
A transcription / pseudocode of my understanding of Jamison's algorithm '''

import numpy as np
import itertools
from .utils  import broadcast 
from .memory import merge_memory
from .policies.multiagent_rollout import multiagent_rollout

def form_clusters(env_map, senses, s):
    a_info    = env_map.agents_info
    a         = a_info.n_agents
    ids       = np.arange(a)
    views     = np.stack([s.view for s in senses])
    goal_mask = np.sum(views == s.codes['goal'], axis=-1)     # Leaders are those than can see goals
    codes     = np.expand_dims(a_info.codes[ids], -1)         # Grid codes [3, infinity)
    id_prior  = np.minimum(np.sum(views > codes, axis=-1), 1) # Unless more important agents see goals too
    leader    = np.where(goal_mask, id_prior > 0, False)      # Resolve leaders 
    clusters  = np.where(leader, ids, -1)                     # Set initial cluster IDs to leader agent IDs
    adj       = np.zeros((a + 1, a + 1), dtype=np.int32)      # Empty adjacency matrix (a, a)
    views_l   = np.where(views >= s.codes['agent'],           # View adjacency list (a, v)
                         views  - s.codes['agent'], -1)       # Adjusted to be indices rather than integer codes
    views_l   = views_l[a_info.codes - 3]                     # Views
    b_ids     = broadcast(ids, views_l.shape[1])              # IDs in the same shape as view adjacency list


    adj[b_ids, views_l] = np.where(views_l >= 0, 1, 0) # Create the adjacency matrix in a single vectorized operation!!
    adj = adj[:a, :a]       # Cut away auxillary dimensions that we added to do vectorized operations

    for _ in range(s.cluster_rounds): # Alternatively stop when (best_clusters == -1).all()
        single           = ids[clusters == -1]                           # Unassigned agents
        adj_clusters     = np.where(adj[single] == 1, clusters[ids], -1) # Adjacent clusters
        best_clusters    = np.max(adj_clusters, axis=-1)                 # Order by best ID
        clusters[single] = best_clusters                                 # Assign!
        if (best_clusters == -1).all():
            break

    leader[ids[clusters == -1]] = True # Unassigned agents are their own "leaders"
    # This must be a python list, since the elements are different sizes
    return [(l, a_info.codes[clusters == c] if c > 0 else l) # Filter singleton clusters
            for c, l in zip(clusters[leader], codes[leader])]            # And return leaders

def share_memory(cluster, memory, s):
    ''' Share maps within a single cluster. Doesn't obey the tree. '''
    for a, b in itertools.combinations(cluster, 2): # O(n^2)
        merge_memory(memory[a], memory[b], s)

def clustering_baseline(env_map, senses, memory, s):
    clusters = form_clusters(env_map, senses, s)
    for leader, cluster in clusters:
        share_memory(cluster, memory, s) 
        # Leader computes MAR for whole cluster
        multiagent_rollout()

    exit()


# def multiagent_rollout(map, sense_info, base_policy, t, s):
    # The plan is shared with children, and then run
