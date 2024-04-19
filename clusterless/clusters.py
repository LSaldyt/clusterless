
''' Ironic. Intended for implementing a baseline.
A transcription / pseudocode of my understanding of Jamison's algorithm '''

import numpy as np
from .utils import broadcast

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
    b_ids     = broadcast(ids, views_l.shape[1])              # IDs in the same shape as view adjacency list

    adj[b_ids, views_l] = 1 # Create the adjacency matrix in a single vectorized operation!!
    adj = adj[:a, :a]       # Cut away auxillary dimensions that we added to do vectorized operations

    for _ in range(s.cluster_rounds):
        single           = ids[clusters == -1]                          # Unassigned agents
        adj_clusters     = np.where(adj[single] == 1, clusters[ids], 0) # Adjacent clusters
        best_clusters    = np.max(adj_clusters, axis=-1)                # Order by best ID
        clusters[single] = best_clusters                                # Assign!

    return clusters

def clustering_baseline(env_map, senses, memory, s):
    clusters = form_clusters(env_map, senses, s)

    exit()

    # Once clusters are formed,
    # Share maps within the cluster
    # Leader computes MAR for whole cluster

    # The plan is shared with children, and then run
