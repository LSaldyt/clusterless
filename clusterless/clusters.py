
''' Ironic. Intended for implementing a baseline.
A transcription / pseudocode of my understanding of Jamison's algorithm '''

import numpy as np
import itertools

from .utils  import broadcast 
from .memory import merge_memory, sense_environment
from .environment import transition
from .policies.multiagent_rollout import multiagent_rollout
from .policies.random             import random as random_policy

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

    c_round = 0
    for c_round in range(s.cluster_rounds): # Alternatively stop when (best_clusters == -1).all()
        single           = ids[clusters == -1]                           # Unassigned agents
        adj_clusters     = np.where(adj[single] == 1, clusters[ids], -1) # Join adjacent clusters
        best_clusters    = np.max(adj_clusters, axis=-1)                 # Order by best ID
        clusters[single] = best_clusters                                 # Assign!
        if (best_clusters == -1).all():
            break
        # TODO Limit cluster size

    leader[ids[clusters == -1]] = True # Unassigned agents are their own "leaders"
    # This must be a python list, since the elements are different sizes
    return [(l, a_info.codes[clusters == c] if c > 0 else l) # Filter singleton clusters and return leaders
            for c, l in zip(clusters[leader], codes[leader])], c_round

def share_memory(cluster, memory, s):
    ''' Share maps within a single cluster. Doesn't obey the tree. '''
    rounds = 0
    for a, b in itertools.combinations(cluster, 2): # O(n^2)
        print(f'Sharing {a} → {b}')
        merge_memory(memory[a], memory[b], s)
        rounds += 1
    return rounds

def cluster_plan(cluster, local, memory, base_policy, s, t):
    ''' Run MAR within a cluster until all goals are reached or we timeout '''
    env_map = local.memory.map
    env_map.color_render()
    for _ in range(s.cluster_plan_rounds_max):
        senses     = [sense for sense in sense_environment(env_map, memory, s, t)
                      if sense.code in set(cluster)] # Filter non-cluster members
        env_map    = senses[0].memory.map.clone()    # As transition will mutate this
        env_map.set_at(env_map.at('unseen'), s.codes['obstacle'])
        if env_map.count('goal') == 0:
            break
        if cluster.shape[0] == 1: # Singleton clusters (no goals)
            actions = random_policy(env_map, senses, base_policy, t, s)
        else:
            actions = multiagent_rollout(env_map, senses, base_policy, t, s)
        ind        = np.array([sense.code - 3 for sense in senses])
        empty      = np.zeros((env_map.agents_info.n_agents, 2), dtype=np.int32)
        empty[ind] = actions
        transition(env_map, empty, s)
        env_map.color_render()
        yield ind, actions

def clustered_multiagent_rollout(env_map, input_senses, memory, base_policy, s, t):
    cluster_plans = []
    total_share_rounds = 0
    clusters, cluster_rounds = form_clusters(env_map, input_senses, s)
    input_senses = {s.code : s for s in input_senses}
    for leader, cluster in clusters:
        total_share_rounds += share_memory(cluster, memory, s) 
        print(f'Cluster: ({" ".join(s.symbols[c] for c in cluster)}), leader {s.symbols[leader[0]]}')
        # Leader computes MAR for whole cluster
        local = input_senses[leader[0]] # Leader's sensory information
        # Produce a *plan* as a sequence of actions while any leader still sees a goal
        cluster_plans.append(list(cluster_plan(cluster, local, memory, base_policy, s, t)))

    empty_acts   = np.zeros((env_map.agents_info.n_agents, 2), dtype=np.int32)
    max_plan_len = max(len(p) for p in cluster_plans)

    queue = [empty_acts.copy() for _ in range(max_plan_len)] # All agents wait for all plans to finish (across clusters, by bound)
    for c_plan in cluster_plans:
        for t, (ind, act) in enumerate(c_plan):
            queue[t][ind] = act # Emplace plans into correct timestep and agent indices
    if s.queue_cluster_actions:
        total_share_rounds 
        pad   = [empty_acts.copy() for _ in range(cluster_rounds + total_share_rounds)]
        queue = pad + queue
    return queue
