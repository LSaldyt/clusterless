from science import Settings, Experiment
from .experiment_classes import BaseExperiment
from clusterless.policies import available_policies

import string
import numpy as np

def defaults(shared=Settings()):
    agent_colors = ['red']
    s = shared.derive(
        # TODO: Just use further unicode symbols and try not to repeat agent characters. Only matters for large grids and frequent agent spawns
        symbols = '·□★' + ('ζξΞѯƔȣ☭' + '♔♕♖♗♘♙♚♛♜♝♞♟' + string.ascii_lowercase + string.ascii_uppercase) + '?☠', 
        colors  = ['', '', 'gold3'] + agent_colors * 100 + ['', 'dark_red'],

        # Environment settings
        gen     = np.random.default_rng(2024),
        probs   = dict(empty=0.53, obstacle=0.25, goal=0.2, agent=0.02), # Order matters! Agents come last

        # Scale parameters
        size                = 8,
        truncated_timesteps = 16,
        timesteps           = 128,
        environment_samples = 8,

        # Policy
        policy='rollout',
        base_policy='wave',

        # Reward settings
        discount=0.95,
        obstacle_cost=1.0, # Just some bruises
        death_cost=10.0,   # One death is worth ten goals, arbitrarily. Our agents are cheap.

        # Actions
        action_space = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]]),
        action_words = ('up', 'down', 'left', 'right', 'stay'),
        view_size=3,
        view_type='circle',

        # Monte carlo
        n_worlds=16, 

        # Clustering
        cluster_rounds=8,
        cluster_plan_rounds_max=32,
        queue_cluster_actions=False, # If we count communication and cluster formation as empty actions
            
        # Debugging/analysis options
        selected_env=-1,
        single_agent=False,

        debug=False,
        debug_trace_depth=4,
        debug_alt_nearest=True,
        detect_cycles=True,
        do_render=True,

        )
    s.update(codes={
                 **{k : i for i, k in enumerate(s.probs.keys())},
                 **{'dead'      : -1,
                    'unseen'    : -2}})
    return s

def define_experiments(registry):
    # Default/common experiment settings
    ''' For symbols: 
        First three symbols are reserved (empty, obstacle, goal)
        Last two symbols are reserved (unseen, dead agent)
        All symbols inbetween are used for agent codes '''
    s = defaults(registry.shared)
    sizes = [8, 16, 32]
    return [BaseExperiment(f'env_{size}', s.derive(size=size))
            for size in sizes]
