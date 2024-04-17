from science import Settings, Experiment
from .experiment_classes import BaseExperiment
from clusterless.policies import available_policies

import string
import numpy as np

def defaults(shared):
    s = shared.derive(
        size=8,
        symbols = '·□★' + ('ζξΞѯƔȣ☭' + '♔♕♖♗♘♙♚♛♜♝♞♟' + string.ascii_lowercase + string.ascii_uppercase)*20 + '?☠',
        gen     = np.random.default_rng(2024),
        # probs   = dict(empty=0.54, obstacle=0.35, goal=0.1, agent=0.01), # Order matters! Agents come last
        # probs   = dict(empty=0.54, obstacle=0.35, goal=0.1, agent=0.01), # Order matters! Agents come last
        # probs   = dict(empty=0.54 + .35, obstacle=0.0, goal=0.1, agent=0.01), # Order matters! Agents come last
        probs   = dict(empty=0.53, obstacle=0.25, goal=0.2, agent=0.02), # Order matters! Agents come last

        discount=0.95,
        truncated_timesteps=16,
        timesteps = 128,
        environment_samples=8,
        view_size=3,
        policy='rollout',
        base_policy='nearest',
        do_render=True,

        obstacle_cost=1.0, # Just some bruises
        death_cost=10.0,   # One death is worth ten goals, arbitrarily. Our agents are cheap.

        view_type='circle',

        action_space = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]]),
            
        # Debugging/analysis options
        selected_env=-1,
        single_agent=False,

        debug=False,
        debug_trace_depth=4,
        detect_cycles=True,

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
