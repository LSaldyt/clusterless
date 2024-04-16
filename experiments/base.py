from science import Settings, Experiment
from .experiment_classes import BaseExperiment
from clusterless.policies import available_policies

import string
import numpy as np

def define_experiments(registry):
    # Default/common experiment settings
    ''' For symbols: 
        First three symbols are reserved (empty, obstacle, goal)
        Last two symbols are reserved (unseen, dead agent)
        All symbols inbetween are used for agent codes '''
    s = registry.shared.derive(
        symbols = '·□★' + 'ζξΞѯƔȣ☭' + '♔♕♖♗♘♙♚♛♜♝♞♟' + string.ascii_lowercase + string.ascii_uppercase + '?☠',
        gen     = np.random.default_rng(2024),
        probs   = dict(empty=0.5, obstacle=0.35, goal=0.1, agent=0.05), # Order matters! Agents come last
        timesteps = 128,
        environment_samples=128,
        view_size=3,
        policy='nearest',

        action_space = np.array([[0, 0], [0, 1], [1, 0], [-1, 0], [0, -1]])
        )
    s.update(codes={
                 **{k : i for i, k in enumerate(s.probs.keys())},
                 **{'dead'      : -1,
                    'unseen'    : -2}})

    sizes = [8, 16, 32]
    return [BaseExperiment(f'env_{size}', s.derive(size=size))
            for size in sizes]
