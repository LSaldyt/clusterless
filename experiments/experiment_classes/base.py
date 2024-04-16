from science import Experiment
from clusterless.environment import Map, simulate 
from clusterless.policies    import available_policies

import numpy as np

class BaseExperiment(Experiment):
    def run(self, *args, **kwargs):
        self.ensure(**kwargs) # e.g. timesteps=8
        s      = self.settings     # Shorthand
        policy = available_policies[s.policy]

        grids = [(i_r, Map(s))
                 for i_r in range(s.environment_samples)]

        for i_r, grid in grids:
            stats = simulate(grid, policy, s, do_render=True)
            stats.update(environment_index=i_r)
            self.log(f'summary', stats)
