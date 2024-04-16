from science import Experiment
from clusterless.environment import Map, simulate 
from clusterless.policies    import available_policies

class BaseExperiment(Experiment):
    def run(self, *args, **kwargs):
        self.ensure(**kwargs) # e.g. timesteps=8
        s           = self.settings     # Shorthand
        policy      = available_policies[s.policy]
        base_policy = available_policies[s.base_policy]

        maps = [(i_r, Map(s))
                 for i_r in range(s.environment_samples)]

        for i_r, map in maps:
            stats = simulate(map, policy, base_policy, s.timesteps, s)
            stats.update(environment_index=i_r)
            self.log(f'summary', stats)
