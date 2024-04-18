from science import Experiment
from clusterless.environment import Map, simulate 
from clusterless.policies    import available_policies

from rich.progress import Progress

class BaseExperiment(Experiment):
    def run(self, *args, **kwargs):
        self.ensure(**kwargs) # e.g. timesteps=8
        s           = self.settings     # Shorthand
        policy      = available_policies[s.policy]
        base_policy = available_policies[s.base_policy]

        maps = [(i_r, Map(s))
                 for i_r in range(s.environment_samples)]

        with Progress() as progress:
            map_task = progress.add_task('Maps', total=s.environment_samples)
            env_task = None

            for i_r, map in maps:
                if env_task is not None:
                    progress.remove_task(env_task)
                env_task = progress.add_task(f'Env ({s.policy})', total=s.timesteps)
                if s.selected_env != -1:
                    if i_r != s.selected_env:
                        continue
                if s.single_agent and map.agents_info.n_agents != 1:
                    continue
                stats = simulate(map, policy, base_policy, s.timesteps, i_r, s, do_render=s.do_render, progress=progress, task=env_task)
                stats.update(environment_index=i_r)
                self.log(f'summary', stats)

                progress.update(map_task, advance=1)
