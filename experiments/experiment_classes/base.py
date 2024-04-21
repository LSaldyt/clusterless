from science import Experiment
from clusterless.environment import Map, simulate 
from clusterless.policies    import available_policies

from rich.progress import Progress

def replace_task(progress, prev, *args, **kwargs):
    if prev is not None:
        progress.remove_task(prev)
    return progress.add_task(*args, **kwargs)

class BaseExperiment(Experiment):
    def run(self, *args, **kwargs):
        self.ensure(**kwargs, log_fn=self.log) # e.g. timesteps=8
        s        = self.settings # Shorthand
        # We can run multiple policies in one experiment
        if isinstance(s.policy, tuple):
            policies = s.policy
        else:
            policies = [s.policy]

        maps = [(i_r, Map(s))
                 for i_r in range(s.environment_samples)]

        with Progress() as progress:
            map_task = progress.add_task(f'Maps ({s.environment_samples})', total=s.environment_samples)
            env_task = None
            pol_task = None

            for i_r, map in maps:
                env_task = replace_task(progress, env_task, f'Env {i_r}', total=s.timesteps)
                for policy_key in policies:
                    pol_task    = replace_task(progress, pol_task, f'Policy {policy_key}', total=len(policies))
                    policy      = available_policies[policy_key]
                    base_policy = available_policies[s.base_policy]
                    if s.selected_env != -1:
                        if i_r != s.selected_env:
                            continue
                    if s.single_agent and map.agents_info.n_agents != 1:
                        continue
                    if s.exact_n_agents > 0 and s.exact_n_agents != map.agents_info.n_agents:
                        continue
                    stats = simulate(map, policy, base_policy, s.timesteps, s, do_render=s.do_render, 
                                     progress=progress, task=env_task, log_fn=self.log, extra=dict(policy=policy_key, env=i_r))
                    stats.update(environment_index=i_r, policy=policy_key)
                    self.log(f'summary', stats)
                    progress.update(pol_task, advance=1)
                progress.update(map_task, advance=1)
        print(f'Results saved in {self.instance_dir}')
