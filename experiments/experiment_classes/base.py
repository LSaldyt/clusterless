from science import Experiment
from clusterless.environment import create_grid, render, transition, get_agents, init_memory, sense_environment, full_render
from clusterless.policies    import available_policies

import numpy as np

class BaseExperiment(Experiment):
    def run(self, *args, **kwargs):
        self.ensure(**kwargs) # e.g. timesteps=8
        s      = self.settings     # Shorthand
        policy = available_policies[s.policy]

        grids = [(i_r,) + create_grid(s.gen, (s.size, s.size), s.probs)
                 for i_r in range(s.environment_samples)]

        for i_r, grid, coordinates in grids:
            score   = 0 # Number of goals achieved
            n_goals = np.sum(grid == s.codes['goal'])

            agent_codes, agent_coords, n_agents = get_agents(grid, coordinates, s.codes)
            memory = init_memory(grid, agent_codes, s.codes)

            for t in range(s.timesteps):
                agent_codes, agent_coords, n_agents = get_agents(grid, coordinates, s.codes)
                sense_input = list(sense_environment(grid, memory, agent_codes, agent_coords, s, t))

                full_render(grid, sense_input, s)

                actions = policy(s, n_agents, sense_input, coordinates)
                info    = transition(grid, actions, agent_coords, agent_codes, s.codes) # Important: Do transition at the end of the loop

                score += info['n_goals_achieved']
                info.update(score=score, env_index=i_r, n_goals=n_goals)
                print(f'Step {t}: {info}')
                self.log(f'run_{i_r}', info)
                print('-' * 80)
            assert score <= n_goals, f'{info}'
            self.log(f'summary', dict(score=score, percent=score/n_goals, env_index=i_r))
