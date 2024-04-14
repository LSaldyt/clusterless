from science import Experiment
from clusterless.environment import create_grid, render, transition, get_agents, init_memory, sense_environment
from clusterless.policies    import available_policies

class BaseExperiment(Experiment):
    def run(self, *args, **kwargs):
        self.ensure(**kwargs) # e.g. timesteps=8
        s      = self.settings     # Shorthand
        policy = available_policies[s.policy]

        for i_r in range(s.environment_samples):
            score = 0 # Number of goals achieved

            grid, coordinates = create_grid(s.gen, (s.size, s.size), s.probs)

            agent_codes, agent_coords, n_agents = get_agents(grid, coordinates, s.codes)
            memory = init_memory(grid, agent_codes, s.codes)

            for t in range(s.timesteps):
                print(render(grid, s.symbols))
                agent_codes, agent_coords, n_agents = get_agents(grid, coordinates, s.codes)
                
                sense_input = list(sense_environment(grid, memory, agent_codes, agent_coords, s.codes, t))

                for c, mem, view in sense_input:
                    print(f'View of agent {s.symbols[c]}')
                    print(render(view, s.symbols))
                    print(f'Memory of agent {s.symbols[c]}')
                    print(render(mem, s.symbols))

                actions = policy(s, n_agents, sense_input)

                info = transition(grid, actions, agent_coords, agent_codes, s.codes) # Important: Do transition at the end of the loop
                score += info['n_goals_achieved']
                info.update(score=score, env_index=i_r)
                print(f'Step {t}: {info}')
                self.log(f'run_{i_r}', info)
                print('-' * 80)
            self.log(f'summary', dict(score=score, env_index=i_r))
