from science import Experiment, Registry, Settings

import string
import numpy as np
from clusterless.environment import create_grid, render, transition, get_agents, init_memory, sense_environment

class BaseExperiment(Experiment):
    def run(self, *args, **kwargs):
        self.ensure(**kwargs) # e.g. timesteps=8
        # self.log('info', dict(reward=0))
        # data = {'hello' : 'world'}
        # self.save_json('result', data)
        s = self.settings # Shorthand

        assert sum(s.probs.values()) < 1.0 + 1e-6

        grid, coordinates = create_grid(s.gen, (s.size, s.size), s.probs)

        agent_codes, agent_coords, n_agents = get_agents(grid, coordinates, s.codes)
        memory = init_memory(grid, agent_codes, s.codes)

        for t in range(s.timesteps):
            print(render(grid, s.symbols))
            agent_codes, agent_coords, n_agents = get_agents(grid, coordinates, s.codes)
            
            for c, mem, view in sense_environment(grid, memory, agent_codes, agent_coords, s.codes):
                print(f'View of agent {s.symbols[c]}')
                print(render(view, s.symbols))
                print(f'Memory of agent {s.symbols[c]}')
                print(render(mem, s.symbols))

            # TODO Call a policy function here
            action_indices = s.gen.integers(low=0, high=s.action_space.shape[0], size=(n_agents,)) 
            actions        = s.action_space[action_indices]

            info = transition(grid, actions, agent_coords, agent_codes, s.codes) # Important: Do transition at the end of the loop
            print(f'Step {t}: {info}')
            print('-' * 80)
