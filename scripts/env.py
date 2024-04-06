import string
import numpy as np
from clusterless.environment import create_grid, render, transition, get_agents, init_memory, sense_environment

def run(*args):
    ''' First three symbols are reserved (empty, obstacle, goal)
        Last two symbols are reserved (unseen, dead agent)
        All symbols inbetween are used for agent codes '''
    symbols   = '·□★' + 'ζξΞѯƔȣ☭' + '♔♕♖♗♘♙♚♛♜♝♞♟' + string.ascii_lowercase + string.ascii_uppercase + '☺☆?☠'
    size      = 8
    timesteps = 128
    gen       = np.random.default_rng(2024)
    probs     = dict(empty=0.5, obstacle=0.35, goal=0.1, agent=0.05) # Order matters! Agents come last
    codes     = {**{k : i for i, k in enumerate(probs.keys())},
                 **{'dead'      : -1,
                    'unseen'    : -2,
                    'old_goal'  : -3,
                    'old_agent' : -4}}

    assert sum(probs.values()) < 1.0 + 1e-6

    grid, coordinates = create_grid(gen, (size, size), probs)
    action_space      = np.array([[0, 0], [0, 1], [1, 0], [-1, 0], [0, -1]])

    agent_codes, agent_coords, n_agents = get_agents(grid, coordinates, codes)
    memory = init_memory(grid, agent_codes, codes)

    for t in range(timesteps):
        print(render(grid, symbols))
        agent_codes, agent_coords, n_agents = get_agents(grid, coordinates, codes)
        
        for c, mem, view in sense_environment(grid, memory, agent_codes, agent_coords, codes):
            print(f'View of agent {symbols[c]}')
            print(render(view, symbols))
            print(f'Memory of agent {symbols[c]}')
            print(render(mem, symbols))

        action_indices = gen.integers(low=0, high=action_space.shape[0], size=(n_agents,)) # type: ignore (Silence! My code is right!)
        actions        = action_space[action_indices]

        info = transition(grid, actions, agent_coords, agent_codes, codes) # Important: Do transition at the end of the loop
        print(f'Step {t}: ' + ('goals acheived: {n_goals_achieved}, '
                               'collisions: {n_collisions_obstacle} (obstacle) {n_collisions_agents} (agent)'
                   ).format(**info))
        print('-' * 80)
