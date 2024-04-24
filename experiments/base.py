from science import Settings
from .experiment_classes import BaseExperiment

import string
import numpy as np

arrows    = (11160, 11167)
stars     = (10016, 10059)
cross     = (10013, 10016)
chess     = (9812,  9823)
cards     = (9824,  9831)
music     = (9833,  9839)
astrology = (9791,  9811)
political = (9763,  9775)
balls_1   = (9312,  9331)
balls_2   = (9398,  9471)
greek     = (7596,  7602)
# dist      = [chr(i) for i in (9617, 9618, 9619, 9607)]
dist      = [chr(i) for i in (0x25a1, 0x25a7, 0x25a8, 0x25a9, 0x25a4, 0x25a5, 0x25a6, 0x25a0)]

geom = (9600, 9726)
misc = '✊⛽⛧ ⚰⚜⚕☸☘ ■ ▩ ⏳⏲⏱⏰⎈∾⁕⚾⚿⛀⛁⛂⛃⛄⛅⛆⛇⛈⛏'
curs = 'ℋℐℒℓ℘ℛℨ'

au = list(chr(a) for a in range(arrows[0], arrows[0] + 4))
action_unicode = [au[i] for i in (1, 3, 0, 2)] + ['⏳']

def defaults(shared=Settings()):
    agent_colors = ['red']
    # Default/common experiment settings
    ''' For symbols: 
        First three symbols are reserved (empty, obstacle, goal)
        Last two symbols are reserved (unseen, dead agent)
        All symbols inbetween are used for agent codes '''
    s = shared.derive(
        # TODO: Just use further unicode symbols and try not to repeat agent characters. Only matters for large grids and frequent agent spawns
        symbols = '·□★' + ('ζξΞѯƔȣ☭' + '♔♕♖♗♘♙♚♛♜♝♞♟' + string.ascii_lowercase + string.ascii_uppercase + misc + curs) + '?☠', 
        colors  = ['', '', 'gold3'] + agent_colors * 100 + ['', 'dark_red'],

        time_symbols = ' ⁕',
        dist_symbols = dist,

        # Environment settings
        gen     = np.random.default_rng(2024),
        probs   = dict(empty=0.53, obstacle=0.25, goal=0.2, agent=0.02), # Order matters! Agents come last
        goal_movement = 'static',

        # Scale parameters
        size                = 8,
        truncated_timesteps = 256,
        timesteps           = 256,
        environment_samples = 128,

        # Policy
        policy='multiagent_rollout',
        base_policy='wave',

        # Reward settings
        discount=0.95,
        obstacle_cost=1.0, # Just some bruises
        death_cost=10.0,   # One death is worth ten goals, arbitrarily. Our agents are cheap.

        # Actions
        action_space = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]]),
        action_words = ('up', 'down', 'left', 'right', 'stay'),
        view_size=3,
        view_type='circle',

        # Monte carlo
        n_worlds=16, 
        no_monte_carlo_agents=True, # Do not hallucinate additional agents outside of belief-state induced predictions

        # Belief
        belief_max_friends=7,
        belief_threshold=0.1,
        track_beliefs=True, # Could be broken out into a per-policy setting
        belief_update_egocentric=True, # If egocentric (b0) beliefs are updated in monte carlo
        russian_trust_factor=1.0,      # Multiplier for our own beliefs over others

        # Clustering
        cluster_rounds=8,
        cluster_plan_rounds_max=32,
        cluster_max_depth=9,
        cluster_plan_duplicates_only=True, # Remove duplicates only when simulating from an agent's memory
        queue_cluster_actions=False,       # If we count communication and cluster formation as empty actions

        rollout_duplicates_only=True, # Remove duplicates only when simulating from an agent's memory
        # Alternatively, remove all agents that aren't in the current timestep, e.g. forget about people out of view range
            
        # Debugging/analysis options
        selected_env=-1,
        single_agent=False,

        render_time=False,

        debug=False,
        debug_trace_depth=4,
        debug_alt_nearest=True,
        detect_cycles=False,
        do_render=True,
        exact_n_agents=-1

        )
    s.update(codes={
                 **{k : i for i, k in enumerate(s.probs.keys())},
                 **{'dead'      : -1,
                    'unseen'    : -2}},
             action_lookup = {str(tuple(a)) : action_unicode[i] for i, a in enumerate(s.action_space)},
             action_number_lookup = {str(tuple(a)) : i for i, a in enumerate(s.action_space)},
             )
    return s

def define_experiments(registry):
    s     = defaults(registry.shared)
    sizes = [8, 16, 32]
    return [BaseExperiment(f'env_{size}', s.derive(size=size))
            for size in sizes]
