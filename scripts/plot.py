import pandas  as pd
import seaborn as sns

import matplotlib.pyplot as plt

from subprocess import run as srun # type: ignore

from .analyze import get_latest

def pareto(dfs):
    fig, ax = plt.subplots(figsize=(7, 6))
    df = dfs['simulation']
    sns.scatterplot(df, x='n_collisions_obstacle', y='score', hue='policy')
    return fig

def line(dfs):
    fig, ax = plt.subplots(figsize=(7, 6))
    df = dfs['simulation']
    print(df['score'])
    print(df['score'].describe())
    print(df['timestep'].describe())
    df = df[df['timestep'] < 64]
    sns.lineplot(df, x='timestep', y='score', hue='policy', style='policy')
    return fig

def barplot(df, x='score'):
    fig, ax = plt.subplots(figsize=(7, 6))
    # ax.set_xscale("log")

    sns.boxplot(df,   x=x, y='policy', hue='policy', palette='vlag')
    sns.stripplot(df, x=x, y='policy', size=4, color=".3")

    ax.xaxis.grid(True)
    ax.set(ylabel="")
    sns.despine(trim=True, left=True)
    return fig
    
def basic(dfs):
    return barplot(dfs['summary'], 'percent')

# def basic(dfs):
#     return barplot(dfs['summary'], 'score')

def collisions(dfs):
    return barplot(dfs['summary'], 'n_collisions_obstacle')

def deaths(dfs):
    return barplot(dfs['summary'], 'n_collisions_agents')

def save(fig, name):
    fig.figure.savefig(f'plots/{name}.pdf', bbox_inches='tight')

def open(name):
    srun(f'xdg-open plots/{name}.pdf 2>&1 > /dev/null', shell=True)

def run(*args, **kwargs):
    selected_env = -1
    pols  = ['random', 'brownian', 'wave', 'multiagent_rollout', 'monte_carlo_rollout', 'belief_monte_carlo_rollout', 'decentralized_multiagent_rollout', 'communication_wave', 'communication_multiagent_rollout']
    # pols  = ['random', 'brownian', 'wave', 'multiagent_rollout', 'belief_monte_carlo_rollout', 'communication_wave', 'communication_multiagent_rollout']
    # pols  = ['wave', 'communication_wave', 'multiagent_rollout', 'communication_multiagent_rollout']
    # pols  = ['belief_monte_carlo_rollout']
    dfs   = dict(simulation=[], summary=[])
    for p in pols:
        for k in dfs:
            try:
                stamp = get_latest('env_8', policy=p, **kwargs)
                df    = pd.read_csv(stamp / f'data/{k}.csv')
                print(p)
                print(df.describe())
                if selected_env != -1:
                    if k == 'summary':
                        df = df[df['environment_index'] == selected_env]
                    else:
                        df = df[df['env'] == selected_env]
                if k == 'simulation':
                    df = df[df['env'] != 1]
                dfs[k].append(df)
            except Exception as e:
                raise
                print(e)
    dfs = {k : pd.concat(v) for k, v in dfs.items()}
    # dfs = {k : v[0] for k, v in dfs.items()}

    sns.set_theme(style='whitegrid')

    plots = dict(deaths=deaths, collisions=collisions, pareto=pareto, basic=basic, line=line)
    # plots = dict(line=line)

    for plot_name, plot_fn in plots.items():
        save(plot_fn(dfs), plot_name)
        open(plot_name)
        plt.clf()
