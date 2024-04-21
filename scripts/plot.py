import pandas  as pd
import seaborn as sns

import matplotlib.pyplot as plt

from subprocess import run as srun # type: ignore

from .analyze import get_latest

def run(exp='env_8', **kwargs):
    stamp = get_latest(exp, **kwargs)
    df    = pd.read_csv(stamp / 'data/simulation.csv')
    print(df.describe())
    df = df[df['env'] == 0]

    sns.set_theme(style='whitegrid')
    # scatter = sns.scatterplot(df, x='timestep', y='score', hue='policy')

    fig, ax = plt.subplots(figsize=(7, 6))
    # ax.set_xscale("log")

    sns.boxplot(df,   x='score', y='policy', hue='policy', palette='vlag')
    sns.stripplot(df, x='score', y='policy', size=4, color=".3")

    ax.xaxis.grid(True)
    ax.set(ylabel="")
    sns.despine(trim=True, left=True)
    fig.figure.savefig('plot.pdf', bbox_inches='tight')

    srun('xdg-open plot.pdf 2>&1 > /dev/null', shell=True)

