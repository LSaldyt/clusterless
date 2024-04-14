import pandas as pd
from pathlib import Path

def run(exp='env_8'):
    parent = Path('data/experiments')
    exp    = parent / exp
    latest = [p for p in exp.iterdir()][-1]
    df = pd.read_csv(latest / 'data/info.csv')
    print(df)
    print(df.describe())
