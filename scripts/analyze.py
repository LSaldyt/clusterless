import pandas as pd
from pathlib import Path
import json
from functools import reduce

def get_latest(exp='env_8', **kwargs):
    parent = Path('data/experiments')
    exp    = parent / exp
    experiments = [p for p in exp.iterdir()]
    latest = sorted(experiments)[::-1]
    stamp = ''
    print(kwargs)
    for x in latest: 
        try:
            with (x/'meta.json').open() as f:
                meta = json.load(f)
            if kwargs: acceptable = reduce(lambda x,k: x and k in meta.keys() and meta[k]==dict(kwargs)[k],dict(kwargs),True)
            else: acceptable=True
        except:
            acceptable = False
        if acceptable:
            stamp = x
            break  
    if not stamp:
        print(f"data for {kwargs} not found")
        exit()
    print(stamp)
    return stamp

def run(exp='env_8', **kwargs):
    stamp = get_latest(exp, **kwargs)
    df    = pd.read_csv(stamp / 'data/summary.csv')
    # print(df)
    print(df.describe())
