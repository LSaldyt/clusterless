# Clusterless Multi Agent Rollout 

## Setup

To clone the repository and all submodules:
`git clone --recurse-submodules -j8 https://github.com/LSaldyt/clusterless_multiagent_rollout.git`  

#### Git Large File System

You will need to use [git lfs](https://git-lfs.com/) for large files (e.g. experiment outputs or input data):  
`git lfs install`  
To add an entire directory recursively (this is already done for `data`):  
`git lfs track "data/**"`
To add files with a specific extension (e.g. .csv):  
`git lfs track "*.csv"`  
Individual files or directories can be added with [git lfs migrate](https://github.com/git-lfs/git-lfs/blob/main/docs/man/git-lfs-migrate.adoc). 
Keep  in mind this will require force-pushing git history.

## [Python](https://github.com/python/cpython/tree/main/Include) Installation

To install Python 3.11 from scratch:  
`git clone https://github.com/python/cpython`  
`cd cpython`  
`git checkout 3.11`  
Edit `Include/patchlevel.h` to not have a `+` character which can break things..  
`./configure --enable-optimizations` (optionally add `--prefix ~/mypython`)  
`make -j 8` (or however many cores you have)  
`make altinstall`  (avoid overwriting system install!)  
`python3.11 --version`  
Note you may need to install pip manually, e.g. common next steps are:  
`python3.11 -m ensurepip`  
`python3.11 -m pip --version`  
`python3.11 -m pip install poetry`  

## [Poetry](https://python-poetry.org/) (Python Virtualenv)
Install [poetry](https://python-poetry.org/).  
To install all dependencies, use `poetry install`.  

If you would prefer to not use poetry, try:  
`poetry export -f requirements.txt -o requirements.txt --without-hashes`  
`python -m venv venv`  
`source venv/bin/activate`  
`pip install -r requirements.txt`

## Usage

Programs are either called via scripts or experiments using `./run`. To see all available scripts and experiments, use:  
```
§ ./run list
Scripts:
    test
Experiments:
    test
```
Note that experiment and script names need to be exclusive.  

### Scripts
Scripts are defined as `.py` files in the `scripts/` directory, which have a special `def run(*args)` function.  

### Experiments
Experiments are defined in `experiments/`, such as `experiments/test.py`. 
These are instances of experiment classes defined in `experiments/experiment_classes`, meant to make it easy to modify small settings like prompt format strings. 
  
#### Testing
Once initial setup is complete, try running `./run test` to test basic functionality. 
