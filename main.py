import sys, os
from importlib import import_module
from fire      import Fire

from science import Registry, Settings
from base import *

from datetime import datetime

def define_settings(real=False):
    ''' Query the local environment to setup basic settings e.g. paths or GPT version '''
    settings = Settings()

    if os.path.isdir('/scratch/'):
        parent_dir = '/scratch/lsaldyt/experiments/'
    else:
        assert os.path.isdir('data'), 'Please create a top level data directory, or specify another'
        parent_dir = 'data/experiments/'

    settings.update(
        main_settings=Settings(experiment_folder='experiments', script_folder='scripts'),
        seed=datetime.now().year,
        parent_dir=parent_dir,
        ablation=Settings(),
        meta=Settings(
            flush_interval=1,
            log_interval=1,
            log_function=log.info,
            timestamp=True
            )
        )
    return settings

def define_experiments(registry):
    ''' Iterate through all files in experiments and add defined experiments to a registry '''
    main_settings       = registry.shared.main_settings
    experiment_dir_path = Path(main_settings.experiment_folder)
    for exp_path in experiment_dir_path.iterdir():
        if (exp_path.stem in {'__init__', '__pycache__', 'experiment_classes'}
            or not str(exp_path).endswith('.py')):
            continue
        exp_module = import_module(f'{main_settings.experiment_folder}.{exp_path.stem}')
        if not hasattr(exp_module, 'define_experiments'):
            log.info(((f'File {exp_path} does not include define_experiments()' +
                       f'expected to return a list of Experiment-derived objects')))
        for exp in exp_module.define_experiments(registry):
            yield exp

def define_ablations(ablation):
    return dict(
        gpt3=ablation.derive(model='gpt-3.5-turbo', as_json=False),
        gpt4=ablation.derive(model='gpt-4', as_json=True))

def find_and_run_script(script_name, args, kwargs, main_settings):
    script_file = f'{main_settings.script_folder}/{script_name}.py'
    log.info(f'Script file: {script_file}')
    if os.path.isfile(script_file):
        module = import_module(f'{main_settings.script_folder}.{script_name}', package='.')
        code = module.run(*args, **kwargs)
        log.info(f'Finished running script {script_file}!')
        if code is not None and code != 0:
            return code
    else:
        log.warning(f'No script named {script_file}')
        raise FileNotFoundError(f'No script named {script_file}')
    return 0

def list_scripts(main_settings):
    for script in Path(main_settings.script_folder).iterdir():
        if script.suffix == '.py':
            print(f'[blue]    {script.stem}')

''' The Registry object defines a find_and_run_experiments, similar to find_and_run_script'''

def main(name, *args, **kwargs):
    registry = Registry()
    registry.add_shared(define_settings())
    registry.add_ablations(define_ablations(registry.shared.ablation))
    for exp in define_experiments(registry):
        registry.add_experiment(exp)
    registry.finalize()

    if name == 'list':
        print(f'Scripts:')
        list_scripts(registry.shared.main_settings)
        print(f'Experiments:')
        registry.list()
        return

    try:
        log.info(f'Searching for experiment name={name} args={args}')
        registry.find_and_run_experiment(name, args, kwargs, registry.shared.main_settings)
    except KeyError as find_experiment_exception:
        try:
            log.info(f'Searching for script name={name} args={args}')
            find_and_run_script(name, args, kwargs, registry.shared.main_settings)
        except FileNotFoundError as find_script_exception:
            log.error(f'Experiment or Script {name} exception!\n'
                      f'Experiment exception: {find_experiment_exception}\n'
                      f'Script exception:     {find_script_exception}')

if __name__ == '__main__':
    Fire(main)
