
def _make_available_policies(excluded={'utils', '__init__'}):
    import importlib
    from pathlib import Path

    available_policies = dict()
    policy_dir = Path(__file__).parent

    for file in policy_dir.iterdir():
        key = file.stem
        if file.suffix == '.py' and key not in excluded:
            mod = importlib.import_module(f'.{key}', package='clusterless.policies')
            assert callable(getattr(mod, key)), \
                f'The python module {file} must define a callable function {key}()'
            available_policies[key] = getattr(mod, key)
    return available_policies

available_policies = _make_available_policies()
