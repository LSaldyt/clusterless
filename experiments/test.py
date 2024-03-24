from science import Settings, Experiment
from .experiment_classes import TestExperiment

def define_experiments(registry):
    s = registry.shared.derive()
    return [TestExperiment('test_experiment', s)]
