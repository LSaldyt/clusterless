from science import Experiment, Registry, Settings

class TestExperiment(Experiment):
    def run(self):
        self.ensure()
        data = {'hello' : 'world'}
        self.save_json('result', data)

