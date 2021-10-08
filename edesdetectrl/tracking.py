import mlflow
from edesdetectrl.config import config

run_id_string = str

class MLflowInitializer:
    def __init__(self, experiment_name, run_name):
        self._run_name = run_name
        self._run_id = None  # If not set, then a random one will be generated and returned after calling start_run().
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(experiment_name)

    def set_run_id(self, run_id):
        self._run_id = run_id

    def start_run(self) -> run_id_string:
        run = mlflow.start_run(
            run_id=self._run_id,
            run_name=self._run_name,
        )
        return run.info.run_id

    def end_run(self):
        pass