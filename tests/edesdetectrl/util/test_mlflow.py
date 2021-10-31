import os
import pickle

from edesdetectrl.util.mlflow import log_artifact


def test_log_artifact():
    artifact = {"Hello": "World!"}
    filename = "foobar"

    class mlflowMockModule:
        def log_artifact(self, file_path):
            with open(file_path, "rb") as f:
                read_artifact = pickle.load(f)
                assert read_artifact == artifact

            assert os.path.basename(filename) == filename

    log_artifact(artifact, filename, mlflow_module=mlflowMockModule())
