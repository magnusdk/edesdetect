import os
import pickle
import uuid

import mlflow
from mlflow.exceptions import MlflowException


def log_artifact(
    artifact,
    filename="artifact",
    tmp_dir="/tmp/edesdetect/",
    mlflow_module=mlflow,
):
    """Create a temporary file for the artifact and log it with mlflow.
    Afterwards, the file is deleted."""
    file_dir = f"{tmp_dir}{uuid.uuid4()}/"
    file_path = file_dir + filename
    os.makedirs(file_dir, exist_ok=True)

    with open(file_path, "w+b") as f:
        pickle.dump(artifact, f)
        f.flush()
        mlflow_module.log_artifact(file_path)

    os.remove(file_path)


def best_score(run_id, metric_name):
    client = mlflow.tracking.MlflowClient()
    try:
        # client.get_metric_history(...) raises an exception if the emtric has not been logged before.
        # See issue: https://github.com/mlflow/mlflow/issues/4973
        metric_history = client.get_metric_history(run_id, metric_name)
    except MlflowException:
        metric_history = []

    return (
        max(map(lambda x: x.value, metric_history))
        if metric_history
        # If metric_history is an empty list return a really low number.
        else float("-inf")
    )
