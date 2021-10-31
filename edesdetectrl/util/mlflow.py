import os
import pickle
import uuid

import mlflow


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
