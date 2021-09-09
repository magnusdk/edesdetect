#!/bin/bash

# This script assumes that all python dependencies have been installed.
# If you're unsure, run:
# $ pipenv install --dev

# Load needed modules
module load Python/3.8.6-GCCcore-10.2.0
module load JupyterLab/2.2.8-GCCcore-10.2.0
module load CUDAcore/11.3.1

# Start Jupyter Lab server in virtual environment
pipenv run python3 -m ipykernel install --user --name=pipenv-venv
pipenv run jupyter-lab --no-browser
