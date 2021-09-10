#!/bin/bash

# This script assumes that all python dependencies have been installed.
# If you're unsure, run:
# $ pipenv install --dev

# Load needed modules
CURRENT_DIR=${0%/*}
. "$CURRENT_DIR/load_modules.sh"

# Start Jupyter Lab server in virtual environment
pipenv run python3 -m ipykernel install --user --name=pipenv-venv
pipenv run jupyter-lab --no-browser
