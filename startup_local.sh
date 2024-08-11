#!bin/bash

pyenv install 3.10
pyenv local 3.10
poetry env use python
poetry install --with dev
poetry run pre-commit install
poetry shell

# Extract paths from JSON and export them
mlip_lib_temp=$(cat libsetter.json | jq -r .path_to_mlip2.MLIP_LIB)
mlip_dir_temp=$(cat libsetter.json | jq -r .path_to_mlip2.MLIP_DIR)

export MLIP_LIB=$mlip_lib_temp
export MLIP_DIR=$mlip_dir_temp

# Suggest adding the exports to bashrc
echo "If no concern, adding MLIP_DIR and MLIP_LIB to bashrc is recommended."
echo "You can do this by adding the following lines to your ~/.bashrc file:"
echo "export MLIP_LIB=$mlip_lib_temp"
echo "export MLIP_DIR=$mlip_dir_temp"
