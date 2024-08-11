#!bin/bash

pyenv install 3.10
pyenv local 3.10
poetry env use python
poetry install --with dev
poetry run pre-commit install
poetry shell

if [ -z "$MLIP_LIB" ]; then
  _MLIP_LIB=$(cat libsetter.json | jq -r .path_to_mlip2.MLIP_LIB)
  source export MLIP_LIB=$_MLIP_DIR
fi

if [ -z "$MLIP_DIR" ]; then
  _MLIP_DIR=$(cat libsetter.json | jq -r .path_to_mlip2.MLIP_DIR)
  source export MLIP_DIR=$_MLIP_DIR
fi

echo if no concern adding $MLIP_DIR, and $MLPI_LIB to bashrc is recommended
