#!bin/bash

pyenv install 3.10
pyenv local 3.10
poetry env use python
poetry install --with dev
poetry run pre-commit install
poetry shell
