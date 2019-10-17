#!/bin/bash
pipenv run pip list && PYTHONHASHSEED=44 GPU_ON=1 pipenv run python preprocess.py && PYTHONHASHSEED=44 GPU_ON=1 pipenv run python generators.py