# Influence-Based-Detection-of-ML-Attacks

This repository contains signals that are formed using Influence Functions to identify data poisons in the training set and adversarial examples in the test set.

## Installation Instructions

Create a virtual environment using conda with Python 3.9.
```
conda create -n ibda python=3.9
conda activate ibda
```
Install [poetry](https://python-poetry.org/docs/) for dependency management and conflict resolution. Then go to the project main directory, i.e., where the *.toml* file is reachable and run
```
poetry install
```
To add a new python package simply run `poetry add <package_name>`

## Commit and Code Instructions 

Make sure that you execute `git pull` before starting to modify any file and to ensure that you have the newest code version.

Before committing, make sure that you run
1. `isort <the modified files or directory>` to ensure that the imports are organized
2. `black <the modified files or directory>` to ensure the same code formatting (e.g. if the command `black .` is executed in the root project directory, each python file will be modified)

Write meaningful and clear commit messages.

For the code, write clean code, i.e., modular code with functions and parameters, clear variable names, comments, and parameter descriptions for the defined functions.
Abstract and factorize your code if necessary, do not have duplicated code fragments or lengthy complicated functions (divide and conquer).

**Careful**: 
1. Do not change the model configurations for the datasets.
2. The influence function configuration can be modified (e.g. reducing the batch size in case of memory issues) but **do not** commit it as each workstation has different resources. 
3. The Makefile can be modified but **do not** commit it because this may create a conflict.

## Data Generation

To download and prepare the data, run `make prepare_data` from the root directory of the project. 
The generated data, e.g. the subset size, can be modified using the command line arguments.
