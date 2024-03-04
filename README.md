# Influence-Based-Detection-of-ML-Attacks

This repository contains signals that are formed using Influence Functions to identify data poisons in the training set and adversarial examples in the test set.

## Installation Instructions

Create a virtual environment using conda with Python 3.9.
```
conda create -n ibda python=3.9
conda activate ibda
```
Install poetry for package management and conflict resolution. Then run
```
poetry install
```
To add a library simply run `poetry add <package_name>`

## Commit and Code Instructions 

Before committing, make sure that you run 
1. `black <the modified files or directory>` to ensure the same code formatting
2. `isort <the modified files or directory>` to ensure that the imports are organized

For the code, write clean code, i.e., modular code with functions and parameters, clear variable names, comments, and parameter descriptions for the defined functions.
Abstract and factorize your code if necessary, do not have duplicated code fragments or lengthy complicated functions (divide and conquer).
