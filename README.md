# Warmstart Preconditioning for SG-MCMC

This repository implements experiments on enhancing stochastic gradient Markov Chain Monte Carlo (SG-MCMC) samplers through strategic warmstarts and mass matrix preconditioning.

## Set-up poetry and install dependencies

1.) Clone or fork this repository

2.) Install pipx:
```
pip install --user pipx
```

3.) Install Poetry:
```
pipx install poetry
```

*3a.) Optionally (if you want the env-folder to be created in your project)*:
```
poetry config virtualenvs.in-project true
```

4.) Install this project:
```
poetry install
```

*4a.) If the specified Python version (3.12 for this project) is not found:*

If the required Python version is not installed on your device, install Python from the official [python.org](https://www.python.org/downloads) website.

Then run
```
poetry env use <path-to-your-python-version>
```
and install the project:
```
poetry install
```
