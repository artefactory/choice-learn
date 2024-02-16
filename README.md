<div align="center">

# choice-learn-private

[![CI status](https://github.com/artefactory/choice-learn-private/actions/workflows/ci.yaml/badge.svg)](https://github.com/artefactory/choice-learn-private/actions/workflows/ci.yaml?query=branch%3Amain)
[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue.svg)]()

[![Linting , formatting, imports sorting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/choice-learn-private/blob/main/.pre-commit-config.yaml)

</div>

<img src="docs/choice_learn_official_logo.png" width="256">

Choice-Learn is a Python package designed to help you build discrete choice models.
The package provides ready-to-use datasets and different models from the literature. It also provides a lower level use if you want to customize any model or create your own from scratch. In particular, you will find smart datasets handling to limit RAM usage and different structure commons to any choice model.

Choice-Learn uses NumPy and pandas as data backend engines and TensorFlow for models.

This repository contains a private version of the package.

## Table of Contents

- [choice-learn-private](#choice-learn-private)
  - [Introduction - Discrete Choice Modelling](#introduction---discrete-choice-modelling)
  - [Table of Contents](#table-of-contents)
  - [What's in there ?](#whats-in-there)
  - [Getting Started](#getting-started---fast-track)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Documentation](#documentation)
  - [Citation](#citation)

## Introduction - Discrete Choice Modelling

Discrete choice models aim at explaining or predicting a choice from a set of alternatives. Well-known use-cases include analyzing people choice of mean of transport or products purchases in stores.

If you are new to choice modelling, you can check this [resource](https://www.publichealth.columbia.edu/research/population-health-methods/discrete-choice-model-and-analysis). The different notebooks from the [Getting Started](#getting-started---fast-track) section can also help you understand choice modelling and more importantly help you for your usecase.

## What's in there ?

### Data
- Generic dataset handling with the ChoiceDataset class
- Ready-To-Use datasets:
  - SwissMetro from Bierlaire et al. (2001)
  - ModeCanada from Koppelman et al. (1993)

### Models
- Ready to use models:
  - Conditional MultiNomialLogit, Train, K.; McFadden, D.; Ben-Akiva, M. (1987)
  - RUMnet, Aouad A.; Désir A. (2022) [1]
- Ready to use models to be implemented:
  - Nested MultiNomialLogit
  - MultiNomialLogit with latent variables (MixedLogit)
  - TasteNet
  - SHOPPER
  - Others ...
- Custom modelling is made easy by subclassing the ChoiceModel class

### Different tools (to come)
- Standardization of evaluation protocols
- Assortment optimization from model
- Interfaces

## Getting Started - Fast Track

You can find the following notebooks to help you getting started with the package:
- [Introduction to data management](notebooks/choice_learn_introduction_data.ipynb)
- [Introduction to modelling with the conditional logit model on ModeCanada dataset](notebooks/choice_learn_introduction_clogit.ipynb)
- [Introduction to custom modelling with the ModeCanada dataset](notebooks/custom_model.ipynb)

## Installation

### User installation

To install the required packages in a virtual environment, run the following command:

** pip-install not possible yet, to come soon**
```bash
pip install choice-learn
```

In the mean time you can clone the repository:
```bash
git clone git@github.com:artefactory/choice-learn-private.git
```

### Dependencies
Choice-Learn requires the following:
- Python (>=3.8)
- NumPy (>=1.24)
- pandas (>=1.5)
For modelling you need:
- TensorFlow (>=2.13)
Finally, an optional requirement used for report and LBFG-s use is:
- tensorflow_probability (>=0.20.1)

## Usage
```python
from choice_learn.data import ChoiceDataset
from choice_learn.models import ConditionalMNL, RUMnet
```

## Documentation

A detailed documentation of this project is available [here](https://artefactory.github.io/choice-learn-private/)

## Citation

### Contributors

## References

### Papers
[1][Representing Random Utility Choice Models with Neural Networks](https://arxiv.org/abs/2207.12877), Aouad A.; Désir A. (2022)

### Code and Repositories
- [PyLogit](https://github.com/timothyb0912/pylogit)
- [Torch Choice](https://gsbdbi.github.io/torch-choice/)
- [1][RUMnet](https://github.com/antoinedesir/rumnet)
