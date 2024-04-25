<div align="center">

# choice-learn-private

[![CI status](https://github.com/artefactory/choice-learn-private/actions/workflows/ci.yaml/badge.svg)](https://github.com/artefactory/choice-learn-private/actions/workflows/ci.yaml?query=branch%3Amain)
[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue.svg)]()

[![Linting , formatting, imports sorting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/choice-learn-private/blob/main/.pre-commit-config.yaml)

</div>

<img src="docs/choice_learn_official_logo.png" width="256">

Choice-Learn is a Python package designed to help you estimate discrete choice models and use them (e.g., assortment optimization plug-in).
The package provides ready-to-use datasets and models from the litterature. It also provides a lower level use if you wish to customize any model or create your own from scratch. In particular you will find efficient data handling to limit RAM usage and structure common to any choice model.

Choice-Learn uses NumPy and pandas as data backend engines and TensorFlow for models.

This repository contains a private version of the package.

## Table of Contents

- [choice-learn-private](#choice-learn-private)
  - [Introduction - Discrete Choice Modelling](#introduction---discrete-choice-modelling)
  - [What's in there ?](#whats-in-there)
  - [Getting Started](#getting-started---fast-track)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Documentation](#documentation)
  - [Citation](#citation)

## Introduction - Discrete Choice Modelling

Discrete choice models aim at explaining or predicting choices over a set of alternatives. Well known use-cases include analyzing people's choice of mean of transport or products purchases in stores.

If you are new to choice modelling, you can check this [resource](https://www.publichealth.columbia.edu/research/population-health-methods/discrete-choice-model-and-analysis). The different notebooks from the [Getting Started](#getting-started---fast-track) section can also help you understand choice modelling and more importantly help you for your usecase.

## What's in there ?

### Data
- Generic dataset handling with the ChoiceDataset class [[Example]](notebooks/introduction/2_data_handling.ipynb)
- Ready-To-Use datasets:
  - [SwissMetro](./choice_learn/datasets/data/swissmetro.csv.gz) [[2]](#citation)
  - [ModeCanada](./choice_learn/datasets/data/ModeCanada.csv.gz) [[3]](#citation)
  - The [Train](./choice_learn/datasets/data/train_data.csv.gz) [[5]](#citation)
  - The [Heating](./choice_learn/datasets/data/heating_data.csv.gz) & [Electricity](./choice_learn/datasets/data/electricity.csv.gz) datasets from Kenneth Train described [here](https://rdrr.io/cran/mlogit/man/Electricity.html) and [here](https://rdrr.io/cran/mlogit/man/Heating.html)
  - The [TaFeng](./choice_learn/datasets/data/ta_feng.csv.zip) dataset from [Kaggle](https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset)
  - The ICDM-2013 [Expedia](./choice_learn/datasets/expedia.py) dataset from [Kaggle](https://www.kaggle.com/c/expedia-personalized-sort) [[6]](#citation)

### Model estimation
- Ready-to-use models:
  - Conditional MultiNomialLogit [[4]](#citation)[[Example]](notebooks/introduction/3_model_clogit.ipynb)
  - Latent Class MultiNomialLogit [[Example]](notebooks/models/latent_class_model.ipynb)
  - RUMnet [[1]](#citation)[[Example]](notebooks/models/rumnet.ipynb)
  - TasteNet [[7]](#citation)[[Example]](notebooks/models/tastenet.ipynb)
- (WIP) - Ready-to-use models to be implemented:
  - Nested Logit
  - [SHOPPER](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-14/issue-1/SHOPPER--A-probabilistic-model-of-consumer-choice-with-substitutes/10.1214/19-AOAS1265.full)
  - Others ...
- Custom modelling is made easy by subclassing the ChoiceModel class [[Example]](notebooks/introduction/4_model_customization.ipynb)

### Auxiliary tools
- Assortment optimization algorithms [[Example]](notebooks/auxiliary_tools/assortment_example.ipynb)
- (WIP) Standardization of evaluation protocols
- (WIP) Interfaces

## Getting Started

You can find the following [notebooks](notebooks/introduction/) to help you getting started with the package:
- [Generic and simple introduction](notebooks/introduction/1_introductive_example.ipynb)
- [Detailed explanations of data handling depending on the data format](notebooks/introduction/2_data_handling.ipynb)
- [A detailed example of conditional logit estimation](notebooks/introduction/3_model_clogit.ipynb)
- [Introduction to custom modelling and more complex parametrization](notebooks/introduction/4_model_customization.ipynb)

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

> :warning: **Warning:** If you are a MAC user with a M1 or M2 chip, importing TensorFlow might lead to Python crashing.
> In such case, use anaconda to install TensorFlow with `conda install -c apple tensorflow`.

Finally, an optional requirement used for report and LBFG-S optimization is:
- TensorFlow Probability (>=0.20.1)

Once you have created your conda/pip python==3.9 environment, you can install requirements by:
```bash
pip install choice-learn
```
## Usage
```python
from choice_learn.data import ChoiceDataset
from choice_learn.models import ConditionalMNL, RUMnet

# Instantiation of a ChoiceDataset from a pandas.DataFrame
# Onl need to specify how the file is encoded:
dataset = ChoiceDataset.from_single_long_df(df=transport_df,
                                            items_id_column="alt",
                                            choices_id_column="case",
                                            choices_column="choice",
                                            shared_features_columns=["income"],
                                            items_features_columns=["cost", "freq", "ovt", "ivt"],
                                            choice_format="item_id")

# Initialization of the model
model = ConditionalMNL()

# Creation of the different weights:

# add_coefficients adds one coefficient for each specified item_index
# intercept, and income are added for each item except the first one that needs to be zeroed
model.add_coefficients(feature_name="intercept",
                       items_indexes=[1, 2, 3])
model.add_coefficients(feature_name="income",
                       items_indexes=[1, 2, 3])
model.add_coefficients(feature_name="ivt",
                       items_indexes=[0, 1, 2, 3])

# shared_coefficient add one coefficient that is used for all items specified in the items_indexes:
# Here, cost, freq and ovt coefficients are shared between all items
model.add_shared_coefficient(feature_name="cost",
                             items_indexes=[0, 1, 2, 3])
model.add_shared_coefficient(feature_name="freq",
                             items_indexes=[0, 1, 2, 3])
model.add_shared_coefficient(feature_name="ovt",
                             items_indexes=[0, 1, 2, 3])

history = model.fit(dataset, get_report=True)
print("The average neg-loglikelihood is:", model.evaluate(dataset).numpy())
print(model.report)
```

## Documentation

A detailed documentation of this project is available [here](https://artefactory.github.io/choice-learn-private/).

## Citation

If you consider this package and any of its feature useful for your research, please cite our paper:

(WIP - Paper to come)

### License

The use of this software is under the MIT license, with no limitation of usage, including for commercial applications.

### Contributors
### Special Thanks

## References

### Papers
[1][Representing Random Utility Choice Models with Neural Networks](https://arxiv.org/abs/2207.12877), Aouad, A.; Désir, A. (2022)\
[2][The Acceptance of Model Innovation: The Case of Swissmetro](https://www.researchgate.net/publication/37456549_The_acceptance_of_modal_innovation_The_case_of_Swissmetro), Bierlaire, M.; Axhausen, K., W.; Abay, G. (2001)\
[3][Applications and Interpretation of Nested Logit Models of Intercity Mode Choice](https://trid.trb.org/view/385097), Forinash, C., V.; Koppelman, F., S. (1993)\
[4][The Demand for Local Telephone Service: A Fully Discrete Model of Residential Calling Patterns and Service Choices](https://www.jstor.org/stable/2555538), Train K., E.; McFadden, D., L.; Moshe, B. (1987)\
[5] [Estimation of Travel Choice Models with Randomly Distributed Values of Time](https://ideas.repec.org/p/fth/lavaen/9303.html), Ben-Akiva, M.; Bolduc, D.; Bradley, M. (1993)\
[6] [Personalize Expedia Hotel Searches - ICDM 2013](https://www.kaggle.com/c/expedia-personalized-sort), Ben Hamner, A.; Friedman, D.; SSA_Expedia. (2013)\
[7] [A Neural-embedded Discrete Choice Model: Learning Taste Representation with Strengthened Interpretability](https://arxiv.org/abs/2002.00922), Han, Y.; Calara Oereuran F.; Ben-Akiva, M.; Zegras, C. (2020)

### Code and Repositories
- [1][RUMnet](https://github.com/antoinedesir/rumnet)
- [PyLogit](https://github.com/timothyb0912/pylogit)
- [Torch Choice](https://gsbdbi.github.io/torch-choice)
- [BioGeme](https://github.com/michelbierlaire/biogeme)
- [mlogit](https://github.com/cran/mlogit)
