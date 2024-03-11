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
The package provides ready to use datasets and different models from the litterature. It also provides a lower level use if you want to customize any model or create your own from scratch. In particular you will find smart datasets handling to limit RAM usage and different structure commons to any choice model.

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

Discrete choice models aim at explaining or predicting a choice from a set of alternatives. Well known use-cases include analyzing people choice of mean of transport or products purchases in stores.

If you are new to choice modelling, you can check this [resource](https://www.publichealth.columbia.edu/research/population-health-methods/discrete-choice-model-and-analysis). The different notebooks from the [Getting Started](#getting-started---fast-track) section can also help you understand choice modelling and more importantly help you for your usecase.

## What's in there ?

### Data
- Generic dataset handling with the ChoiceDataset class [[Example]](https://github.com/artefactory/choice-learn-private/blob/main/notebooks/choice_learn_introduction_data.ipynb)
- Ready-To-Use datasets:
  - [SwissMetro](./choice_learn/datasets/data/swissmetro.csv.gz) [[2]](#citation)
  - [ModeCanada](./choice_learn/datasets/data/ModeCanada.csv.gz) [[3]](#citation)
  - The [Train](./choice_learn/datasets/data/train_data.csv.gz) [[5]](#citation)
  - The [Heating](./choice_learn/datasets/data/heating_data.csv.gz) & [Electricity](./choice_learn/datasets/data/electricity.csv.gz) datasets from Kenneth Train described [here](https://rdrr.io/cran/mlogit/man/Electricity.html) and [here](https://rdrr.io/cran/mlogit/man/Heating.html)
  - The [TaFeng](./choice_learn/datasets/data/ta_feng.csv.zip) dataset from [Kaggle](https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset)

### Models
- Ready-to-use models:
  - Conditional MultiNomialLogit [[4]](#citation)[[Example]](https://github.com/artefactory/choice-learn-private/blob/main/notebooks/choice_learn_introduction_clogit.ipynb)
  - Latent Class MultiNomialLogit [[Example]](https://github.com/artefactory/choice-learn-private/blob/main/notebooks/latent_class_model.ipynb)
  - RUMnet [[1]](#citation)[[Example]](https://github.com/artefactory/choice-learn-private/blob/main/notebooks/rumnet_example.ipynb)
- Ready-to-use models to be implemented:
  - Nested MultiNomialLogit
  - [TasteNet](https://arxiv.org/abs/2002.00922)
  - [SHOPPER](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-14/issue-1/SHOPPER--A-probabilistic-model-of-consumer-choice-with-substitutes/10.1214/19-AOAS1265.full)
  - Others ...
- Custom modelling is made easy by subclassing the ChoiceModel class [[Example]](https://github.com/artefactory/choice-learn-private/blob/main/notebooks/custom_model.ipynb)

### Different tools
- Assortment optimization from model [[Example]](https://github.com/artefactory/choice-learn-private/blob/main/notebooks/assortment_example.ipynb)
- (WIP) Standardization of evaluation protocols
- (WIP) Interfaces

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

Finally, an optional requirement used for report and LBFG-S optimization is:
- TensorFlow Probability (>=0.20.1)

## Usage
```python
from choice_learn.data import ChoiceDataset
from choice_learn.models import ConditionalMNL, RUMnet

# Instantiation of a ChoiceDataset from a pandas.DataFrame
# Onl need to specify how the file is encoded:
dataset = ChoiceDataset.from_single_long_df(df=transport_df,
                                            items_id_column="alt",
                                            contexts_id_column="case",
                                            choices_column="choice",
                                            contexts_features_columns=["income"],
                                            contexts_items_features_columns=["cost", "freq", "ovt", "ivt"],
                                            choice_format="item_id")

# Initialization of the model
model = ConditionalMNL(optimizer="lbfgs")

# Creation of the different weights:


# add_coefficients adds one coefficient for each specified item_index
# intercept, and income are added for each item except the first one that needs to be zeroed
model.add_coefficients(coefficient_name="beta_inter", feature_name="intercept", items_indexes=[1, 2, 3])
model.add_coefficients(coefficient_name="beta_income", feature_name="income", items_indexes=[1, 2, 3])

# ivt is added for each item:
model.add_coefficients(coefficient_name="beta_ivt", feature_name="ivt", items_indexes=[0, 1, 2, 3])

# shared_coefficient add one coefficient that is used for all items specified in the items_indexes:
# Here, cost, freq and ovt coefficients are shared between all items
model.add_shared_coefficient(coefficient_name="beta_cost", feature_name="cost", items_indexes=[0, 1, 2, 3])
model.add_shared_coefficient(coefficient_name="beta_freq", feature_name="freq", items_indexes=[0, 1, 2, 3])
model.add_shared_coefficient(coefficient_name="beta_ovt", feature_name="ovt", items_indexes=[0, 1, 2, 3])

history = model.fit(dataset, epochs=1000, get_report=True)
print("The average neg-loglikelihood is:", model.evaluate(dataset).numpy())
print(model.report)
```

## Documentation

A detailed documentation of this project is available [here](https://artefactory.github.io/choice-learn-private/)

## Citation

### Contributors
### Special Thanks

## References

### Papers
[1][Representing Random Utility Choice Models with Neural Networks](https://arxiv.org/abs/2207.12877), Aouad, A.; Désir, A. (2022)\
[2][The Acceptance of Model Innovation: The Case of Swissmetro](https://www.researchgate.net/publication/37456549_The_acceptance_of_modal_innovation_The_case_of_Swissmetro), Bierlaire, M.; Axhausen, K., W.; Abay, G. (2001)\
[3][Applications and Interpretation of Nested Logit Models of Intercity Mode Choice](https://trid.trb.org/view/385097), Forinash, C., V.; Koppelman, F., S. (1993)\
[4][The Demand for Local Telephone Service: A Fully Discrete Model of Residential Calling Patterns and Service Choices](https://www.jstor.org/stable/2555538), Train K., E.; McFadden, D., L.; Moshe, B. (1987)\
[5] [Estimation of Travel Choice Models with Randomly Distributed Values of Time](https://ideas.repec.org/p/fth/lavaen/9303.html), Ben-Akiva M; Bolduc D; Bradley M(1993)

### Code and Repositories
- [1][RUMnet](https://github.com/antoinedesir/rumnet)
- [PyLogit](https://github.com/timothyb0912/pylogit)
- [Torch Choice](https://gsbdbi.github.io/torch-choice)
- [BioGeme](https://github.com/michelbierlaire/biogeme)
- [mlogit](https://github.com/cran/mlogit)
