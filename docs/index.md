# Welcome to the choice-learn documentation!

A toolbox for choice-modeling


Choice-Learn is a Python package designed to help you build discrete choice models.
The package provides ready to use datasets and different models from the litterature. It also provides a lower level use if you want to customize any model or create your own from scratch. In particular you will find smart datasets handling to limit RAM usage and different structure commons to any choice model.

Choice-Learn uses NumPy and pandas as data backend engines and TensorFlow for models.

In this documentation you will find examples to be quickly getting started as well as some more in-depth example.

## Introduction - Discrete Choice Modelling

Discrete choice models aim at explaining or predicting choices over a set of alternatives. Well known use-cases include analyzing people's choice of mean of transport or products purchases in stores.

If you are new to choice modelling, you can check this [resource](https://www.publichealth.columbia.edu/research/population-health-methods/discrete-choice-model-and-analysis). Otherwise, you can also take a look at the [introductive example](notebooks/introduction/1_introductive_example.md).

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

Finally, an optional requirement used for statsitcal reporting and LBFG-S optimization is:
- TensorFlow Probability (>=0.20.1)

Once you have created your conda/pip python==3.9 environment, you can install requirements by:
```bash
pip install choice-learn
```

## Citation

If you consider this package and any of its feature useful for your research, please cite our paper:

(WIP - Paper to come)

### License

The use of this software is under the MIT license, with no limitation of usage, including for commercial applications.

### Contributors

### Special Thanks

### Affiliations

This package has been developped within the [Artefact Research Center](https://www.artefact.com/data-consulting-transformation/artefact-research-center/) in collaboration with CentraleSupélec, université Paris-Saclay.
