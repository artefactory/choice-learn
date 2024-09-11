# Welcome to the Choice-Learn documentation!

<center>
<img src="illustrations/logos/logo_choice_learn.png" width="300">
</center>

<center>

*Large-scale choice modeling through the lens of machine learning*

</center>

<center>

| [<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/GitHub_Invertocat_Logo.svg/langfr-400px-GitHub_Invertocat_Logo.svg.png" alt="drawing" width="30"/>](https://github.com/artefactory/choice-learn) | If you are not coming from GitHub, [check it out](https://github.com/artefactory/choice-learn) |
|-|-|
| [<img src="https://upload.wikimedia.org/wikipedia/commons/8/87/PDF_file_icon.svg" alt="drawing" width="25"/>](https://github.com/artefactory/choice-learn/actions/runs/9642058728/artifacts/1630710678) | You can also read our [academic paper](https://github.com/artefactory/choice-learn/actions/runs/9642058728/artifacts/1630710678) ! |

</center>

<br>

Choice-Learn is a Python package designed to help you estimate discrete choice models and use them (e.g., assortment optimization plug-in).
The package provides ready-to-use datasets and models from the litterature. It also provides a lower level use if you wish to customize any choice model or create your own from scratch. Choice-Learn efficiently handles data with the objective to limit RAM usage. It is made particularly easy to estimate choice models with your own, large datasets.

Choice-Learn uses NumPy and pandas as data backend engines and TensorFlow for models.els.

In this documentation you will find examples to be quickly getting started as well as some more in-depth example.

## What's in there ?

Here is a quick overview of the different functionalities offered by Choice-Learn. Further details are given in the rest of the documentation.

### Data
- [Custom data handling](./references/data/references_choice_dataset.md) for choice datasets with possible memory usage optimizations
- Some Open-Source ready-to use datasets are included within the datasets:
    - [SwissMetro](./references/datasets/references_base.md)
    - [ModeCanada](./references/datasets/references_base.md)
    - The [Train](./references/datasets/references_base.md) dataset
    - The [Heating](./references/datasets/references_base.md) & [Electricity](./references/dataset/references_base.md)datasets from Kenneth Train
    - [Stated car preferences](./references/datasets/references_base.md)
    - The [TaFeng](./references/datasets/references_tafeng.md) dataset from Kaggle
    - The ICDM-2013 [Expedia](./references/datasets/references_expedia.md) dataset from Kaggle

### Models
- [Custom modelling](./notebooks/introduction/4_model_customization.md)
- Ready to be used models:
    - *Linear Models:*
        - [Multinomial Logit](./references/models/references_simple_mnl.md)
        - [Conditional Logit](./references/models/references_clogit.md)
        - [Latent class MNL](./references/models/references_latent_class_mnl.md)
        - [Nested Logit](./references/models/references_nested_logit.md)
    - *Non-Linear Models:*
        - [RUMnet](./references/models/references_rumnet.md)
        - [TasteNet](./references/models/references_tastenet.md)

### Tools
- [Assortment Optimization](./references/toolbox/references_assortment_optimizer.md)
- [Assortment and Pricing](./references/toolbox/references_assortment_optimizer.md)

### Examples

Diverse examples are provided in the How-To section, give it a look !

## Introduction - Discrete Choice Modelling

Discrete choice models aim at explaining or predicting choices over a set of alternatives. Well known use-cases include analyzing people's choice of mean of transport or products purchases in stores.

If you are new to choice modelling, you can check this [resource](https://www.publichealth.columbia.edu/research/population-health-methods/discrete-choice-model-and-analysis). Otherwise, you can also take a look at the [introductive example](notebooks/introduction/1_introductive_example.md).

## Installation

### User installation

To install the required packages in a virtual environment, run the following command:

The easiest is to pip-install the package:b7fc2d1f6e2ff5a5a94a896344592542
```bash
pip install choice-learn
```

Otherwise you can use the git repository to get the latest version:
```bash
git clone git@github.com:artefactory/choice-learn.git
```

### Dependencies
Choice-Learn requires the following:
- Python (>=3.9)
- NumPy (>=1.24)
- pandas (>=1.5)

For modelling you need:
- TensorFlow (>=2.13)

> :warning: **Warning:** If you are a MAC user with a M1 or M2 chip, importing TensorFlow might lead to Python crashing.
> In such case, use anaconda to install TensorFlow with `conda install -c apple tensorflow`.

Finally, an optional requirement used for statsitcal reporting and LBFG-S optimization is:
- TensorFlow Probability (>=0.20.1)

Finally for pricing or assortment optimization, you need either Gurobi or OR-Tools:
- gurobipy (>=11.0.0)
- ortools (>=9.6.2534)


## Contributing
You are welcome to contribute to the project ! You can help in various ways:
- raise issues
- resolve issues already opened
- develop new features
- provide additional examples of use
- fix typos, improve code quality
- develop new tests

We recommend to first open an [issue](https://github.com/artefactory/choice-learn/issues) to discuss your ideas.

## Citation

If you consider this package and any of its feature useful for your research, please cite us.

### License

The use of this software is under the MIT license, with no limitation of usage, including for commercial applications.

### Affiliations

Choice-Learn has been developed through a collaboration between researchers at the Artefact Research Center and the laboratory MICS from CentraleSupélec, Université Paris Saclay.

<center>
[![ARC](https://raw.githubusercontent.com/artefactory/choice-learn/main/docs/illustrations/logos/logo_arc.png){: style="height:60px"}](https://www.artefact.com/data-consulting-transformation/artefact-research-center/)
&nbsp;&nbsp;&nbsp;&nbsp;
[![FNG](https://raw.githubusercontent.com/artefactory/choice-learn/main/docs/illustrations/logos/logo_atf.png){: style="height:65px"}](https://www.artefact.com/)

[![PS](https://raw.githubusercontent.com/artefactory/choice-learn/main/docs/illustrations/logos/logo_paris_saclay.png){: style="height:60px"}](https://www.universite-paris-saclay.fr/)
&nbsp;&nbsp;&nbsp;&nbsp;
[![CS](https://raw.githubusercontent.com/artefactory/choice-learn/main/docs/illustrations/logos/logo_CS.png){: style="height:65px"}](https://mics.centralesupelec.fr/)
&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;
[![LBS](https://raw.githubusercontent.com/artefactory/choice-learn/main/docs/illustrations/logos/logo_lbs.jpeg){: style="height:60px"}](https://www.london.edu/)
&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;
[![INSEAD](https://raw.githubusercontent.com/artefactory/choice-learn/main/docs/illustrations/logos/logo_insead.png){: style="height:65px"}](https://www.insead.edu/)

</center>
