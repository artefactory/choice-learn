<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/artefactory/choice-learn/main/docs/illustrations/logos/logo_choice_learn.png">
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/artefactory/choice-learn/main/docs/white_logo_choice_learn.png">
  <img src="https://raw.githubusercontent.com/artefactory/choice-learn/refs/heads/white_logo/docs/white_logo_choice_learn.png" width="300">
</picture>

*Large-scale choice modeling through the lens of machine learning*

[![CI status](https://github.com/artefactory/choice-learn/actions/workflows/ci.yaml/badge.svg)](https://github.com/artefactory/choice-learn/actions/workflows/ci.yaml?query=branch%3Amain)
[![Linting , formatting, imports sorting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/choice-learn/blob/main/.pre-commit-config.yaml)


![PyPI - Python Version](https://img.shields.io/pypi/pyversions/choice-learn?logo=python)
![PyPI - Version](https://img.shields.io/pypi/v/choice-learn)
![PyPI - License](https://img.shields.io/pypi/l/choice-learn?color=purple)

[![DOI](https://joss.theoj.org/papers/10.21105/joss.06899/status.svg)](https://doi.org/10.21105/joss.06899)
[![cite](https://img.shields.io/badge/Citation-BibTeX-cyan)](./CITATION.bib)
</div>


Choice-Learn is a Python package designed to help you formulate, estimate, and deploy discrete choice models, e.g., for assortment planning.
The package provides ready-to-use datasets and models studied in the academic literature. It also provides a lower level use if you wish to customize the specification of the choice model or formulate your own model from scratch. Choice-Learn efficiently handles large-scale choice data by limiting RAM usage.

Choice-Learn uses NumPy and pandas as data backend engines and TensorFlow for models.

## :trident: Table of Contents
  - [Introduction - Discrete Choice modeling](#trident-introduction---discrete-choice-modeling)
  - [What's in there ?](#trident-whats-in-there-)
  - [Getting Started](#trident-getting-started)
  - [Installation](#trident-installation)
  - [Usage](#trident-usage)
  - [Documentation](#trident-documentation)
  - [Contributing](#trident-contributing)
  - [Citation](#trident-citation)
  - [References](#trident-references)

## :trident: Introduction - Discrete Choice modeling

Discrete choice models aim at explaining or predicting choices over a set of alternatives. Well known use-cases include analyzing people's choice of mean of transport or products purchases in stores.

If you are new to choice modeling, you can check this [resource](https://www.publichealth.columbia.edu/research/population-health-methods/discrete-choice-model-and-analysis). The different notebooks from the [Getting Started](#trident-getting-started) section can also help you understand choice modeling and more importantly help you for your usecase.

## :trident: What's in there ?

### Data
- The **ChoiceDataset** class can handle choice datasets with efficient memory management. It can be used on your own dataset. [[Example]](notebooks/introduction/2_data_handling.ipynb)
- Many academic datasets are integrated in the library and ready to be used:

| Dataset    | Raw Data | Origin | *from choice_learn.datasets import* | Doc |
| ---------- | :----: | ------ | ------ | :---: |
| SwissMetro  | [csv](./choice_learn/datasets/data/swissmetro.csv.gz) | Bierlaire et al. (2001) [[2]](#trident-references) | *load_swissmetro* | [#](https://artefactory.github.io/choice-learn/references/datasets/references_base/#choice_learn.datasets.base.load_swissmetro) |
| ModeCanada | [csv](./choice_learn/datasets/data/ModeCanada.csv.gz) | Forinash and Koppelman (1993) [[3]](#trident-references) | *load_modecanada* | [#](https://artefactory.github.io/choice-learn/references/datasets/references_base/#choice_learn.datasets.base.load_modecanada) |
| Train    | [csv](./choice_learn/datasets/data/train_data.csv.gz) | Ben-Akiva et al. (1993) [[5]](#trident-references) |*load_train* | [#](https://artefactory.github.io/choice-learn/references/datasets/references_base/#choice_learn.datasets.base.load_train) |
| Heating | [csv](./choice_learn/datasets/data/heating_data.csv.gz) | Kenneth Train's [website](https://eml.berkeley.edu/~train/) | *load_heating* | [#](https://artefactory.github.io/choice-learn/references/datasets/references_base/#choice_learn.datasets.base.load_heating) |
| HC | [csv](./choice_learn/datasets/data/HC.csv.gz) | Kenneth Train's [website](https://eml.berkeley.edu/~train/) | *load_hc* | [#](https://artefactory.github.io/choice-learn/references/datasets/references_base/#choice_learn.datasets.base.load_hc) |
| Electricity | [csv](./choice_learn/datasets/data/electricity.csv.gz) | Kenneth Train's [website](https://eml.berkeley.edu/~train/) | *load_electricity* | [#](https://artefactory.github.io/choice-learn/references/datasets/references_base/#choice_learn.datasets.base.load_electricity) |
| Stated Car Preferences | [csv](./choice_learn/datasets/data/car.csv.gz) | McFadden and Train (2000) [[9]](#trident-references) | *load_car_preferences* | [#](https://artefactory.github.io/choice-learn/references/datasets/references_base/#choice_learn.datasets.base.load_car_preferences) |
| TaFeng Grocery Dataset | [csv](./choice_learn/datasets/data/ta_feng.csv.zip) | [Kaggle](https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset) | *load_tafeng* | [#](https://artefactory.github.io/choice-learn/references/datasets/references_tafeng/) |
| ICDM-2013 Expedia | [url](https://www.kaggle.com/c/expedia-personalized-sort) |  Ben Hamner and Friedman (2013) [[6]](#trident-references) | *load_expedia* | [#](https://artefactory.github.io/choice-learn/references/datasets/references_expedia/) |
| London Passenger Mode Choice | [url]() | Hillel et al. (2018) [[11]](#trident-references) | *load_londonpassenger* | [#](https://artefactory.github.io/choice-learn/references/datasets/references_base/#choice_learn.datasets.base.load_londonpassenger) |


### Model estimation
- Different models are already implemented. You can import and parametrize the models for your own usage.
- Otherwise, **custom modeling** is made easy by subclassing the ChoiceModel class and specifying your own utility function. [[Example]](notebooks/introduction/4_model_customization.ipynb)

*List of implemented & ready-to-use models:*
| Model | Example | Colab | Related Paper | *from choice_learn.models import* | Doc |
| ---------- | -------- | -------- | ------ | ------ | :---: |
| MNL | [notebook](notebooks/models/simple_mnl.ipynb) | [![Open In Colab](https://img.shields.io/badge/-grey?logo=googlecolab)](https://colab.research.google.com/github/artefactory/choice-learn/blob/main/notebooks/models/simple_mnl.ipynb) | | *SimpleMNL* | [#](https://artefactory.github.io/choice-learn/references/models/references_simple_mnl/) |
| Conditional Logit | [notebook](notebooks/introduction/3_model_clogit.ipynb) | [![Open In Colab](https://img.shields.io/badge/-grey?logo=googlecolab)](https://colab.research.google.com/github/artefactory/choice-learn/blob/main/notebooks/introduction/3_model_clogit.ipynb) | Train et al. [[4]](#trident-references) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| *ConditionalLogit* | [#](https://artefactory.github.io/choice-learn/references/models/references_clogit/) |
| Nested Logit | [notebook](notebooks/models/nested_logit.ipynb) | [![Open In Colab](https://img.shields.io/badge/-grey?logo=googlecolab)](https://colab.research.google.com/github/artefactory/choice-learn/blob/main/notebooks/models/nested_logit.ipynb) | McFadden [[10]](#trident-references)  | *NestedLogit* | [#](https://artefactory.github.io/choice-learn/references/models/references_nested_logit/) |
| Latent Class MNL | [notebook](notebooks/models/latent_class_model.ipynb) | [![Open In Colab](https://img.shields.io/badge/-grey?logo=googlecolab)](https://colab.research.google.com/github/artefactory/choice-learn/blob/main/notebooks/models/latent_class_model.ipynb) | | *LatentClassConditionalLogit* | [#](https://artefactory.github.io/choice-learn/references/models/references_latent_class_mnl/) |
| Halo MNL | [notebook](notebooks/models/halo_mnl.ipynb) | [![Open In Colab](https://img.shields.io/badge/-grey?logo=googlecolab)](https://colab.research.google.com/github/artefactory/choice-learn/blob/main/notebooks/models/halo_mnl.ipynb) | Maragheh et al. [[14]](#trident-references) | *HaloMNL* | [#](https://artefactory.github.io/choice-learn/references/models/halo_mnl/) |
| Low-Rank Halo MNL | [notebook](notebooks/models/halo_mnl.ipynb) | [![Open In Colab](https://img.shields.io/badge/-grey?logo=googlecolab)](https://colab.research.google.com/github/artefactory/choice-learn/blob/main/notebooks/models/halo_mnl.ipynb) | Ko and Li [[15]](#trident-references) | *LowRankHaloMNL* | [#](https://artefactory.github.io/choice-learn/references/models/halo_mnl/) |

| NN-based Model | Example | Colab | Related Paper | *from choice_learn.models import* | Doc |
| ---------- | -------- | ------ | ---- | ------ | :---: |
| RUMnet| [notebook](notebooks/models/rumnet.ipynb) | [![Open In Colab](https://img.shields.io/badge/-grey?logo=googlecolab)](https://colab.research.google.com/github/artefactory/choice-learn/blob/main/notebooks/models/rumnet.ipynb) | Aouad and Désir [[1]](#trident-references) | *RUMnet* | [#](https://artefactory.github.io/choice-learn/references/models/references_rumnet/#choice_learn.models.rumnet.PaperRUMnet) |
| TasteNet | [notebook](notebooks/models/tastenet.ipynb) | [![Open In Colab](https://img.shields.io/badge/-grey?logo=googlecolab)](https://colab.research.google.com/github/artefactory/choice-learn/blob/main/notebooks/models/tastenet.ipynb) | Han et al. [[7]](#trident-references) | *TasteNet* | [#](https://artefactory.github.io/choice-learn/references/models/references_tastenet/) |
| Learning-MNL | [notebook](notebooks/models/learning_mnl.ipynb) | [![Open In Colab](https://img.shields.io/badge/-grey?logo=googlecolab)](https://colab.research.google.com/github/artefactory/choice-learn/blob/main/notebooks/models/learning_mnl.ipynb) | Sifringer et al. [[13]](#trident-references) | *LearningMNL* | [#](https://artefactory.github.io/choice-learn/references/models/references_learning_mnl/) |
| ResLogit | [notebook](notebooks/models/reslogit.ipynb) | [![Open In Colab](https://img.shields.io/badge/-grey?logo=googlecolab)](https://colab.research.google.com/github/artefactory/choice-learn/blob/main/notebooks/models/reslogit.ipynb) | Wong and Farooq [[12]](#trident-references) | *ResLogit* | [#](https://artefactory.github.io/choice-learn/references/models/references_reslogit/) |

| Basket Model | Example | Colab | Related Paper | *from choice_learn.basket_models import* | Doc |
| ---------- | -------- | ------ | ---- | ------ | :---: |
| Shopper | [notebook](notebooks/basket_models/shopper.ipynb) | [![Open In Colab](https://img.shields.io/badge/-grey?logo=googlecolab)](https://colab.research.google.com/github/artefactory/choice-learn/blob/main/notebooks/basket_models/shopper.ipynb) | Ruiz et al. [[16]](#trident-references) | *Shopper* | [#](https://artefactory.github.io/choice-learn/references/basket_models/references_shopper/) |
| Alea Carta | [notebook](notebooks/basket_models/alea_carta.ipynb) | [![Open In Colab](https://img.shields.io/badge/-grey?logo=googlecolab)](https://colab.research.google.com/github/artefactory/choice-learn/blob/main/notebooks/basket_models/alea_carta.ipynb) | Désir et al. [[17]](#trident-references) | *AleaCarta* | [#]() |


### Auxiliary tools
Algorithms leveraging choice models are integrated within the library:
- Assortment & Pricing optimization algorithms [[Example]](notebooks/auxiliary_tools/assortment_example.ipynb) [[8]](#trident-references) [![Open In Colab](https://img.shields.io/badge/-grey?logo=googlecolab)](https://colab.research.google.com/github/artefactory/choice-learn/blob/main/notebooks/auxiliary_tools/assortment_example.ipynb)

## :trident: Getting Started

You can find the following tutorials to help you getting started with the package:
- Generic and simple introduction [[notebook]](notebooks/introduction/1_introductive_example.ipynb)[[doc]](https://expert-dollop-1wemk8l.pages.github.io/notebooks/introduction/1_introductive_example/) [![Open In Colab](https://img.shields.io/badge/-grey?logo=googlecolab
)](https://colab.research.google.com/github/artefactory/choice-learn/blob/main/notebooks/introduction/1_introductive_example.ipynb)
- Detailed explanations of data handling depending on the data format [[noteboook]](notebooks/introduction/2_data_handling.ipynb)[[doc]](https://expert-dollop-1wemk8l.pages.github.io/notebooks/introduction/2_data_handling/)  [![Open In Colab](https://img.shields.io/badge/-grey?logo=googlecolab)](https://colab.research.google.com/github/artefactory/choice-learn/blob/main/notebooks/introduction/2_data_handling.ipynb)
- A detailed example of conditional logit estimation [[notebook]](notebooks/introduction/3_model_clogit.ipynb)[[doc]](https://expert-dollop-1wemk8l.pages.github.io/notebooks/introduction/3_model_clogit/)  [![Open In Colab](https://img.shields.io/badge/-grey?logo=googlecolab)](https://colab.research.google.com/github/artefactory/choice-learn/blob/main/notebooks/introduction/3_model_clogit.ipynb)
- Introduction to custom modeling and more complex parametrization [[notebook]](notebooks/introduction/4_model_customization.ipynb)[[doc]](https://expert-dollop-1wemk8l.pages.github.io/notebooks/introduction/4_model_customization/) [![Open In Colab](https://img.shields.io/badge/-grey?logo=googlecolab)](https://colab.research.google.com/github/artefactory/choice-learn/blob/main/notebooks/introduction/4_model_customization.ipynb)
- All models and algorithms have a companion example in the notebook [directory](./notebooks/)

## :trident: Installation

### User installation

To install the required packages in a virtual environment, run the following command:
```bash
make install
```

The easiest is to pip-install the package:
```bash
pip install choice-learn
```

Otherwise you can use the git repository to get the latest version:
```bash
git clone git@github.com:artefactory/choice-learn.git
```

### Dependencies
For manual installation, Choice-Learn requires the following:
- Python (>=3.9, <3.13)
- NumPy (>=1.24)
- pandas (>=1.5)

For modeling you need:
- TensorFlow (>=2.14, <2.17)

> :warning: **Warning:** If you are a MAC user with a M1 or M2 chip, importing TensorFlow might lead to Python crashing.
> In such case, use anaconda to install TensorFlow with `conda install -c apple tensorflow`.

An optional requirement used for coefficients analysis and L-BFGS optimization is:
- TensorFlow Probability (>=0.22)

Finally for pricing or assortment optimization, you need either Gurobi or OR-Tools:
- gurobipy (>=11.0)
- ortools (>=9.6)

<p align="center">
  <a href="https://numpy.org/">
    <img src="https://raw.githubusercontent.com/artefactory/choice-learn/main/docs/illustrations/logos/numpy_logo.png" width="60" />
  </a>
  &nbsp;
  &nbsp;
  <a href="https://pandas.pydata.org/">
    <img src="https://raw.githubusercontent.com/artefactory/choice-learn/main/docs/illustrations/logos/pandas_logo.png" width="60" />
  </a>
  &nbsp;
  &nbsp;
  <a href="https://www.tensorflow.org">
    <img src="https://raw.githubusercontent.com/artefactory/choice-learn/main/docs/illustrations/logos/tf_logo.png" width="60" />
  </a>
  &nbsp;
  &nbsp;
  <a href="https://www.gurobi.com/">
    <img src="https://raw.githubusercontent.com/artefactory/choice-learn/main/docs/illustrations/logos/gurobi_logo.png" width="60" />
  </a>
  &nbsp;
  &nbsp;
  <a href="https://developers.google.com/optimization?hl=fr">
    <img src="https://raw.githubusercontent.com/artefactory/choice-learn/main/docs/illustrations/logos/or_tools_logo.png" width="70" />
  </a>
</p>

> :bulb: **Tip:** You can use the poetry.lock or requirements-complete.txt files with poetry or pip to install a fully predetermined and working environment.

## :trident: Usage
Here is a short example of model parametrization to estimate a Conditional Logit on the ModeCanada dataset.

```python
from choice_learn.data import ChoiceDataset
from choice_learn.models import ConditionalLogit, RUMnet
from choice_learn.datasets import load_modecanada

transport_df = load_modecanada(as_frame=True)
# Instantiation of a ChoiceDataset from a pandas.DataFrame
dataset = ChoiceDataset.from_single_long_df(df=transport_df,
                                            items_id_column="alt",
                                            choices_id_column="case",
                                            choices_column="choice",
                                            shared_features_columns=["income"],
                                            items_features_columns=["cost", "freq", "ovt", "ivt"],
                                            choice_format="one_zero")

# Initialization of the model
model = ConditionalLogit()

# Creation of the different weights:

# add_coefficients adds one coefficient for each specified item_index
# intercept, and income are added for each item except the first one that needs to be zeroed
model.add_coefficients(feature_name="intercept",
                       items_indexes=[1, 2, 3])
model.add_coefficients(feature_name="income",
                       items_indexes=[1, 2, 3])
model.add_coefficients(feature_name="ivt",
                       items_indexes=[0, 1, 2, 3])

# add_shared_coefficient add one coefficient that is used for all items specified in the items_indexes:
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

## :trident: Documentation

A detailed documentation of this project is available [here](https://artefactory.github.io/choice-learn/).\
TensorFlow also has extensive [documentation](https://www.tensorflow.org/) that can help you.\
An academic paper has been published in the Journal of Open-Source Software, [here](https://joss.theoj.org/papers/10.21105/joss.06899).

## :trident: Contributing
You are welcome to contribute to the project ! You can help in various ways:
- raise issues
- resolve issues already opened
- develop new features
- provide additional examples of use
- fix typos, improve code quality
- develop new tests

We recommend to first open an [issue](https://github.com/artefactory/choice-learn/issues) to discuss your ideas. More details are given [here](./CONTRIBUTING.md).

## :trident: Citation

If you consider this package or any of its feature useful for your research, consider citing our [paper](https://joss.theoj.org/papers/10.21105/joss.06899):

<a href="https://joss.theoj.org/papers/10.21105/joss.06899">
<img align="left" width="100"src="https://github.com/openjournals/joss/blob/main/docs/logos/joss-logo.png?raw=true" />
</a>

```bash
@article{Auriau2024,
  doi = {10.21105/joss.06899},
  url = {https://doi.org/10.21105/joss.06899},
  year = {2024},
  publisher = {The Open Journal},
  volume = {9},
  number = {101},
  pages = {6899},
  author = {Vincent Auriau and Ali Aouad and Antoine Désir and Emmanuel Malherbe},
  title = {Choice-Learn: Large-scale choice modeling for operational contexts through the lens of machine learning},
  journal = {Journal of Open Source Software} }
```

If you make use of the *AleaCarta* model [[17]](#trident-references), consider citing the corresponding [paper]():

<a href="https://joss.theoj.org/papers/10.21105/joss.06899">
<img align="left" width="100"src="https://ecmlpkdd-storage.s3.eu-central-1.amazonaws.com/ECML_1_e012008d41.png" />
</a>

```bash
@article{
  doi = {},
  url = {},
  year = {},
  publisher = {},
  volume = {},
  number = {},
  pages = {},
  author = {},
  title = {},
  journal = {} }
```
### License

The use of this software is under the MIT license, with no limitation of usage, including for commercial applications.

### Affiliations

Choice-Learn has been developed through a collaboration between researchers at the Artefact Research Center and the laboratory MICS from CentraleSupélec, Université Paris Saclay.

<p align="center">
  <a href="https://www.artefact.com/data-consulting-transformation/artefact-research-center/">
    <img src="https://raw.githubusercontent.com/artefactory/choice-learn/main/docs/illustrations/logos/logo_arc.png" height="60" />
  </a>
  &emsp;
  &emsp;
  <a href="https://www.artefact.com/">
    <img src="https://raw.githubusercontent.com/artefactory/choice-learn/main/docs/illustrations/logos/logo_atf.png" height="65" />
  </a>
</p>

<p align="center">
  <a href="https://www.universite-paris-saclay.fr/">
    <img src="https://raw.githubusercontent.com/artefactory/choice-learn/main/docs/illustrations/logos/logo_paris_saclay.png" height="60" />
  </a>
  &emsp;
  &emsp;
  <a href="https://mics.centralesupelec.fr/">
    <img src="https://raw.githubusercontent.com/artefactory/choice-learn/main/docs/illustrations/logos/logo_CS.png" height="60" />
  </a>
  &emsp;
  &emsp;
  <a href="https://www.london.edu/">
    <img src="https://raw.githubusercontent.com/artefactory/choice-learn/main/docs/illustrations/logos/logo_lbs.jpeg" height="60" />
  </a>
  &emsp;
  &emsp;
  <a href="https://www.insead.edu/">
    <img src="https://raw.githubusercontent.com/artefactory/choice-learn/main/docs/illustrations/logos/logo_insead.png" height="60" />
  </a>
</p>

## :trident: References

### Papers
[1][Representing Random Utility Choice Models with Neural Networks](https://arxiv.org/abs/2207.12877), Aouad, A.; Désir, A. (2022)\
[2][The Acceptance of Model Innovation: The Case of Swissmetro](https://www.researchgate.net/publication/37456549_The_acceptance_of_modal_innovation_The_case_of_Swissmetro), Bierlaire, M.; Axhausen, K., W.; Abay, G. (2001)\
[3][Applications and Interpretation of Nested Logit Models of Intercity Mode Choice](https://trid.trb.org/view/385097), Forinash, C., V.; Koppelman, F., S. (1993)\
[4][The Demand for Local Telephone Service: A Fully Discrete Model of Residential Calling Patterns and Service Choices](https://www.jstor.org/stable/2555538), Train K., E.; McFadden, D., L.; Moshe, B. (1987)\
[5] [Estimation of Travel Choice Models with Randomly Distributed Values of Time](https://ideas.repec.org/p/fth/lavaen/9303.html), Ben-Akiva, M.; Bolduc, D.; Bradley, M. (1993)\
[6] [Personalize Expedia Hotel Searches - ICDM 2013](https://www.kaggle.com/c/expedia-personalized-sort), Ben Hamner, A.; Friedman, D.; SSA_Expedia. (2013)\
[7] [A Neural-embedded Discrete Choice Model: Learning Taste Representation with Strengthened Interpretability](https://arxiv.org/abs/2002.00922), Han, Y.; Calara Oereuran F.; Ben-Akiva, M.; Zegras, C. (2020)\
[8] [A branch-and-cut algorithm for the latent-class logit assortment problem](https://www.sciencedirect.com/science/article/pii/S0166218X12001072), Méndez-Díaz, I.; Miranda-Bront, J. J.; Vulcano, G.; Zabala, P. (2014)\
[9] [Stated Preferences for Car Choice in Mixed MNL models for discrete response.](https://www.jstor.org/stable/2678603), McFadden, D. and Kenneth Train (2000)\
[10] [Modeling the Choice of Residential Location](https://onlinepubs.trb.org/Onlinepubs/trr/1978/673/673-012.pdf), McFadden, D. (1978)\
[11] [Recreating passenger mode choice-sets for transport simulation: A case study of London, UK](https://www.icevirtuallibrary.com/doi/10.1680/jsmic.17.00018), Hillel, T.; Elshafie, M. Z. E. B.; Jin, Y. (2018)\
[12] [ResLogit: A residual neural network logit model for data-driven choice modelling](https://doi.org/10.1016/j.trc.2021.103050), Wong, M.; Farooq, B. (2021)\
[13] [Enhancing Discrete Choice Models with Representation Learning](https://arxiv.org/abs/1812.09747), Sifringer, B.; Lurkin, V.; Alahi, A. (2018)\
[14] [A Customer Choice Model with HALO Effect](https://arxiv.org/abs/1805.01603), Maragheh, R., Y.; Chronopoulou, A.; Davis, J., M. (2018)\
[15] [Modeling Choice via Self-Attention](https://arxiv.org/abs/2311.07607), Ko, J.; Li, A., A. (2023)\
[16] [SHOPPER: A Probabilistic Model of Consumer Choice with Substitutes and Complements](https://arxiv.org/abs/1711.03560), Ruiz, F. J. R.; Athey, S.; Blei, D. M. (2019)

### Code and Repositories

*Official models implementations:*

[1] [RUMnet](https://github.com/antoinedesir/rumnet)\
[7] TasteNet [[Repo1](https://github.com/YafeiHan-MIT/TasteNet-MNL)] [[Repo2](https://github.com/deborahmit/TasteNet-MNL)]\
[12] [ResLogit](https://github.com/LiTrans/reslogit)\
[13] [Learning-MNL](https://github.com/BSifringer/EnhancedDCM)\
[16] [SHOPPER](https://github.com/franrruiz/shopper-src)
