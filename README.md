<div align="center">

# choice-learn-private

[![CI status](https://github.com/artefactory/choice-learn-private/actions/workflows/ci.yaml/badge.svg)](https://github.com/artefactory/choice-learn-private/actions/workflows/ci.yaml?query=branch%3Amain)
[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue.svg)]()

[![Linting , formatting, imports sorting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/choice-learn-private/blob/main/.pre-commit-config.yaml)
</div>

Choice-Learn is a Python package designed to help you build with ease discrete choice models.
The package provides ready to use datasets and different models from the litterature. It also provides a lower level use if you want to customize any model or create your own from scratch. In particular you will find smart datasets handling to limit RAM usage and different structure commons to any choice model.

This repository contains a private version of the package.

## Table of Contents

- [choice-learn-private](#choice-learn-private)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Documentation](#documentation)
  - [Repository Structure](#repository-structure)

## Installation

To install the required packages in a virtual environment, run the following command:

** pip-install not possible yet, to come soon**
```bash
pip install choice-learn
```

## Usage

## Documentation

TODO: Github pages is not enabled by default, you need to enable it in the repository settings: Settings > Pages > Source: "Deploy from a branch" / Branch: "gh-pages" / Folder: "/(root)"

A detailed documentation of this project is available [here](https://artefactory.github.io/choice-learn-private/)

To serve the documentation locally, run the following command:

```bash
mkdocs serve
```

To build it and deploy it to GitHub pages, run the following command:

```bash
make deploy_docs
```

## Repository Structure

```
.
├── .github    <- GitHub Actions workflows and PR template
├── bin        <- Bash files
├── config     <- Configuration files
├── docs       <- Documentation files (mkdocs)
├── lib        <- Python modules
├── notebooks  <- Jupyter notebooks
├── secrets    <- Secret files (ignored by git)
└── tests      <- Unit tests
```
