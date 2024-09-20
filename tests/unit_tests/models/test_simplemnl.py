"""Tests for the SimpleMNL model."""

import numpy as np

from choice_learn.data import ChoiceDataset
from choice_learn.models import SimpleMNL

test_dataset = ChoiceDataset(
    shared_features_by_choice=np.array([[1, 3, 0], [0, 3, 1], [3, 2, 1], [3, 3, 1]]),
    items_features_by_choice=np.array(
        [
            [[1.1, 2.2], [2.9, 3.3], [3.3, 4.4]],
            [[1.2, 3.3], [2.3, 2.2], [4.3, 4.5]],
            [[1.4, 3.1], [2.4, 4.5], [3.4, 2.1]],
            [[1.7, 3.3], [2.3, 4.4], [3.7, 2.2]],
        ]
    ),
    available_items_by_choice=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0]]),
    choices=[0, 1, 2, 0],
)


def test_simplemnl_instantiation():
    """Tests SimpleMNL instantiation."""
    model = SimpleMNL(intercept="abc")
    model.instantiate(n_items=4, n_items_features=10, n_shared_features=20)
    assert True


def test_fit_lbfgs():
    """Tests instantiation with item-full and fit with lbfgs."""
    model = SimpleMNL(intercept="item-full", optimizer="lbfgs", epochs=20)
    model.instantiate(n_items=3, n_items_features=2, n_shared_features=3)
    nll_b = model.evaluate(test_dataset)
    model.fit(test_dataset, get_report=True)
    nll_a = model.evaluate(test_dataset)
    assert nll_a < nll_b

    assert model.compute_batch_utility(
        shared_features_by_choice=np.array([[1, 3, 0], [0, 3, 1], [3, 3, 1], [3, 3, 1]]),
        items_features_by_choice=np.array(
            [
                [[1.1, 2.2], [2.9, 3.3], [3.3, 4.4]],
                [[1.2, 3.3], [2.3, 2.2], [4.3, 4.4]],
                [[1.4, 3.3], [2.4, 4.5], [3.4, 2.2]],
                [[1.7, 3.3], [2.3, 4.4], [3.7, 2.2]],
            ]
        ),
        available_items_by_choice=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0]]),
        choices=[0, 1, 2, 0],
    ).shape == (4, 3)


def test_fit_adam():
    """Tests instantiation with item and fit with Adam."""
    model = SimpleMNL(intercept="item", optimizer="Adam", epochs=100, lr=0.1)
    model.instantiate(n_items=3, n_items_features=2, n_shared_features=3)
    nll_b = model.evaluate(test_dataset)
    model.fit(test_dataset, get_report=True)
    nll_a = model.evaluate(test_dataset)
    assert nll_a < nll_b

    assert model.report.to_numpy().shape == (7, 5)
