"""Basic tests for the Nested Logit model."""

import numpy as np
import pytest
import tensorflow as tf

from choice_learn.data import ChoiceDataset
from choice_learn.models import NestedLogit

test_dataset = ChoiceDataset(
    shared_features_by_choice=(np.array([[1, 0.3, 0.7], [0, 3.2, 1], [3.9, 2, 1], [2.9, 3.4, 1]]),),
    items_features_by_choice=(
        np.array(
            [
                [[1.1, 2.2], [2.9, 3.3], [5.3, 4.4]],
                [[1.2, 5.3], [1.3, 6.2], [4.3, 4.5]],
                [[2.4, 3.1], [2.4, 7.5], [3.4, 7.1]],
                [[1.7, 3.3], [8.3, 4.4], [2.7, 1.2]],
            ]
        ),
    ),
    items_features_by_choice_names=(["if1", "if2"],),
    shared_features_by_choice_names=(["sf1", "sf2", "sf3"],),
    available_items_by_choice=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0]]),
    choices=[0, 1, 2, 0],
)


def test_error_nest():
    """Tests that at least 2 nests are needed."""
    spec = {
        "sf1": "item",
        "sf2": "item",
        "sf3": "item",
        "if1": "constant",
        "if2": "constant",
    }
    with pytest.raises(ValueError):
        model = NestedLogit(
            coefficients=spec,
            items_nests=[[0, 1, 2]],
            optimizer="lbfgs",
            shared_gammas_over_nests=True,
        )
    with pytest.raises(ValueError):
        model = NestedLogit(
            coefficients=spec,
            items_nests=[[0, 1], [2], []],
            optimizer="lbfgs",
            shared_gammas_over_nests=True,
        )
    with pytest.raises(ValueError):
        model = NestedLogit(
            coefficients=spec,
            items_nests=[[0, 1], [2], [4]],
            optimizer="lbfgs",
            shared_gammas_over_nests=True,
        )
    with pytest.raises(ValueError):
        model = NestedLogit(
            coefficients=spec,
            items_nests=[[0, 1], [2], [0]],
            optimizer="lbfgs",
            shared_gammas_over_nests=True,
        )
    with pytest.raises(ValueError):
        model = NestedLogit(
            coefficients=spec,
            items_nests=[[0, 1], [2]],
            optimizer="lbfgs",
            shared_gammas_over_nests=True,
        )
        model.add_shared_coefficient(feature_name="sf1", items_indexes=[0, 1, 2])
    assert True


def test_fit_adam():
    """Tests the Nested Logit fit with Adam on dummy dataset."""
    global test_dataset
    tf.config.run_functions_eagerly(True)

    model = NestedLogit(
        items_nests=[[0, 1], [2]],
        optimizer="Adam",
        epochs=3,
        batch_size=-1,
        shared_gammas_over_nests=True,
    )
    model.add_coefficients(feature_name="sf1", items_indexes=[0, 1, 2])
    model.add_shared_coefficient(feature_name="if1", items_indexes=[1, 2])

    model.instantiate(test_dataset)
    nll_b = model.evaluate(test_dataset)
    model.fit(test_dataset, get_report=True)
    nll_a = model.evaluate(test_dataset)
    assert nll_a < nll_b


def test_fit_adam_specific_specification():
    """Tests the Nested Logit fit with Adam on dummy dataset and specific specification."""
    global test_dataset
    tf.config.run_functions_eagerly(True)

    test_dataset_2 = ChoiceDataset(
        shared_features_by_choice=(
            np.array([[1, 0.3, 0.7], [0, 3.2, 1], [3.9, 2, 1], [2.9, 3.4, 1]]),
        ),
        items_features_by_choice=(
            np.array(
                [
                    [[1.1, 2.2], [2.9, 3.3], [5.3, 4.4], [5.2, 4.0]],
                    [[1.2, 5.3], [1.3, 6.2], [4.3, 4.5], [2.1, 8.6]],
                    [[2.4, 3.1], [9.4, 7.5], [3.4, 7.1], [7.3, 4.1]],
                    [[9.7, 3.3], [8.3, 4.4], [2.7, 1.2], [4.3, 1.2]],
                ]
            ),
        ),
        items_features_by_choice_names=(["if1", "if2"],),
        shared_features_by_choice_names=(["sf1", "sf2", "sf3"],),
        available_items_by_choice=np.array(
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        ),
        choices=[0, 1, 3, 2],
    )

    spec = {
        "sf1": "item",
        "if2": "constant",
        "sf2": "item-full",
        "if1": "nest",
    }

    model = NestedLogit(
        coefficients=spec,
        items_nests=[[0, 1], [2, 3]],
        optimizer="sgd",
        epochs=100,
        lr=1e-5,
        batch_size=-1,
        shared_gammas_over_nests=False,
        regularization="l2",
        regularization_strength=1e-5,
    )

    model.instantiate(test_dataset_2)

    spec = {
        "sf1": "item",
        "if2": "constant",
        "sf2": "item-full",
    }

    model = NestedLogit(
        coefficients=spec,
        items_nests=[[0, 1], [2, 3]],
        optimizer="sgd",
        epochs=2,
        lr=1e-5,
        batch_size=-1,
        shared_gammas_over_nests=False,
        regularization="l2",
        regularization_strength=1e-5,
    )

    model.instantiate(test_dataset_2)
    nll_b = model.evaluate(test_dataset_2)
    model.fit(test_dataset_2, get_report=True)
    nll_a = model.evaluate(test_dataset_2)
    assert nll_a < nll_b
