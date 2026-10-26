"""Tests for the MNL model."""

import shutil

import numpy as np
import pytest
import tensorflow as tf

from choice_learn.data import ChoiceDataset
from choice_learn.models import MultiNomialLogit

test_dataset = ChoiceDataset(
    shared_features_by_choice=(np.array([[1, 3, 0], [0, 3, 1], [3, 2, 1], [3, 3, 1]]),),
    items_features_by_choice=(
        np.array(
            [
                [[1.1, 2.2], [2.9, 3.3], [3.3, 4.4]],
                [[1.2, 3.3], [2.3, 2.2], [4.3, 4.5]],
                [[1.4, 3.1], [2.4, 4.5], [3.4, 2.1]],
                [[1.7, 3.3], [2.3, 4.4], [3.7, 2.2]],
            ]
        ),
    ),
    available_items_by_choice=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0]]),
    choices=[0, 1, 2, 0],
)

small_dataset = ChoiceDataset(
    shared_features_by_choice=np.array([[1.0], [1.0], [2.0]]),
    items_features_by_choice=np.array(
        [[[2.0, 1.0], [1.0, 2.0]], [[2.0, 1.0], [1.0, 2.0]], [[3.0, 3.0], [3.0, 3.0]]]
    ),
    available_items_by_choice=np.ones((3, 2)),
    choices=[0.0, 0.0, 1.0],
)
reduced_small_dataset = ChoiceDataset(
    shared_features_by_choice=np.array([[1.0], [2.0]]),
    items_features_by_choice=np.array([[[2.0, 1.0], [1.0, 2.0]], [[3.0, 3.0], [3.0, 3.0]]]),
    available_items_by_choice=np.ones((2, 2)),
    choices=[0.0, 1.0],
)


def test_simplemnl_instantiation():
    """Tests MNL instantiation."""
    model = MultiNomialLogit(intercept="abc")
    model.instantiate(n_items=4, n_items_features=10, n_shared_features=20)

    for weight_1, weight_2 in zip(model.get_weights(), model._trainable_weights):
        assert tf.reduce_all(tf.equal(weight_1, weight_2))


def test_fit_lbfgs():
    """Tests instantiation with item-full and fit with lbfgs."""
    tf.config.run_functions_eagerly(True)
    model = MultiNomialLogit(
        intercept="item-full",
        optimizer="lbfgs",
        epochs=20,
        regularization="l2",
        regularization_strength=0.01,
    )
    model.instantiate(n_items=3, n_items_features=2, n_shared_features=3)
    nll_b = model.evaluate(test_dataset)
    model.fit(test_dataset, get_report=True, verbose=3)
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
    tf.config.run_functions_eagerly(True)
    model = MultiNomialLogit(intercept="item", optimizer="Adam", epochs=100, lr=0.1)
    model.instantiate(n_items=3, n_items_features=2, n_shared_features=3)
    nll_b = model.evaluate(test_dataset)
    model.fit(test_dataset, get_report=True, val_dataset=test_dataset)
    nll_a = model.evaluate(test_dataset, batch_size=-1)
    assert nll_a < nll_b

    assert model.report.to_numpy().shape == (7, 5)


def test_fit_adam_weights():
    """Tests instantiation with item and fit with Adam."""
    tf.config.run_functions_eagerly(True)
    model = MultiNomialLogit(
        intercept="item",
        optimizer="Adam",
        epochs=100,
        lr=0.1,
        regularization="l1",
        regularization_strength=0.01,
    )
    model.instantiate(n_items=3, n_items_features=2, n_shared_features=3)
    nll_b = model.evaluate(test_dataset)
    model.fit(
        test_dataset,
        sample_weight=np.array([0.2, 0.4, 0.8, 1.0]),
        get_report=True,
        val_dataset=test_dataset,
        verbose=2,
    )
    nll_a = model.evaluate(test_dataset, batch_size=-1)
    nll_c = model.evaluate(test_dataset, batch_size=3)
    assert nll_a < nll_b
    assert np.isclose(nll_c, nll_a, atol=1e-5)

    assert model.report.to_numpy().shape == (7, 5)

    model = MultiNomialLogit(
        intercept="item",
        optimizer="Adam",
        epochs=10,
        lr=0.01,
        regularization="l1",
        regularization_strength=0.01,
    )
    model.instantiate(n_items=3, n_items_features=2, n_shared_features=3)
    nll_b = model.evaluate(test_dataset)
    model.fit(
        test_dataset,
        sample_weight=np.array([0.2, 0.4, 0.8, 1.0]),
        val_dataset=test_dataset,
        verbose=0,
    )
    nll_a = model.evaluate(test_dataset, batch_size=-1)
    assert nll_a < nll_b


def test_evaluate_weights():
    """Tests instantiation with item and fit with Adam."""
    tf.config.run_functions_eagerly(True)
    model = MultiNomialLogit(
        intercept="item",
        optimizer="Adam",
        epochs=100,
        lr=0.1,
        regularization="l1",
        regularization_strength=0.01,
    )
    model.instantiate(n_items=2, n_items_features=2, n_shared_features=1)
    nll_test = model.evaluate(small_dataset)

    nll_1 = model.evaluate(reduced_small_dataset, batch_size=-1, sample_weight=[2.0, 1.0])
    nll_1_2 = model.evaluate(reduced_small_dataset, batch_size=1, sample_weight=[2.0, 1.0])
    nll_2 = model.evaluate(small_dataset, batch_size=-1, sample_weight=[1.0, 1.0, 1.0])
    nll_3 = model.evaluate(small_dataset, batch_size=1, sample_weight=[1.0, 1.0, 1.0])
    assert np.isclose(nll_test, nll_1 * 2 / 3, atol=1e-5)
    assert np.isclose(nll_1_2, nll_1, atol=1e-5)
    assert np.isclose(nll_test, nll_2, atol=1e-5)
    assert np.isclose(nll_test, nll_3, atol=1e-5)

    nll_4 = model.evaluate(small_dataset, mode="optim")
    nll_5 = model.evaluate(small_dataset, mode="optim", sample_weight=[1.0, 1.0, 1.0])
    assert nll_4 >= nll_test
    assert np.isclose(nll_4, nll_5, atol=1e-5)


def test_save_load():
    """Tests instantiation with item and fit with Adam."""
    model = MultiNomialLogit(intercept="item", optimizer="Adam", epochs=100, lr=0.1)
    model.instantiate(n_items=3, n_items_features=2, n_shared_features=3)
    nll_b = model.evaluate(test_dataset)
    model.save_model("test_save")
    loaded_model = MultiNomialLogit.load_model("test_save")
    nll_a = loaded_model.evaluate(test_dataset)

    assert nll_a == nll_b
    shutil.rmtree("test_save")


def test_weighted_val_dataset():
    """Tests instantiation with item and fit with Adam."""
    tf.config.run_functions_eagerly(True)
    model = MultiNomialLogit(intercept="item", optimizer="Adam", epochs=100, lr=0.1)
    model.instantiate(n_items=3, n_items_features=2, n_shared_features=3)
    nll_b = model.evaluate(test_dataset)
    model.fit(
        test_dataset, get_report=True, val_dataset=(test_dataset, np.ones((len(test_dataset),)))
    )
    nll_a = model.evaluate(test_dataset, batch_size=-1)
    assert nll_a < nll_b

    assert model.report.to_numpy().shape == (7, 5)

    with pytest.raises(ValueError):
        model = MultiNomialLogit(intercept="item", optimizer="Adam", epochs=100, lr=0.1)
        model.instantiate(n_items=3, n_items_features=2, n_shared_features=3)
        model.fit(
            test_dataset,
            val_dataset=(
                test_dataset,
                np.ones((len(test_dataset),)),
                np.ones((len(test_dataset),)),
            ),
        )

    with pytest.raises(ValueError):
        model = MultiNomialLogit(intercept="item", optimizer="Adam", epochs=100, lr=0.1)
        model.instantiate(n_items=3, n_items_features=2, n_shared_features=3)
        model.fit(
            test_dataset,
            val_dataset={
                "dataset": test_dataset,
                "sample_weight": np.ones((len(test_dataset),)),
            },
        )


def test_seeding():
    """Tests that setting the random seed produces consistent results."""
    tf.config.run_functions_eagerly(True)
    seed = 42
    model_1 = MultiNomialLogit(intercept="item", optimizer="Adam", epochs=100, lr=0.1)
    model_1.instantiate(n_items=3, n_items_features=2, n_shared_features=3, seed=seed)
    nll_1 = model_1.evaluate(test_dataset)

    model_2 = MultiNomialLogit(intercept="item", optimizer="Adam", epochs=100, lr=0.1)
    model_2.instantiate(n_items=3, n_items_features=2, n_shared_features=3, seed=seed)
    nll_2 = model_2.evaluate(test_dataset)

    assert np.isclose(nll_1, nll_2, atol=1e-5)
