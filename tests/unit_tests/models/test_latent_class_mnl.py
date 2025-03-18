"""Tests for the latent_class_mnl.py module."""

import numpy as np
import pytest
import tensorflow as tf

from choice_learn.data import ChoiceDataset
from choice_learn.models.latent_class_mnl import LatentClassSimpleMNL


# Create a simple test dataset
@pytest.fixture
def test_dataset():
    """Create a test dataset for testing."""
    return ChoiceDataset(
        shared_features_by_choice=(
            np.array(
                [
                    [1, 3, 0],
                    [0, 3, 1],
                    [3, 2, 1],
                    [3, 3, 1],
                    [1, 2, 2],
                    [2, 1, 3],
                    [0, 0, 1],
                    [2, 2, 2],
                ]
            ),
        ),
        items_features_by_choice=(
            np.array(
                [
                    [[1.1, 2.2], [2.9, 3.3], [3.3, 4.4]],
                    [[1.2, 3.3], [2.3, 2.2], [4.3, 4.5]],
                    [[1.4, 3.1], [2.4, 4.5], [3.4, 2.1]],
                    [[1.7, 3.3], [2.3, 4.4], [3.7, 2.2]],
                    [[1.5, 2.5], [2.5, 3.5], [3.5, 4.5]],
                    [[1.3, 3.2], [2.2, 2.9], [4.1, 4.2]],
                    [[1.6, 3.0], [2.1, 4.0], [3.2, 2.0]],
                    [[1.8, 3.4], [2.6, 4.6], [3.9, 2.3]],
                ]
            ),
        ),
        available_items_by_choice=np.ones((8, 3), dtype=np.int32),
        choices=[0, 1, 2, 0, 1, 2, 0, 1],
    )


def test_latent_class_mnl_instantiation():
    """Test that the LatentClassSimpleMNL model can be instantiated."""
    model = LatentClassSimpleMNL(
        n_latent_classes=2,
        fit_method="MLE",
        epochs=5,
        add_exit_choice=False,
        intercept="item",
        optimizer="Adam",
        lr=0.01,
    )

    assert model.n_latent_classes == 2
    assert model.fit_method == "MLE"
    assert model.epochs == 5
    assert model.add_exit_choice is False
    assert model.intercept == "item"
    assert model.optimizer == "Adam"
    assert model.lr == 0.01


def test_latent_class_mnl_instantiate(test_dataset):
    """Test that the LatentClassSimpleMNL model can be instantiated with dataset dimensions."""
    model = LatentClassSimpleMNL(n_latent_classes=2, fit_method="MLE", epochs=5)

    n_items = test_dataset.get_n_items()
    n_shared_features = test_dataset.get_n_shared_features()
    n_items_features = test_dataset.get_n_items_features()

    model.instantiate(n_items, n_shared_features, n_items_features)

    # Check that latent logits and models are created
    assert hasattr(model, "latent_logits")
    assert hasattr(model, "models")
    assert len(model.models) == 2  # n_latent_classes
    assert model.latent_logits.shape == (1,)  # n_latent_classes - 1


def test_latent_class_mnl_convergence(test_dataset):
    """Test that the LatentClassSimpleMNL model converges during training."""
    tf.config.run_functions_eagerly(True)

    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    model = LatentClassSimpleMNL(
        n_latent_classes=2,
        fit_method="MLE",  # Using MLE as it's faster for testing
        epochs=10,
        batch_size=4,
        add_exit_choice=False,
        intercept="item",
        optimizer="Adam",
        lr=0.1,  # Higher learning rate to converge faster in tests
    )

    n_items = test_dataset.get_n_items()
    n_shared_features = test_dataset.get_n_shared_features()
    n_items_features = test_dataset.get_n_items_features()

    model.instantiate(n_items, n_shared_features, n_items_features)
    model.instantiate_latent_models(n_items, n_shared_features, n_items_features)

    # Evaluate before training
    nll_before = model.evaluate(test_dataset)

    # Train the model
    model.fit(test_dataset, get_report=True)

    # Evaluate after training
    nll_after = model.evaluate(test_dataset)

    # Check that the model improved
    assert nll_after < nll_before, f"Model did not converge: {nll_before} -> {nll_after}"


def test_latent_class_mnl_prediction(test_dataset):
    """Test that the LatentClassSimpleMNL model can make predictions."""
    tf.config.run_functions_eagerly(True)

    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    model = LatentClassSimpleMNL(
        n_latent_classes=2, fit_method="MLE", epochs=5, batch_size=4, optimizer="Adam", lr=0.1
    )

    n_items = test_dataset.get_n_items()
    n_shared_features = test_dataset.get_n_shared_features()
    n_items_features = test_dataset.get_n_items_features()

    model.instantiate(n_items, n_shared_features, n_items_features)
    model.instantiate_latent_models(n_items, n_shared_features, n_items_features)

    # Train the model
    model.fit(test_dataset)

    # Make predictions
    predictions = model.predict(test_dataset)

    # Check predictions shape and values
    assert predictions.shape == (len(test_dataset), n_items)
    assert np.all(predictions >= 0) and np.all(predictions <= 1)
    assert np.allclose(np.sum(predictions, axis=1), 1.0, atol=1e-5)
