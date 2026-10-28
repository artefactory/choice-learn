"""Tests basic stuff for the latent class models."""

import pytest
import tensorflow as tf

from choice_learn.datasets import load_electricity, load_modecanada
from choice_learn.models.latent_class_mnl import LatentClassConditionalLogit

tf.config.run_functions_eagerly(True)


def test_latent_class_mnl_instantiation():
    """Test the instantiation of the LCCL model."""
    elec_dataset = load_electricity(as_frame=False)
    lc_model = LatentClassConditionalLogit(
        n_latent_classes=3, fit_method="mle", optimizer="sgd", epochs=2, lbfgs_tolerance=1e-8
    )
    lc_model.add_coefficients(coefficient_name="pf", feature_name="pf", items_indexes=[0, 1, 2, 3])
    lc_model.add_shared_coefficient(
        coefficient_name="loc", feature_name="loc", items_indexes=[0, 1, 2, 3]
    )
    _ = lc_model.fit(elec_dataset)
    assert isinstance(lc_model.optimizer, tf.keras.optimizers.SGD)


def test_optimizers():
    """Test the different optimizers for the latent class models."""
    elec_dataset = load_electricity(as_frame=False)
    lc_model = LatentClassConditionalLogit(
        n_latent_classes=3, fit_method="mle", optimizer="adamax", epochs=2, lbfgs_tolerance=1e-8
    )
    lc_model.add_coefficients(coefficient_name="pf", feature_name="pf", items_indexes=[0, 1, 2, 3])
    _ = lc_model.fit(elec_dataset)
    assert isinstance(lc_model.optimizer, tf.keras.optimizers.Adamax)
    lc_model = LatentClassConditionalLogit(
        n_latent_classes=3, fit_method="mle", optimizer="truc", epochs=2, lbfgs_tolerance=1e-8
    )
    lc_model.add_coefficients(coefficient_name="pf", feature_name="pf", items_indexes=[0, 1, 2, 3])
    _ = lc_model.fit(elec_dataset)
    assert isinstance(lc_model.optimizer, tf.keras.optimizers.Adam)


def test_errors_lcmnl():
    """Test the errors raised by the latent class MNL model."""
    tf.config.run_functions_eagerly(True)
    # Instantiation with the coefficients dictionnary
    coefficients = {
        "income": "item",
        "cost": "constant",
        "freq": "constant",
        "ovt": "constant",
        "ivt": "item-full",
        "intercept": "item",
    }

    canada_dataset = load_modecanada(as_frame=False, preprocessing="tutorial")[:10]
    model = LatentClassConditionalLogit(
        n_latent_classes=3,
        fit_method="mle",
        optimizer="sgd",
        epochs=2,
        lbfgs_tolerance=1e-8,
        coefficients=coefficients,
    )
    with pytest.raises(ValueError):
        model.add_coefficients(
            coefficient_name="freq2", feature_name="freq", items_indexes=[0, 1, 2, 3]
        )

    with pytest.raises(ValueError):
        model.add_shared_coefficient(
            coefficient_name="freq2", feature_name="freq", items_indexes=[0, 1]
        )
    model.instantiate(canada_dataset)
    model.fit(canada_dataset)
