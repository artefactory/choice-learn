"""Tests basic stuff for the latent class models."""

import tensorflow as tf

from choice_learn.datasets import load_electricity
from choice_learn.models.latent_class_base_model import BaseLatentClassModel
from choice_learn.models.latent_class_mnl import LatentClassConditionalLogit, LatentClassSimpleMNL
from choice_learn.models.simple_mnl import SimpleMNL

elec_dataset = load_electricity(as_frame=False)


def test_latent_simple_mnl():
    """Test the simple latent class model fit() method."""
    tf.config.run_functions_eagerly(True)
    lc_model = LatentClassSimpleMNL(
        n_latent_classes=3, fit_method="mle", optimizer="lbfgs", epochs=1000, lbfgs_tolerance=1e-20
    )
    _, _ = lc_model.fit(elec_dataset)

    assert lc_model.evaluate(elec_dataset).numpy() < 1.15


def test_latent_clogit():
    """Test the conditional logit latent class model fit() method."""
    tf.config.run_functions_eagerly(True)
    lc_model = LatentClassConditionalLogit(
        n_latent_classes=3, fit_method="mle", optimizer="lbfgs", epochs=1000, lbfgs_tolerance=1e-12
    )
    lc_model.add_shared_coefficient(
        coefficient_name="pf", feature_name="pf", items_indexes=[0, 1, 2, 3]
    )
    lc_model.add_shared_coefficient(
        coefficient_name="cl", feature_name="cl", items_indexes=[0, 1, 2, 3]
    )
    lc_model.add_shared_coefficient(
        coefficient_name="loc", feature_name="loc", items_indexes=[0, 1, 2, 3]
    )
    lc_model.add_shared_coefficient(
        coefficient_name="wk", feature_name="wk", items_indexes=[0, 1, 2, 3]
    )
    lc_model.add_shared_coefficient(
        coefficient_name="tod", feature_name="tod", items_indexes=[0, 1, 2, 3]
    )
    lc_model.add_shared_coefficient(
        coefficient_name="seas", feature_name="seas", items_indexes=[0, 1, 2, 3]
    )
    _, _ = lc_model.fit(elec_dataset)

    assert lc_model.evaluate(elec_dataset).numpy() < 1.15


def test_manual_lc():
    """Test manual specification of Latent Class Simple MNL model."""
    tf.config.run_functions_eagerly(True)
    manual_lc = BaseLatentClassModel(
        model_class=SimpleMNL,
        model_parameters={"add_exit_choice": False},
        n_latent_classes=3,
        fit_method="mle",
        epochs=1000,
        optimizer="lbfgs",
        lbfgs_tolerance=1e-12,
    )

    manual_lc.instantiate(n_items=4, n_shared_features=0, n_items_features=6)
    _ = manual_lc.fit(elec_dataset)
    assert manual_lc.evaluate(elec_dataset) < 1.15

def test_manual_lc_gd():
    """Test manual specification of Latent Class Simple MNL model with gradient descent."""
    tf.config.run_functions_eagerly(True)
    manual_lc = BaseLatentClassModel(
        model_class=SimpleMNL,
        model_parameters={"add_exit_choice": False},
        n_latent_classes=3,
        fit_method="mle",
        epochs=1000,
        optimizer="Adam",
    )
    nll_before = manual_lc.evaluate(elec_dataset)
    manual_lc.instantiate(n_items=4, n_shared_features=0, n_items_features=6)
    _ = manual_lc.fit(elec_dataset)
    assert manual_lc.evaluate(elec_dataset) < nll_before
