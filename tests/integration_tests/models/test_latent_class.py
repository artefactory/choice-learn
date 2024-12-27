"""Tests basic stuff for the latent class models."""

import numpy as np
import tensorflow as tf

tf.config.run_functions_eagerly(True)

from choice_learn.datasets import load_electricity  # noqa: E402
from choice_learn.models.latent_class_base_model import BaseLatentClassModel  # noqa: E402
from choice_learn.models.latent_class_mnl import (  # noqa: E402
    LatentClassConditionalLogit,
    LatentClassSimpleMNL,
)
from choice_learn.models.simple_mnl import SimpleMNL  # noqa: E402

elec_dataset = load_electricity(as_frame=False)


def test_latent_simple_mnl():
    """Test the simple latent class model fit() method."""
    lc_model = LatentClassSimpleMNL(
        n_latent_classes=2, fit_method="mle", optimizer="lbfgs", epochs=1000, lbfgs_tolerance=1e-12
    )
    _, _ = lc_model.fit(elec_dataset)
    lc_model.compute_report(elec_dataset)

    probas = lc_model.predict_modelwise_probas(elec_dataset)
    assert probas.shape == (2, len(elec_dataset), 4)
    probas = lc_model.predict_probas(elec_dataset)
    assert probas.shape == (len(elec_dataset), 4)

    assert lc_model.evaluate(elec_dataset).numpy() < 1.15


def test_latent_clogit():
    """Test the conditional logit latent class model fit() method."""
    lc_model = LatentClassConditionalLogit(
        n_latent_classes=3, fit_method="mle", optimizer="lbfgs", epochs=40, lbfgs_tolerance=1e-8
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
    manual_lc = BaseLatentClassModel(
        model_class=SimpleMNL,
        model_parameters={"add_exit_choice": False},
        n_latent_classes=3,
        fit_method="mle",
        epochs=40,
        optimizer="lbfgs",
        lbfgs_tolerance=1e-8,
    )

    manual_lc.instantiate(n_items=4, n_shared_features=0, n_items_features=6)
    _ = manual_lc.fit(elec_dataset)
    assert manual_lc.evaluate(elec_dataset) < 1.15


def test_manual_lc_gd():
    """Test manual specification of Latent Class Simple MNL model with gradient descent."""
    manual_lc = BaseLatentClassModel(
        model_class=SimpleMNL,
        model_parameters={"add_exit_choice": False},
        n_latent_classes=3,
        fit_method="mle",
        epochs=10,
        optimizer="Adam",
    )
    manual_lc.instantiate(n_items=4, n_shared_features=0, n_items_features=6)
    nll_before = manual_lc.evaluate(elec_dataset)
    _ = manual_lc.fit(
        elec_dataset, sample_weight=np.ones(len(elec_dataset)), val_dataset=elec_dataset[-10:]
    )
    assert manual_lc.evaluate(elec_dataset) < nll_before


def test_em_fit():
    """Test EM algorithm to estimate Latent Class Model."""
    lc_model_em = LatentClassSimpleMNL(
        n_latent_classes=3, fit_method="EM", optimizer="lbfgs", epochs=15, lbfgs_tolerance=1e-6
    )
    lc_model_em.instantiate(
        n_items=elec_dataset.get_n_items(),
        n_shared_features=elec_dataset.get_n_shared_features(),
        n_items_features=elec_dataset.get_n_items_features(),
    )
    nll_b = lc_model_em.evaluate(elec_dataset)
    _, _ = lc_model_em.fit(elec_dataset, verbose=0)
    nll_a = lc_model_em.evaluate(elec_dataset)
    assert nll_a < nll_b
