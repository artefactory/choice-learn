"""Tests specific config of cLogit and .evaluate() with ground truth weight."""

import numpy as np

from choice_learn.data import ChoiceDataset
from choice_learn.datasets.base import load_heating
from choice_learn.models.halo_mnl import HaloMNL, LowRankHaloMNL

heating_df = load_heating(as_frame=True)

shared_features_by_choice = ["income", "agehed", "rooms"]
choice = ["depvar"]
items_features_by_choice = ["ic.", "oc."]
items = ["hp", "gc", "gr", "ec", "er"]

choices = np.array([items.index(val) for val in heating_df[choice].to_numpy().ravel()])
shared_features_by_choice = heating_df[shared_features_by_choice].to_numpy().astype("float32")
items_features_by_choice = np.stack(
    [heating_df[[feat + item for feat in items_features_by_choice]].to_numpy() for item in items],
    axis=1,
)
dataset = ChoiceDataset(items_features_by_choice=items_features_by_choice, choices=choices)


def test_halo():
    """Very basic test to check that the model runs."""
    model = HaloMNL(intercept="item", optimizer="lbfgs")
    _ = model.fit(dataset, verbose=0, get_report=True)
    assert True


def test_low_rank():
    """Very basic test to check that the model runs."""
    model = LowRankHaloMNL(halo_latent_dim=2, intercept=None)
    _ = model.fit(dataset, verbose=0, get_report=True)
    assert True
