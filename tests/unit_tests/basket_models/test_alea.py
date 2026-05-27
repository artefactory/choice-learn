"""Contain unit tests for the AleaCarta model."""

import numpy as np
import pytest

from choice_learn.basket_models import AleaCarta
from choice_learn.basket_models.data import Trip


def test_instantiation():
    """Test the different instantion issues and all."""
    model = AleaCarta(
        item_intercept=True,
        price_effects=False,
        seasonal_effects=False,
        n_negative_samples=50,
        lr=0.001,
        n_epochs=10,
        batch_size=512,
        latent_sizes={},
    )
    assert "preferences" in model.latent_sizes.keys()
    assert "price" in model.latent_sizes.keys()
    assert "season" in model.latent_sizes.keys()

    with pytest.raises(ValueError):
        model = AleaCarta(
            item_intercept=True,
            price_effects=False,
            seasonal_effects=False,
            n_negative_samples=-50,
            lr=0.001,
            n_epochs=10,
            batch_size=512,
            latent_sizes={},
        )

    model.instantiate(n_items=100, n_stores=0)


def test_compute_basket_utility():
    """Test that the compute_basket_utility method runs."""
    model = AleaCarta(
        item_intercept=True,
        price_effects=False,
        seasonal_effects=False,
        n_negative_samples=50,
        lr=0.001,
        n_epochs=10,
        batch_size=512,
        latent_sizes={"preferences": 2, "price": 0, "season": 0},
    )

    model.instantiate(n_items=10, n_stores=2)

    model.compute_basket_utility(
        Trip(purchases=[1, 2], assortment=np.ones((10,)), prices=np.zeros((10,))),
        store_id=0,
        season_id=0,
    )
