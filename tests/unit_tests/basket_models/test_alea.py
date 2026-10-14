"""Contain unit tests for the AleaCarta model."""

import numpy as np
import pytest

from choice_learn.basket_models import AleaCarta
from choice_learn.basket_models.data import Trip


def test_instantiation():
    """Test the different instantion issues and all."""
    model = AleaCarta(
        item_intercept=True,
        price_effects=True,
        seasonal_effects=True,
        n_negative_samples=50,
        lr=0.001,
        epochs=10,
        batch_size=512,
        latent_sizes={},
    )
    assert "preferences" in model.latent_sizes.keys()
    print(model.latent_sizes)
    assert "price" in model.latent_sizes.keys()
    assert "season" in model.latent_sizes.keys()

    with pytest.raises(ValueError):
        model = AleaCarta(
            item_intercept=True,
            price_effects=False,
            seasonal_effects=False,
            n_negative_samples=-50,
            lr=0.001,
            epochs=10,
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
        epochs=10,
        batch_size=512,
        latent_sizes={"preferences": 2, "price": 0, "season": 0},
    )

    model.instantiate(n_items=10, n_stores=2)

    model.compute_basket_utility(
        trip=Trip(purchases=[1, 2], assortment=np.ones((10,)), prices=np.zeros((10,))),
    )

    with pytest.raises(ValueError):
        model.compute_basket_utility(
            trip=None,
        )


def test_compute_errors():
    """Test a few raised errors from the model."""

    with pytest.raises(ValueError):
        model = AleaCarta(
            item_intercept=True,
            price_effects=False,
            seasonal_effects=False,
            n_negative_samples=50,
            lr=0.001,
            epochs=10,
            batch_size=512,
            latent_sizes={"pref": 2, "price": 0, "season": 0},
        )
    model = AleaCarta(
        item_intercept=True,
        price_effects=False,
        seasonal_effects=False,
        n_negative_samples=50,
        lr=0.001,
        epochs=10,
        batch_size=512,
        latent_sizes={"preferences": 2, "price": 0, "season": 0},
    )
    model.instantiate(n_items=10, n_stores=2)

    with pytest.raises(ValueError):
        model.compute_batch_utility(
            item_batch=np.array([1, 2]),
            basket_batch=np.array([[0], [1]]),
            store_batch=np.array([0, 0]),
            week_batch=np.array([0, 0]),
            price_batch=np.zeros((2, 10)),
            available_item_batch=np.ones((2, 10)),
            user_batch=np.array([0, 0]),
        )
