"""Unit tests for the Shopper model."""

import logging

import pytest

from choice_learn.basket_models import Shopper


def test_init_errors_warnings(caplog):
    """Test raised errors and warnings when initializing a Shopper object with wrong parameters."""
    with caplog.at_level(logging.WARNING):
        # No "preferences" key in latent_sizes dict
        Shopper(latent_sizes={"price": 10, "season": 10})
        assert "No latent size value has been specified for preferences" in caplog.text

    with caplog.at_level(logging.WARNING):
        # No "price" key in latent_sizes dict while price_effects=True
        Shopper(price_effects=True, latent_sizes={"preferences": 10, "season": 10})
        assert "No latent size value has been specified for price_effects" in caplog.text

    with caplog.at_level(logging.WARNING):
        # No "season" key in latent_sizes dict while seasonal_effects=True
        Shopper(seasonal_effects=True, latent_sizes={"preferences": 10, "price": 10})
        assert "No latent size value has been specified for seasonal_effects" in caplog.text

    with pytest.raises(ValueError):
        # Unknown key in latent_sizes dict.
        Shopper(latent_sizes={"preferences": 10, "price": 10, "season": 10, "unknown": 10})

    with pytest.raises(ValueError):
        # Unknown key in latent_sizes dict.
        Shopper(n_negative_samples=0)


def test_optimizers(caplog):
    """Test the different optimizers available for the Shopper model."""
    Shopper(optimizer="amsgrad")
    Shopper(optimizer="adamax")
    Shopper(optimizer="rmsprop")
    Shopper(optimizer="sgd")

    with caplog.at_level(logging.WARNING):
        Shopper(optimizer="not_implemented")
        assert (
            "Optimizer not_implemented not implemented, switching for default Adam" in caplog.text
        )
