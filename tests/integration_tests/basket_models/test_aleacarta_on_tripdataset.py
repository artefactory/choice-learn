"""Integration tests for Shopper model and TripDataset."""

import itertools
import logging

import numpy as np
import pytest
import tensorflow as tf

from choice_learn.basket_models import AleaCarta
from choice_learn.basket_models.data import Trip, TripDataset

# Toy dataset 1: different items between trips
trip_list_1 = [
    Trip(
        purchases=[7, 4, 8, 0],
        store=0,
        week=0,
        prices=[120, 130, 140, 150, 140, 160, 170, 100, 200, 180, 190, 210, 220, 230, 240],
        assortment=0,
    ),
    Trip(
        purchases=[2, 1, 3],
        store=3,
        week=5,
        prices=[200, 140, 110, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260],
        assortment=1,
    ),
    Trip(
        purchases=[1, 7, 3],
        store=1,
        week=2,
        prices=[100, 110, 120, 200, 130, 140, 150, 170, 180, 190, 200, 210, 150, 140, 170],
        assortment=0,
    ),
    Trip(
        purchases=[5, 6, 2, 0],
        store=2,
        week=19,
        prices=[110, 120, 150, 130, 140, 100, 140, 150, 160, 170, 180, 190, 90, 110, 140],
        assortment=2,
    ),
    Trip(
        purchases=[8, 1, 9],
        store=3,
        week=34,
        prices=[100, 140, 150, 160, 170, 180, 190, 200, 85, 200, 210, 220, 150, 170, 130],
        assortment=3,
    ),
    Trip(
        purchases=[10, 4, 11],
        store=1,
        week=51,
        prices=[130, 140, 150, 160, 170, 180, 190, 200, 220, 100, 210, 240, 170, 110, 90],
        assortment=0,
    ),
]
assortment_1, assortment_2, assortment_3, assortment_4 = (
    np.ones(15),
    np.ones(15),
    np.ones(15),
    np.ones(15),
)
assortment_2[13:] = 0
assortment_3[:3] = 0
assortment_4[4], assortment_4[11] = 0, 0
available_items_1 = np.array(
    [
        assortment_1,
        assortment_2,
        assortment_3,
        assortment_4,
    ]
)
trip_dataset_1 = TripDataset(trips=trip_list_1, available_items=available_items_1)
n_items_1 = trip_dataset_1.n_items
n_stores_1 = trip_dataset_1.n_stores

# Toy dataset 2: all the possibilities for an assortment of 3 items
# with store, week and prices fixed
trip_list_2 = [
    Trip(
        purchases=list(p),
        store=0,
        week=0,
        prices=[170, 110, 150],
        assortment=0,
    )
    for p in itertools.permutations([0, 1, 2])
]
# One more item available in the assortment to be able to use negative sampling
available_items_2 = np.expand_dims(np.ones(3), axis=0)
trip_dataset_2 = TripDataset(trips=trip_list_2, available_items=available_items_2)
n_items_2 = trip_dataset_2.n_items
n_stores_2 = trip_dataset_2.n_stores


def test_item_probabilities_sum_to_1() -> None:
    """Test that the item probabilities sum to 1."""
    model = AleaCarta(
        item_intercept=True,
        price_effects=True,
        seasonal_effects=True,
    )
    model.instantiate(
        n_items=n_items_1,
        n_stores=n_stores_1,
    )
    model.fit(trip_dataset=trip_dataset_1, val_dataset=trip_dataset_1)

    for trip in trip_dataset_1.trips:
        # For a given trip, check at each step that the sum of the probabilities for each
        # item to be the next purchased item given the items already purchased in the basket is 1
        for step in range(len(trip.purchases)):
            assert (
                np.abs(
                    np.sum(
                        model.compute_item_likelihood(
                            basket=trip.purchases[:step],
                            available_items=np.ones(n_items_1),
                            store=trip.store,
                            week=trip.week,
                            prices=trip.prices,
                        )
                    )
                    - 1.0
                )
                < 1e-4
            )


def test_ordered_basket_probabilities_sum_to_1() -> None:
    """Test that the ordered basket probabilities sum to 1."""
    model = AleaCarta(
        item_intercept=True,
        price_effects=False,
        seasonal_effects=True,
        latent_sizes={"preferences": 2, "price": 2, "season": 2},
        n_negative_samples=1,
    )
    model.instantiate(n_items=n_items_2, n_stores=n_stores_2)
    # For a basket {1, 2, 0} of size 3:
    # compute_ordered_basket_likelihood = 1/3 * 1/2 * 1/1 = 1/6

    assert (
        np.abs(
            np.sum(
                [
                    model.compute_ordered_basket_likelihood(
                        basket=trip.purchases,
                        available_items=np.ones((trip_dataset_2.n_items,)),
                        store=trip.store,
                        week=trip.week,
                        prices=trip.prices,
                    )
                    for trip in trip_dataset_2.trips
                ]
            )
            - 1.0
        )
        < 2e-2
    )


def test_no_intercept() -> None:
    """Test the Shopper model without item intercepts."""
    model = AleaCarta(item_intercept=False)
    model.instantiate(
        n_items=n_items_1,
        n_stores=n_stores_1,
    )

    batch_size = 4
    model.compute_batch_utility(
        item_batch=np.array([4, 5, 6, 0]),
        basket_batch=np.array([[1, 2, 3]] * batch_size),
        store_batch=np.array([0] * batch_size),
        week_batch=np.array([0] * batch_size),
        price_batch=np.random.uniform(1, 10, batch_size),
    )


def test_compute_item_likelihood() -> None:
    """Test the compute_item_likelihood method."""
    model = AleaCarta()
    model.instantiate(
        n_items=n_items_1,
        n_stores=n_stores_1,
    )

    with pytest.raises(ValueError):
        # Trip not provided as an argument
        # Then basket, available_items, store, week and prices must be provided
        model.compute_item_likelihood(
            basket=np.array([1, 2, 3]),
            available_items=np.ones(n_items_1),
            store=0,
            week=0,
        )

    with pytest.raises(ValueError):
        # Trip directly provided as an argument
        # Then trip.assortment must be an np.ndarray
        trip = Trip(
            purchases=[1, 2, 3],
            store=0,
            week=0,
            prices=np.random.uniform(1, 10, n_items_1),
            assortment=0,
        )
        model.compute_item_likelihood(trip=trip)


def test_compute_ordered_basket_likelihood() -> None:
    """Test the compute_ordered_basket_likelihood method."""
    model = AleaCarta()
    model.instantiate(
        n_items=n_items_1,
        n_stores=n_stores_1,
    )

    with pytest.raises(ValueError):
        # Trip not provided as an argument
        # Then basket, available_items, store, week and prices must be provided
        model.compute_ordered_basket_likelihood(
            basket=np.array([1, 2, 0]),
            available_items=np.ones(n_items_1),
            store=0,
            week=0,
        )

    with pytest.raises(ValueError):
        # Trip directly provided as an argument
        # Then trip.assortment must be an np.ndarray
        trip = Trip(
            purchases=[1, 2, 0],
            store=0,
            week=0,
            prices=np.random.uniform(1, 10, n_items_1),
            assortment=0,
        )
        model.compute_ordered_basket_likelihood(trip=trip)


def test_compute_basket_likelihood(caplog) -> None:
    """Test the compute_basket_likelihood method."""
    model = AleaCarta()
    model.instantiate(
        n_items=n_items_1,
        n_stores=n_stores_1,
    )

    with pytest.raises(ValueError):
        # Trip not provided as an argument
        # Then basket, available_items, store, week and prices must be provided
        model.compute_basket_likelihood(
            basket=np.array([1, 2, 0]),
            available_items=np.ones(n_items_1),
            store=0,
            week=0,
        )

    with pytest.raises(ValueError):
        # Trip directly provided as an argument
        # Then trip.assortment must be an np.ndarray
        trip = Trip(
            purchases=[1, 2, 0],
            store=0,
            week=0,
            prices=np.random.uniform(1, 10, n_items_1),
            assortment=0,
        )
        model.compute_basket_likelihood(trip=trip)

    # With verbose
    model.compute_basket_likelihood(
        basket=np.array([1, 2, 0]),
        available_items=np.ones(n_items_1),
        store=0,
        week=0,
        prices=np.random.uniform(1, 10, n_items_1),
        verbose=1,
    )

    # Too many permutations
    with caplog.at_level(logging.WARNING):
        model.compute_basket_likelihood(
            basket=np.array([1, 2, 0]),
            available_items=np.ones(n_items_1),
            store=0,
            week=0,
            prices=np.random.uniform(1, 10, n_items_1),
            n_permutations=3,  # > 2! = 2
        )
        assert "Warning: n_permutations > n! (all permutations)." in caplog.text


def test_get_negative_samples() -> None:
    """Test the get_negative_samples method."""
    model = AleaCarta()
    model.instantiate(
        n_items=n_items_1,
        n_stores=n_stores_1,
    )

    with pytest.raises(tf.errors.InvalidArgumentError):
        neg_samples = model.get_negative_samples(
            available_items=np.ones(n_items_1),
            purchased_items=np.array([1, 2]),
            next_item=0,
            n_samples=n_items_1,  # Too many samples
        )
        for item in [0, 1, 2]:
            assert item not in neg_samples


def test_fit() -> None:
    """Test the fit method."""
    model = AleaCarta(batch_size=-1)
    model.instantiate(
        n_items=n_items_1,
        n_stores=n_stores_1,
    )
    # Test lazy instantiation + verbose + batch_size=-1
    model.fit(trip_dataset=trip_dataset_1, val_dataset=trip_dataset_1, verbose=1)


def test_evaluate_load_and_save() -> None:
    """Test evaluate endpoint."""
    model = AleaCarta(
        item_intercept=True,
        price_effects=False,
        seasonal_effects=True,
        latent_sizes={"preferences": 2, "price": 2, "season": 2},
    )
    model.instantiate(
        n_items=n_items_1,
        n_stores=n_stores_1,
    )
    eff_loss = model.evaluate(trip_dataset=trip_dataset_1,)
    model.save_model("test_aleacarta")
    loaded_model = AleaCarta.load_model("test_aleacarta")
    loaded_loss = loaded_model.evaluate(trip_dataset=trip_dataset_1, )
    for w1, w2 in zip(model.trainable_weights, loaded_model.trainable_weights):
        assert np.allclose(w1.numpy(), w2.numpy())
    assert np.isclose(eff_loss, loaded_loss)
