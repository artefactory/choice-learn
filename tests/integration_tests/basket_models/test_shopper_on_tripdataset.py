"""Integration tests for Shopper model and TripDataset."""

import logging

import numpy as np
import pytest

from choice_learn.basket_models import Shopper
from choice_learn.basket_models.dataset import Trip, TripDataset

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
        purchases=[2, 1, 3, 0],
        store=3,
        week=5,
        prices=[200, 140, 110, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260],
        assortment=1,
    ),
    Trip(
        purchases=[1, 7, 3, 0],
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
        purchases=[8, 1, 9, 0],
        store=3,
        week=34,
        prices=[100, 140, 150, 160, 170, 180, 190, 200, 85, 200, 210, 220, 150, 170, 130],
        assortment=3,
    ),
    Trip(
        purchases=[10, 4, 11, 0],
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
        purchases=[0],  # Empty basket
        store=0,
        week=0,
        prices=[1, 100, 170, 110, 150],
        assortment=0,
    ),
    Trip(
        purchases=[1, 0],
        store=0,
        week=0,
        prices=[1, 100, 170, 110, 150],
        assortment=0,
    ),
    Trip(
        purchases=[2, 0],
        store=0,
        week=0,
        prices=[1, 100, 170, 110, 150],
        assortment=0,
    ),
    Trip(
        purchases=[3, 0],
        store=0,
        week=0,
        prices=[1, 100, 170, 110, 150],
        assortment=0,
    ),
    Trip(
        purchases=[1, 2, 0],
        store=0,
        week=0,
        prices=[1, 100, 170, 110, 150],
        assortment=0,
    ),
    Trip(
        purchases=[1, 3, 0],
        store=0,
        week=0,
        prices=[1, 100, 170, 110, 150],
        assortment=0,
    ),
    Trip(
        purchases=[2, 1, 0],
        store=0,
        week=0,
        prices=[1, 100, 170, 110, 150],
        assortment=0,
    ),
    Trip(
        purchases=[2, 3, 0],
        store=0,
        week=0,
        prices=[1, 100, 170, 110, 150],
        assortment=0,
    ),
    Trip(
        purchases=[3, 1, 0],
        store=0,
        week=0,
        prices=[1, 100, 170, 110, 150],
        assortment=0,
    ),
    Trip(
        purchases=[3, 2, 0],
        store=0,
        week=0,
        prices=[1, 100, 170, 110, 150],
        assortment=0,
    ),
    Trip(
        purchases=[1, 2, 3, 0],
        store=0,
        week=0,
        prices=[1, 100, 170, 110, 150],
        assortment=0,
    ),
    Trip(
        purchases=[1, 3, 2, 0],
        store=0,
        week=0,
        prices=[1, 100, 170, 110, 150],
        assortment=0,
    ),
    Trip(
        purchases=[2, 1, 3, 0],
        store=0,
        week=0,
        prices=[1, 100, 170, 110, 150],
        assortment=0,
    ),
    Trip(
        purchases=[2, 3, 1, 0],
        store=0,
        week=0,
        prices=[1, 100, 170, 110, 150],
        assortment=0,
    ),
    Trip(
        purchases=[3, 1, 2, 0],
        store=0,
        week=0,
        prices=[1, 100, 170, 110, 150],
        assortment=0,
    ),
    Trip(
        purchases=[3, 2, 1, 0],
        store=0,
        week=0,
        prices=[1, 100, 170, 110, 150],
        assortment=0,
    ),
]
# One more item available in the assortment to be able to use negative sampling
available_items_2 = np.expand_dims(np.ones(5), axis=0)
trip_dataset_2 = TripDataset(trips=trip_list_2, available_items=available_items_2)
n_items_2 = trip_dataset_2.n_items
n_stores_2 = trip_dataset_2.n_stores


def test_item_probabilities_sum_to_1() -> None:
    """Test that the item probabilities sum to 1."""
    model = Shopper(
        item_intercept=True,
        price_effects=True,
        seasonal_effects=True,
        think_ahead=False,
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
    model = Shopper(
        item_intercept=True,
        price_effects=False,
        seasonal_effects=True,
        think_ahead=False,
        latent_sizes={"preferences": 2, "price": 2, "season": 2},
        n_negative_samples=1,
    )
    model.instantiate(n_items=n_items_2, n_stores=n_stores_2)
    model.fit(trip_dataset=trip_dataset_2)

    # For a basket {1, 2, 3, 0} of size 3:
    # compute_ordered_basket_likelihood = 1/3 * 1/3 * 1/2 * 1/1 = 1/18
    # (1/nb_possibilities but the checkout item is not considered during the 1st step)

    # List of all the possible availability matrices verifying 2 conditions
    # to get a basket probability > 0:
    # - The checkout item must be available
    # - The checkout item must not be the only item available
    # (because the proba of an empty basket is 0 and cannot sum to 1)
    list_availability_matrices = [
        np.array([1, 1, 1, 1, 1]),
        np.array([1, 0, 1, 1, 1]),
        np.array([1, 1, 0, 1, 1]),
        np.array([1, 1, 1, 0, 1]),
        np.array([1, 1, 1, 1, 0]),
        np.array([1, 0, 0, 0, 1]),
        np.array([1, 0, 0, 1, 0]),
        np.array([1, 0, 1, 0, 0]),
        np.array([1, 1, 0, 0, 0]),
    ]
    for availability_matrix in list_availability_matrices:
        # Try with different availability matrices
        assert (
            np.abs(
                np.sum(
                    [
                        model.compute_ordered_basket_likelihood(
                            basket=trip.purchases,
                            available_items=availability_matrix,
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


def test_thinking_ahead() -> None:
    """Test the Shopper model with thinking ahead."""
    model = Shopper(think_ahead=True)
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
        available_item_batch=np.array([np.ones(n_items_1)] * batch_size),
    )


def test_no_intercept() -> None:
    """Test the Shopper model without item intercepts."""
    model = Shopper(item_intercept=False)
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
        available_item_batch=np.array([np.ones(n_items_1)] * batch_size),
    )


def test_compute_item_likelihood() -> None:
    """Test the compute_item_likelihood method."""
    model = Shopper()
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
    model = Shopper()
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
    model = Shopper()
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
    model = Shopper()
    model.instantiate(
        n_items=n_items_1,
        n_stores=n_stores_1,
    )

    with pytest.raises(ValueError):
        model.get_negative_samples(
            available_items=np.ones(n_items_1),
            purchased_items=np.array([1, 2]),
            future_purchases=np.array([3, 0]),
            next_item=0,
            n_samples=n_items_1,  # Too many samples
        )


def test_fit() -> None:
    """Test the fit method."""
    model = Shopper(batch_size=-1)
    model.instantiate(
        n_items=n_items_1,
        n_stores=n_stores_1,
    )
    # Test lazy instantiation + verbose + batch_size=-1
    model.fit(trip_dataset=trip_dataset_1, val_dataset=trip_dataset_1, verbose=1)


def test_evaluate_load_and_save() -> None:
    """Test evaluate endpoint."""
    model = Shopper(
        item_intercept=True,
        price_effects=False,
        seasonal_effects=True,
        think_ahead=False,
        latent_sizes={"preferences": 2, "price": 2, "season": 2},
    )
    model.instantiate(
        n_items=n_items_1,
        n_stores=n_stores_1,
    )
    model.evaluate(trip_dataset=trip_dataset_1)
    model.save_model("test_model")
    _ = Shopper.load_model("test_model")
    assert True
