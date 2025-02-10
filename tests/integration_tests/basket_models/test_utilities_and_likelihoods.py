"""Integration tests for the utility and likelihood computation of a Shopper model."""

import numpy as np

from choice_learn.basket_models.shopper import Shopper
from choice_learn.basket_models.trip_dataset import Trip, TripDataset

# Toy dataset 1: different items between trips
trip_list_1 = [
    Trip(
        id=0,
        purchases=[7, 4, 8, 0],
        customer=0,
        week=0,
        prices=[120, 130, 140, 150, 140, 160, 170, 100, 200, 180, 190, 210, 220, 230, 240],
        assortment=0,
    ),
    Trip(
        id=1,
        purchases=[2, 1, 3, 0],
        customer=3,
        week=5,
        prices=[200, 140, 110, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260],
        assortment=1,
    ),
    Trip(
        id=2,
        purchases=[1, 7, 3, 0],
        customer=1,
        week=2,
        prices=[100, 110, 120, 200, 130, 140, 150, 170, 180, 190, 200, 210, 150, 140, 170],
        assortment=0,
    ),
    Trip(
        id=3,
        purchases=[5, 6, 2, 0],
        customer=2,
        week=19,
        prices=[110, 120, 150, 130, 140, 100, 140, 150, 160, 170, 180, 190, 90, 110, 140],
        assortment=2,
    ),
    Trip(
        id=4,
        purchases=[8, 1, 9, 0],
        customer=3,
        week=34,
        prices=[100, 140, 150, 160, 170, 180, 190, 200, 85, 200, 210, 220, 150, 170, 130],
        assortment=3,
    ),
    Trip(
        id=5,
        purchases=[10, 4, 11, 0],
        customer=1,
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
assortments_1 = np.array(
    [
        assortment_1,
        assortment_2,
        assortment_3,
        assortment_4,
    ]
)
trip_dataset_1 = TripDataset(trips=trip_list_1, assortments=assortments_1)
n_items_1 = trip_dataset_1.n_items
n_customers_1 = trip_dataset_1.n_customers

# Toy dataset 2: all the possibilities for an assortment of 3 items
# with customer, week and prices fixed
trip_list_2 = [
    Trip(
        id=0,
        purchases=[0],  # Empty basket
        customer=0,
        week=0,
        prices=[1, 100, 170, 110, 150],
        assortment=0,
    ),
    Trip(
        id=1,
        purchases=[1, 0],
        customer=0,
        week=0,
        prices=[1, 100, 170, 110, 150],
        assortment=0,
    ),
    Trip(
        id=2,
        purchases=[2, 0],
        customer=0,
        week=0,
        prices=[1, 100, 170, 110, 150],
        assortment=0,
    ),
    Trip(
        id=3,
        purchases=[3, 0],
        customer=0,
        week=0,
        prices=[1, 100, 170, 110, 150],
        assortment=0,
    ),
    Trip(
        id=4,
        purchases=[1, 2, 0],
        customer=0,
        week=0,
        prices=[1, 100, 170, 110, 150],
        assortment=0,
    ),
    Trip(
        id=5,
        purchases=[1, 3, 0],
        customer=0,
        week=0,
        prices=[1, 100, 170, 110, 150],
        assortment=0,
    ),
    Trip(
        id=6,
        purchases=[2, 1, 0],
        customer=0,
        week=0,
        prices=[1, 100, 170, 110, 150],
        assortment=0,
    ),
    Trip(
        id=7,
        purchases=[2, 3, 0],
        customer=0,
        week=0,
        prices=[1, 100, 170, 110, 150],
        assortment=0,
    ),
    Trip(
        id=8,
        purchases=[3, 1, 0],
        customer=0,
        week=0,
        prices=[1, 100, 170, 110, 150],
        assortment=0,
    ),
    Trip(
        id=9,
        purchases=[3, 2, 0],
        customer=0,
        week=0,
        prices=[1, 100, 170, 110, 150],
        assortment=0,
    ),
    Trip(
        id=10,
        purchases=[1, 2, 3, 0],
        customer=0,
        week=0,
        prices=[1, 100, 170, 110, 150],
        assortment=0,
    ),
    Trip(
        id=11,
        purchases=[1, 3, 2, 0],
        customer=0,
        week=0,
        prices=[1, 100, 170, 110, 150],
        assortment=0,
    ),
    Trip(
        id=12,
        purchases=[2, 1, 3, 0],
        customer=0,
        week=0,
        prices=[1, 100, 170, 110, 150],
        assortment=0,
    ),
    Trip(
        id=13,
        purchases=[2, 3, 1, 0],
        customer=0,
        week=0,
        prices=[1, 100, 170, 110, 150],
        assortment=0,
    ),
    Trip(
        id=14,
        purchases=[3, 1, 2, 0],
        customer=0,
        week=0,
        prices=[1, 100, 170, 110, 150],
        assortment=0,
    ),
    Trip(
        id=15,
        purchases=[3, 2, 1, 0],
        customer=0,
        week=0,
        prices=[1, 100, 170, 110, 150],
        assortment=0,
    ),
]
# One more item available in the assortment to be able to use negative sampling
assortments_2 = np.expand_dims(np.ones(5), axis=0)
trip_dataset_2 = TripDataset(trips=trip_list_2, assortments=assortments_2)
n_items_2 = trip_dataset_2.n_items
n_customers_2 = trip_dataset_2.n_customers


def test_item_probabilities_sum_to_1() -> None:
    """Test that the item probabilities sum to 1."""
    model = Shopper(
        item_popularity=True,
        price_effects=True,
        seasonal_effects=True,
        think_ahead=False,
        latent_sizes={"preferences": 10, "price": 10, "season": 10},
    )
    model.instantiate(
        n_items=n_items_1,
        n_customers=n_customers_1,
    )
    model.fit(trip_dataset=trip_dataset_1)

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
                            customer=trip.customer,
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
        item_popularity=True,
        price_effects=False,
        seasonal_effects=True,
        think_ahead=False,
        latent_sizes={"preferences": 2, "price": 2, "season": 2},
        n_negative_samples=1,
    )
    model.instantiate(n_items=n_items_2, n_customers=n_customers_2)
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
                            customer=trip.customer,
                            week=trip.week,
                            prices=trip.prices,
                        )
                        for trip in trip_dataset_2.trips
                    ]
                )
                - 1.0
            )
            < 1e-2
        )


def test_evaluate_load_and_save() -> None:
    """Test evaluate endpoint."""
    model = Shopper(
        item_popularity=True,
        price_effects=False,
        seasonal_effects=True,
        think_ahead=False,
        latent_sizes={"preferences": 2, "price": 2, "season": 2},
    )
    model.instantiate(
        n_items=n_items_1,
        n_customers=n_customers_1,
    )
    model.evaluate(trip_dataset=trip_dataset_1)
    model.save_model("test_model")
    _ = Shopper.load_model("test_model")
    return True
