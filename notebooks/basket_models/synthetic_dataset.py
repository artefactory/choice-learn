"""Synthetic dataset for basket models."""

import numpy as np

from choice_learn.basket_models import Trip, TripDataset


def get_dataset(num_baskets: int = 100) -> TripDataset:
    """Return synthetic dataset.

    Parameters
    ----------
    num_baskets: int
        Number of baskets to generate.

    Returns
    -------
    TripDataset
        A synthetic dataset.
    """
    # Create a list of Trip objects:
    n_items = 7
    purchases_stores_1 = [[1, 0], [2, 0], [1, 3, 4, 0]]
    purchases_stores_2 = [[1, 0], [6, 5, 0]]

    assortment_store_1 = np.array([1, 1, 1, 1, 1, 0, 0])
    assortment_store_2 = np.array([1, 1, 0, 0, 0, 1, 1])
    available_items = np.array([assortment_store_1, assortment_store_2])

    trips_list = []

    for _ in range(num_baskets):
        trips_list += [
            Trip(
                purchases=purchases_stores_1[0],
                # Let's consider here totally random prices for the products
                prices=np.random.uniform(1, 10, n_items),
                assortment=0,
            ),
            Trip(
                purchases=purchases_stores_1[1],
                prices=np.random.uniform(1, 10, n_items),
                assortment=0,
            ),
            Trip(
                purchases=purchases_stores_1[2],
                prices=np.random.uniform(1, 10, n_items),
                assortment=0,
            ),
            Trip(
                purchases=purchases_stores_2[0],
                prices=np.random.uniform(1, 10, n_items),
                assortment=1,
            ),
            Trip(
                purchases=purchases_stores_2[1],
                prices=np.random.uniform(1, 10, n_items),
                assortment=1,
            ),
        ]

    return TripDataset(trips=trips_list, available_items=available_items)
