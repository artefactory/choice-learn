"""Unit tests for the Trip and TripDataset classes."""

import numpy as np
import pytest

from choice_learn.basket_models import Trip, TripDataset

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


def test_getitem():
    """Test that getitem works correctly."""
    global trip_dataset_1
    assert isinstance(trip_dataset_1[0], TripDataset)
    assert isinstance(trip_dataset_1[0:2], TripDataset)
    assert isinstance(trip_dataset_1[[0, 1, 2]], TripDataset)
    with pytest.raises(TypeError):
        # Type of index must be int, list, np.ndarray, range or slice
        trip_dataset_1["a"]


def test_errors():
    """Test various raised errors."""
    with pytest.raises(ValueError):
        # Week number must be between 0 and 51, inclusive.
        Trip(week=100, purchases=np.array([0, 1, 2]), prices=np.array([1, 2, 3]), assortment=0)


def test_trip_dataset_methods():
    """Test few methods."""
    global trip_list_1
    global trip_dataset_1
    items = trip_list_1[0].get_items_up_to_index(2)
    assert items == [7, 4]

    _ = str(trip_dataset_1)

    for _ in trip_dataset_1:
        pass

    assert True


def test_trip_dataset_concat():
    """Test the concat operation."""
    global trip_dataset_1

    trip_dataset_2 = TripDataset(trips=trip_list_1, available_items=available_items_1)
    trip_dataset_3 = trip_dataset_1.concatenate(trip_dataset_2)
    assert len(trip_dataset_3) == 2 * len(trip_dataset_1)

    trip_dataset_2.concatenate(trip_dataset_1, inplace=True)
    assert len(trip_dataset_2) == len(trip_dataset_3)


def test_str():
    """Test the str method."""
    global trip_dataset_1

    assert isinstance(str(trip_dataset_1), str)


def test_get_trip():
    """Test the get_trip method."""
    global trip_dataset_1

    assert isinstance(trip_dataset_1.get_trip(0), Trip)


def test_get_all():
    """Test the different get_all_... methods."""
    global trip_dataset_1

    assert isinstance(trip_dataset_1.get_all_items(), np.ndarray)
    assert isinstance(trip_dataset_1.get_all_baskets(), np.ndarray)
    assert isinstance(trip_dataset_1.get_all_stores(), np.ndarray)
    assert isinstance(trip_dataset_1.get_all_weeks(), np.ndarray)
    assert isinstance(trip_dataset_1.get_all_prices(), np.ndarray)


def test_n():
    """Test the n_... methods."""
    global trip_dataset_1

    assert isinstance(trip_dataset_1.n_items, int)
    assert isinstance(trip_dataset_1.n_stores, int)
    assert isinstance(trip_dataset_1.n_assortments, int)


def test_get_augmented_data_from_trip_index():
    """Test the get_augmented_data_from_trip_index method."""
    trip_dataset = TripDataset(
        trips=[
            Trip(
                purchases=[1, 2, 0],
                store=0,
                week=0,
                prices=[1, 2, 3, 4],
                # Directly the availability matrix, not an integer
                assortment=[1, 1, 1, 1],
            )
        ],
        available_items=np.array([[1, 1, 1, 1]]),
    )

    assert isinstance(trip_dataset.get_subbaskets_augmented_data_from_trip_index(0), tuple)
    assert isinstance(trip_dataset.get_one_vs_all_augmented_data_from_trip_index(0), tuple)
