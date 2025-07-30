"""Unit tests for the SyntheticDataGenerator class.

This module tests the methods of the SyntheticDataGenerator class
"""

import numpy as np

from choice_learn.basket_models.dataset import Trip
from choice_learn.basket_models.synthetic_dataset import SyntheticDataGenerator

data_gen = SyntheticDataGenerator(
    proba_complementary_items=0.7,
    proba_neutral_items=0.3,
    noise_proba=0.15,
    items_nest={0: [0, 1, 2], 1: [3, 4, 5], 2: [6], 3: [7]},
    nests_interactions=[
        ["", "compl", "neutral", "neutral"],
        ["compl", "", "neutral", "neutral"],
        ["neutral", "neutral", "", "neutral"],
        ["neutral", "neutral", "neutral", ""],
    ],
)

n_items = 8
assortments_matrix = np.array([[1, 0, 1, 1, 0, 1, 1, 0]])


def test_get_available_sets():
    """
    Test the get_available_sets method.

    This method should return the available nests based on the current assortment.
    """
    assortment = np.array([1] * n_items)
    assortment_items = set(np.where(assortment == 1)[0])
    available_sets = data_gen.get_available_sets(assortment_items)
    expected_sets = [0, 1, 2, 3]
    assert set(available_sets) == set(expected_sets), (
        f"Expected {expected_sets}, got {available_sets}"
    )

    if n_items == 8:
        assortment = np.array([0, 1, 1, 0, 0, 0, 1, 0])
        assortment_items = set(np.where(assortment == 1)[0])
        available_sets = data_gen.get_available_sets(assortment_items)
        expected_sets = [0, 2]
        assert set(available_sets) == set(expected_sets), (
            f"Expected {expected_sets}, got {available_sets}"
        )

    if n_items == 8:
        assortment = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        assortment_items = set(np.where(assortment == 1)[0])
        available_sets = data_gen.get_available_sets(assortment_items)
        expected_sets = []
        assert set(available_sets) == set(expected_sets), (
            f"Expected {expected_sets}, got {available_sets}"
        )


def test_generate_basket():
    """
    Test the generate_basket method.

    This method should generate a basket of items based on the current assortment.
    """
    # Test with the default complete assortment
    assortment = np.ones(n_items, dtype=int)
    basket = data_gen.generate_basket(assortment)
    assortment_items = set(np.where(assortment == 1)[0])
    unique_items = set(basket)
    assert isinstance(basket, np.ndarray), "Basket should be a numpy array"
    assert len(basket) > 0, "Basket should not be empty"
    assert assortment_items.issuperset(set(basket)), "items in basket should be from the assortment"
    assert len(unique_items) == len(basket), "Basket should not contain duplicate items"

    # Test with an empty assortment
    assortment = np.array([0] * n_items)
    basket = data_gen.generate_basket(assortment)
    assert len(basket) == 0, "Basket should be empty for an empty assortment"

    # Test with a random assortment
    assortment = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    assortment_items = set(np.where(assortment == 1)[0])
    basket = data_gen.generate_basket(assortment)
    unique_items = set(basket)
    assert len(basket) >= 0, "Basket should not be negative length"
    assert assortment_items.issuperset(set(basket)), "items in basket should be from the assortment"
    assert len(unique_items) == len(basket), "Basket should not contain duplicate items"

    # Test basket length
    len_basket = 3
    assortment = np.array([1] * n_items)
    basket = data_gen.generate_basket(assortment, len_basket=len_basket)
    unique_items = set(basket)
    assert len(basket) == len_basket, f"Basket length should be {len_basket}, got {len(basket)}"
    assert len(unique_items) == len_basket, "Basket should not contain duplicate items"


def test_select_first_item():
    """
    Test generate_basket's select_first_item method.

    This method should select the first item and its nest randomly from the available sets.
    """
    # Test with the default complete assortment
    assortment = np.ones(n_items, dtype=int)

    check_all_values_possible = np.zeros(n_items, dtype=int)
    to_finish_cpt = 0
    while np.prod(check_all_values_possible) == 0 and to_finish_cpt < 5000:
        basket = data_gen.generate_basket(assortment)
        first_item = basket[0]
        check_all_values_possible[first_item] = 1
    assert np.prod(check_all_values_possible) == 1, "All items can be first item"


def test_generate_trip():
    """
    Test the generate_trip method.

    This method should generate a trip with a basket of items based on the current assortment.
    """
    # Test with the default complete assortment
    assortment = np.ones(n_items, dtype=int)
    assortment_items = set(np.where(assortment == 1)[0])
    trip = data_gen.generate_trip(assortment)

    assert isinstance(trip, Trip), "result of generate_trip should be a Trip object"
    assert isinstance(trip.purchases, np.ndarray), "Basket in Trip should be a numpy array"
    assert assortment_items.issuperset(set(trip.purchases)), "items in should be from assortment"


def test_generate_trip_dataset():
    """
    Test the generate_trip_dataset method.

    This method should generate a dataset of trips
    with baskets of items based on the current assortment.
    """
    n_baskets = 10

    # Test with the default parameters (n_baskets = 400, assortment = [[1, 1, 1, 1, 1, 1, 1, 1],])
    dataset = data_gen.generate_trip_dataset(n_baskets, assortments_matrix)
    assert len(dataset.trips) == n_baskets, "Should contain exact number of basket"
    assert all(isinstance(trip, Trip) for trip in dataset.trips), (
        "All trips in the dataset should be Trip objects"
    )
    assert all(isinstance(trip.purchases, np.ndarray) for trip in dataset.trips), (
        "All trip purchases should be numpy arrays"
    )
    assert all(len(trip.purchases) > 0 for trip in dataset.trips), (
        "All Trip purchases should not be empty"
    )

    # Test with a custom number of baskets
    n_baskets = 10
    dataset = data_gen.generate_trip_dataset(
        n_baskets=n_baskets, assortments_matrix=assortments_matrix
    )
    assert len(dataset.trips) == n_baskets, f"Dataset should contain {n_baskets} baskets"

    # Test with a custom assortment matrix
    n_baskets = 10
    assortment_matrix = np.array([[1, 1, 0, 0, 1, 0, 1, 0]])
    available_items = set(np.where(assortment_matrix[0] == 1)[0])
    dataset = data_gen.generate_trip_dataset(
        n_baskets=n_baskets, assortments_matrix=assortment_matrix
    )
    assert all(set(trip.purchases).issubset(available_items) for trip in dataset.trips), (
        "All trip purchases not in the assortment matrix"
    )
