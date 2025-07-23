"""Unit tests for the SyntheticDataGenerator class."""

import numpy as np
import pytest

from choice_learn.basket_models.DataGen import SyntheticDataGenerator
from choice_learn.basket_models.dataset import Trip


def test_get_assortment():
    """
    Test the get_assortment method.
    This method should return the assortment based on the provided index or array.
    """

    data_gen = SyntheticDataGenerator()

    # Test assortment = None
    # In this case the default assortment binary representation is the first row of the assortment matrix
    n_items = data_gen.assortment_matrix.shape[1]
    assortment = None
    result = data_gen.get_assortment_items(assortment)
    expected = np.array([i for i in range(n_items) if data_gen.assortment_matrix[0, i] == 1])
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"

    # Test assortment as an integer index
    # In this case the assortment binary representation is the row of the assortment matrix corresponding to the index
    row_id = 0
    assortment = row_id
    result = data_gen.get_assortment_items(assortment)
    expected = np.array([i for i in range(n_items) if data_gen.assortment_matrix[row_id, i] == 1])
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"

    # Test assortment as a numpy array
    # In this case the assortment binary representation is the array itself
    assortment = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    result = data_gen.get_assortment_items(assortment)
    expected = np.array([i for i in range(n_items) if assortment[i] == 1])
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"

    # Test assortment is none of the above
    # In this case the assortment binary representation is the first row of the assortment matrix
    assortment = "Hello World"
    expected = np.array([i for i in range(n_items) if data_gen.assortment_matrix[0, i] == 1])
    result = data_gen.get_assortment_items(assortment)
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"


def test_get_available_sets():
    """
    Test the get_available_sets method.
    This method should return the available nests based on the current assortment.
    """

    data_gen = SyntheticDataGenerator()
    n_items = data_gen.assortment_matrix.shape[1]

    assortment = np.array([1] * n_items)
    available_sets = data_gen.get_available_sets(assortment)
    expected_sets = [0, 1, 2, 3]
    assert set(available_sets) == set(expected_sets), (
        f"Expected {expected_sets}, got {available_sets}"
    )

    if n_items == 8:
        assortment = np.array([0, 1, 1, 0, 0, 0, 1, 0])
        available_sets = data_gen.get_available_sets(assortment)
        expected_sets = [0, 2]
        assert set(available_sets) == set(expected_sets), (
            f"Expected {expected_sets}, got {available_sets}"
        )

    if n_items == 8:
        assortment = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        available_sets = data_gen.get_available_sets(assortment)
        expected_sets = []
        assert set(available_sets) == set(expected_sets), (
            f"Expected {expected_sets}, got {available_sets}"
        )


def test_generate_basket():
    """
    Test the generate_basket method.
    This method should generate a basket of items based on the current assortment.
    """

    data_gen = SyntheticDataGenerator()
    n_items = data_gen.assortment_matrix.shape[1]

    # Test with the default complete assortment
    assortment = np.ones(n_items, dtype=int)
    basket = data_gen.generate_basket(assortment)
    assortment_items = set(data_gen.get_assortment_items(assortment))
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
    assortment_items = set(data_gen.get_assortment_items(assortment))
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

    data_gen = SyntheticDataGenerator()
    n_items = data_gen.assortment_matrix.shape[1]

    # Test with the default complete assortment
    assortment = np.ones(n_items, dtype=int)
    available_items = data_gen.get_assortment_items(assortment)
    available_sets = data_gen.get_available_sets(assortment)

    check_all_values_possible = np.zeros(n_items, dtype=int)
    to_finish_cpt = 0
    while np.prod(check_all_values_possible) == 0 and to_finish_cpt < 5000:
        basket = data_gen.generate_basket(assortment)
        first_item = basket[0]
        check_all_values_possible[first_item] = 1
    assert np.prod(check_all_values_possible) == 1, (
        "All items should be possible to be selected as the first item"
    )


def test_generate_trip():
    """
    Test the generate_trip method.
    This method should generate a trip with a basket of items based on the current assortment.
    """

    data_gen = SyntheticDataGenerator()
    n_items = data_gen.assortment_matrix.shape[1]

    # Test with the default complete assortment
    assortment = np.ones(n_items, dtype=int)
    assortment_items = set(data_gen.get_assortment_items(assortment))
    trip = data_gen.generate_trip(assortment)

    assert isinstance(trip, Trip), "result of generate_trip should be a Trip object"
    assert isinstance(trip.purchases, np.ndarray), "Basket in Trip should be a numpy array"
    assert assortment_items.issuperset(set(trip.purchases)), (
        "items in trip purchases should be from the assortment"
    )


def test_generate_trip_dataset():
    """
    Test the generate_trip_dataset method.
    This method should generate a dataset of trips with baskets of items based on the current assortment.
    """

    data_gen = SyntheticDataGenerator()

    # Test with the default parameters (n_baskets = 400, assortment = [[1, 1, 1, 1, 1, 1, 1, 1],])
    dataset = data_gen.generate_trip_dataset()
    assert len(dataset.trips) == data_gen.n_baskets_default, (
        "Dataset should contain the default number of baskets"
    )
    assert all(isinstance(trip, Trip) for trip in dataset.trips), (
        "All trips in the dataset should be Trip objects"
    )
    assert all(isinstance(trip.purchases, np.ndarray) for trip in dataset.trips), (
        "All trip purchases should be numpy arrays"
    )
    assert all(len(trip.purchases) > 0 for trip in dataset.trips), (
        "All trip purchases should not be empty"
    )

    # Test with a custom number of baskets
    n_baskets = 10
    dataset = data_gen.generate_trip_dataset(n_baskets=n_baskets)
    assert len(dataset.trips) == n_baskets, f"Dataset should contain {n_baskets} baskets"

    # Test with a custom assortment matrix
    n_baskets = 10
    assortment_matrix = np.array([[1, 1, 0, 0, 1, 0, 1, 0]])
    available_items = set(data_gen.get_assortment_items(assortment_matrix[0]))
    dataset = data_gen.generate_trip_dataset(
        n_baskets=n_baskets, assortments_matrix=assortment_matrix
    )
    assert all(set(trip.purchases).issubset(available_items) for trip in dataset.trips), (
        "All trip purchases should be from the available items in the assortment matrix"
    )
