"""Unit tests for the SyntheticDataGenerator class.

This module tests the methods of the SyntheticDataGenerator class
"""

import numpy as np

from choice_learn.basket_models.data import Trip
from choice_learn.basket_models.data.synthetic_dataset import SyntheticDataGenerator

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


def test_init_parameter_validation():
    """Test initialization with various parameter combinations."""
    # Test valid initialization
    generator = SyntheticDataGenerator(
        proba_complementary_items=0.8,
        proba_neutral_items=0.2,
        noise_proba=0.1,
        items_nest={0: [0, 1], 1: [2, 3]},
        nests_interactions=[["", "compl"], ["compl", ""]],
    )

    assert generator.proba_complementary_items == 0.8
    assert generator.proba_neutral_items == 0.2
    assert generator.noise_proba == 0.1
    assert generator.items_nest == {0: [0, 1], 1: [2, 3]}
    assert generator.nests_interactions == [["", "compl"], ["compl", ""]]


def test_get_available_sets_edge_cases():
    """Test get_available_sets with edge cases."""
    generator = SyntheticDataGenerator(
        items_nest={0: [0, 1, 2], 1: [3, 4], 2: [5], 3: [6, 7, 8]},
        nests_interactions=[["", "compl", "neutral", "neutral"] for _ in range(4)],
    )

    # Test with partial overlap
    assortment_items = {0, 3, 5, 9}  # 9 is not in any nest
    available_sets = generator.get_available_sets(assortment_items)
    expected_sets = [0, 1, 2]  # nests that have intersection
    assert set(available_sets) == set(expected_sets)

    # Test with single item per nest
    assortment_items = {1, 4, 6}
    available_sets = generator.get_available_sets(assortment_items)
    expected_sets = [0, 1, 3]
    assert set(available_sets) == set(expected_sets)

    # Test with items not in any nest
    assortment_items = {10, 11, 12}
    available_sets = generator.get_available_sets(assortment_items)
    assert len(available_sets) == 0


def test_select_first_item_deterministic():
    """Test select_first_item with controlled randomness."""
    np.random.seed(123)  # Set seed for reproducibility

    generator = SyntheticDataGenerator(
        items_nest={0: [0, 1, 2], 1: [3, 4, 5]}, nests_interactions=[["", "compl"], ["compl", ""]]
    )

    available_sets = np.array([0, 1])
    available_items = np.array([0, 1, 3, 4])

    item, nest = generator.select_first_item(available_sets, available_items)

    assert item in available_items
    assert nest in available_sets
    assert item in generator.items_nest[nest]


def test_select_first_item_single_option():
    """Test select_first_item when only one option is available."""
    generator = SyntheticDataGenerator(
        items_nest={0: [0, 1, 2], 1: [3, 4, 5]}, nests_interactions=[["", "compl"], ["compl", ""]]
    )

    # Only one nest available
    available_sets = np.array([0])
    available_items = np.array([1])  # Only one item from nest 0

    item, nest = generator.select_first_item(available_sets, available_items)

    assert item == 1
    assert nest == 0


def test_complete_basket_all_relations():
    """Test complete_basket with different relation types."""
    np.random.seed(456)

    generator = SyntheticDataGenerator(
        proba_complementary_items=1.0,
        proba_neutral_items=1.0,
        noise_proba=0.0,
        items_nest={0: [0, 1], 1: [2, 3], 2: [4], 3: [5, 6]},
        nests_interactions=[
            ["", "compl", "neutral", ""],
            ["compl", "", "", "neutral"],
            ["neutral", "", "", ""],
            ["", "neutral", "", ""],
        ],
    )

    available_items = [0, 1, 2, 3, 4, 5, 6]
    basket = generator.complete_basket(first_item=0, first_nest=0, available_items=available_items)

    assert 0 in basket
    assert len(basket) >= 1
    assert all(item in available_items for item in basket)


def test_complete_basket_unavailable_items():
    """Test complete_basket when complementary/neutral items are unavailable."""
    generator = SyntheticDataGenerator(
        proba_complementary_items=1.0,
        proba_neutral_items=1.0,
        noise_proba=0.0,
        items_nest={0: [0, 1], 1: [2, 3], 2: [4]},
        nests_interactions=[["", "compl", "neutral"], ["compl", "", ""], ["neutral", "", ""]],
    )

    # Items from complementary/neutral nests are not available
    available_items = [0, 1]
    basket = generator.complete_basket(first_item=0, first_nest=0, available_items=available_items)

    assert basket == [0]  # Should only contain the first item


def test_add_noise_functionality():
    """Test add_noise method functionality."""
    np.random.seed(789)

    generator = SyntheticDataGenerator(
        noise_proba=1.0, items_nest={0: [0, 1]}, nests_interactions=[[""]]
    )

    basket = [0, 1]
    available_items = [0, 1, 2, 3, 4, 5, 6, 7]

    noisy_basket = generator.add_noise(basket, available_items)

    assert len(noisy_basket) >= len(basket)
    assert all(item in available_items for item in noisy_basket)
    assert len(set(noisy_basket)) == len(noisy_basket)

    if len(noisy_basket) > len(basket):
        added_items = set(noisy_basket) - set(basket)
        assert len(added_items) == 1
        added_item = list(added_items)[0]
        assert added_item in available_items
        assert added_item not in basket


def test_generate_trip_properties():
    """Test generate_trip creates proper Trip objects."""
    generator = SyntheticDataGenerator(items_nest={0: [0, 1, 2]}, nests_interactions=[[""]])

    assortment = np.array([1, 1, 1, 0, 0])
    trip = generator.generate_trip(assortment)

    assert isinstance(trip, Trip)
    assert isinstance(trip.purchases, np.ndarray)
    assert np.array_equal(trip.assortment, assortment)
    assert trip.prices.shape == (1, len(assortment))
    assert np.all(trip.prices == 1.0)  # All prices should be 1.0


def test_generate_trip_with_length():
    """Test generate_trip with specific length."""
    generator = SyntheticDataGenerator(items_nest={0: [0, 1, 2, 3]}, nests_interactions=[[""]])

    assortment = np.ones(4, dtype=int)
    target_length = 2

    trip = generator.generate_trip(assortment, len_basket=target_length)

    assert len(trip.purchases) == target_length


def test_generate_trip_dataset_multiple_assortments():
    """Test generate_trip_dataset with multiple assortments."""
    np.random.seed(111)

    generator = SyntheticDataGenerator(
        items_nest={0: [0, 1], 1: [2, 3]}, nests_interactions=[["", "compl"], ["compl", ""]]
    )

    n_baskets = 20
    assortments_matrix = np.array([[1, 1, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]])

    dataset = generator.generate_trip_dataset(n_baskets, assortments_matrix)

    assert len(dataset.trips) == n_baskets

    # Check that generated trips respect their assortments
    for trip in dataset.trips:
        available_items = set(np.where(trip.assortment == 1)[0])
        assert set(trip.purchases).issubset(available_items)


def test_generate_trip_dataset_with_length():
    """Test generate_trip_dataset with specific basket length."""
    generator = SyntheticDataGenerator(items_nest={0: [0, 1, 2, 3, 4]}, nests_interactions=[[""]])

    n_baskets = 3
    assortments_matrix = np.array([[1, 1, 1, 1, 1]])
    target_length = 2

    dataset = generator.generate_trip_dataset(
        n_baskets, assortments_matrix, len_basket=target_length
    )

    assert all(len(trip.purchases) == target_length for trip in dataset.trips)


def test_nests_interactions_symmetry():
    """Test that asymmetric nest interactions work correctly."""
    generator = SyntheticDataGenerator(
        proba_complementary_items=1.0,
        proba_neutral_items=0.0,
        noise_proba=0.0,
        items_nest={0: [0], 1: [1], 2: [2]},
        nests_interactions=[["", "compl", ""], ["", "", "compl"], ["compl", "", ""]],
    )

    assortment = np.ones(3, dtype=int)

    # Test starting from nest 0
    basket_from_0 = generator.complete_basket(0, 0, assortment)

    # Test starting from nest 1
    basket_from_1 = generator.complete_basket(1, 1, assortment)

    # Test starting from nest 2
    basket_from_2 = generator.complete_basket(2, 2, assortment)

    assert len(basket_from_0) >= 1
    assert len(basket_from_1) >= 1
    assert len(basket_from_2) >= 1


def test_empty_nests():
    """Test behavior with empty item nests."""
    generator = SyntheticDataGenerator(
        items_nest={0: [], 1: [0, 1], 2: []}, nests_interactions=[["", "", ""] for _ in range(3)]
    )

    assortment_items = {0, 1}
    available_sets = generator.get_available_sets(assortment_items)

    # Only nest 1 should be available (has items that intersect)
    assert list(available_sets) == [1]


def test_large_dataset_generation():
    """Test generation of large datasets for performance."""
    generator = SyntheticDataGenerator(
        items_nest={0: [0, 1, 2], 1: [3, 4, 5]}, nests_interactions=[["", "compl"], ["compl", ""]]
    )

    n_baskets = 100
    assortments_matrix = np.array([[1, 1, 1, 1, 1, 1]])

    dataset = generator.generate_trip_dataset(n_baskets, assortments_matrix)

    assert len(dataset.trips) == n_baskets
    assert all(isinstance(trip, Trip) for trip in dataset.trips)
    assert all(len(trip.purchases) > 0 for trip in dataset.trips)
