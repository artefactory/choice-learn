"""Tests for the store.py module."""

import numpy as np

from choice_learn.data.indexer import OneHotStoreIndexer, StoreIndexer
from choice_learn.data.store import FeaturesStore, OneHotStore, Store


class TestStore:
    """Tests for the Store class."""

    def test_init_with_indexes(self):
        """Test initialization with provided indexes."""
        indexes = [0, 1, 2]
        values = [10, 20, 30]
        sequence = [0, 1, 2, 0, 1]
        store = Store(indexes=indexes, values=values, sequence=sequence)

        assert store.store == {0: 10, 1: 20, 2: 30}
        np.testing.assert_array_equal(store.sequence, np.array(sequence))
        assert store.shape == (5, 1)
        assert isinstance(store.indexer, StoreIndexer)

    def test_init_without_indexes(self):
        """Test initialization without indexes (auto-generated)."""
        values = [10, 20, 30]
        sequence = [0, 1, 2, 0, 1]
        store = Store(values=values, sequence=sequence)

        assert store.store == {0: 10, 1: 20, 2: 30}
        np.testing.assert_array_equal(store.sequence, np.array(sequence))
        assert store.shape == (5, 1)

    def test_init_with_multidimensional_values(self):
        """Test initialization with multi-dimensional values."""
        values = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
        sequence = [0, 1, 2, 0, 1]
        store = Store(values=values, sequence=sequence)

        assert len(store.store) == 3
        np.testing.assert_array_equal(store.store[0], np.array([1, 2]))
        np.testing.assert_array_equal(store.store[1], np.array([3, 4]))
        np.testing.assert_array_equal(store.store[2], np.array([5, 6]))
        assert store.shape == (5, 2)

    def test_init_with_name(self):
        """Test initialization with a name parameter."""
        values = [10, 20, 30]
        sequence = [0, 1, 2, 0, 1]
        store = Store(values=values, sequence=sequence, name="test_store")

        assert store.name == "test_store"

    def test_init_without_sequence_or_values(self):
        """Test initialization without sequence or values."""
        indexes = [0, 1, 2]
        values = [10, 20, 30]
        store = Store(indexes=indexes, values=values)

        assert store.store == {0: 10, 1: 20, 2: 30}

        # Test with just sequence
        sequence = [0, 1, 2, 0, 1]
        store = Store(sequence=sequence)
        assert hasattr(store, "sequence")
        np.testing.assert_array_equal(store.sequence, np.array(sequence))

    def test_get_store_element_single_index(self):
        """Test _get_store_element with a single index."""
        indexes = [0, 1, 2]
        values = [10, 20, 30]
        sequence = [0, 1, 2, 0, 1]
        store = Store(indexes=indexes, values=values, sequence=sequence)

        assert store._get_store_element(0) == 10
        assert store._get_store_element(1) == 20
        assert store._get_store_element(2) == 30

    def test_get_store_element_list_index(self):
        """Test _get_store_element with a list of indexes."""
        indexes = [0, 1, 2]
        values = [10, 20, 30]
        sequence = [0, 1, 2, 0, 1]
        store = Store(indexes=indexes, values=values, sequence=sequence)

        assert store._get_store_element([0, 1]) == [10, 20]
        assert store._get_store_element([1, 2]) == [20, 30]
        assert store._get_store_element([0, 1, 2]) == [10, 20, 30]

    def test_len(self):
        """Test the __len__ method."""
        indexes = [0, 1, 2]
        values = [10, 20, 30]
        sequence = [0, 1, 2, 0, 1]
        store = Store(indexes=indexes, values=values, sequence=sequence)

        assert len(store) == 5

    def test_batch_property(self):
        """Test the batch property."""
        indexes = [0, 1, 2]
        values = [10, 20, 30]
        sequence = [0, 1, 2, 0, 1]
        store = Store(indexes=indexes, values=values, sequence=sequence)

        assert store.batch is store.indexer


class TestFeaturesStore:
    """Tests for the FeaturesStore class."""

    def test_from_dict(self):
        """Test the from_dict class method."""
        values_dict = {0: np.array([1, 2]), 1: np.array([3, 4]), 2: np.array([5, 6])}
        sequence = [0, 1, 2, 0, 1]

        features_store = FeaturesStore.from_dict(values_dict, sequence)

        assert isinstance(features_store, FeaturesStore)
        assert len(features_store.store) == 3
        np.testing.assert_array_equal(features_store.store[0], np.array([1, 2]))
        np.testing.assert_array_equal(features_store.store[1], np.array([3, 4]))
        np.testing.assert_array_equal(features_store.store[2], np.array([5, 6]))
        np.testing.assert_array_equal(features_store.sequence, np.array(sequence))
        assert features_store.shape == (5, 2)

    def test_from_list(self):
        """Test the from_list class method."""
        values_list = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
        sequence = [0, 1, 2, 0, 1]

        features_store = FeaturesStore.from_list(values_list, sequence)

        assert isinstance(features_store, FeaturesStore)
        assert len(features_store.store) == 3
        np.testing.assert_array_equal(features_store.store[0], np.array([1, 2]))
        np.testing.assert_array_equal(features_store.store[1], np.array([3, 4]))
        np.testing.assert_array_equal(features_store.store[2], np.array([5, 6]))
        np.testing.assert_array_equal(features_store.sequence, np.array(sequence))
        assert features_store.shape == (5, 2)

    def test_getitem_single_index(self):
        """Test the __getitem__ method with a single index."""
        values_list = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
        sequence = [0, 1, 2, 0, 1]

        features_store = FeaturesStore.from_list(values_list, sequence)
        subset = features_store[0]

        assert isinstance(subset, FeaturesStore)
        assert len(subset.store) == 1
        np.testing.assert_array_equal(subset.store[0], np.array([1, 2]))
        np.testing.assert_array_equal(subset.sequence, np.array([0]))

    def test_getitem_slice(self):
        """Test the __getitem__ method with a slice."""
        values_list = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
        sequence = [0, 1, 2, 0, 1]

        features_store = FeaturesStore.from_list(values_list, sequence)
        subset = features_store[1:4]

        assert isinstance(subset, FeaturesStore)
        assert len(subset.store) <= 3  # Could be less if some keys aren't in the slice
        np.testing.assert_array_equal(subset.sequence, np.array([1, 2, 0]))

    def test_astype(self):
        """Test the astype method."""
        values_list = [
            np.array([1, 2], dtype=np.int32),
            np.array([3, 4], dtype=np.int32),
            np.array([5, 6], dtype=np.int32),
        ]
        sequence = [0, 1, 2, 0, 1]

        features_store = FeaturesStore.from_list(values_list, sequence)
        features_store.astype(np.float64)

        assert features_store.store[0].dtype == np.float64
        assert features_store.store[1].dtype == np.float64
        assert features_store.store[2].dtype == np.float64


class TestOneHotStore:
    """Tests for the OneHotStore class."""

    def test_init_with_values(self):
        """Test initialization with provided values."""
        indexes = [10, 20, 30]
        values = np.array([0, 1, 2])
        sequence = [10, 20, 30, 10, 20]

        onehot_store = OneHotStore(indexes=indexes, values=values, sequence=sequence)

        assert len(onehot_store.store) == 3
        assert onehot_store.store[10] == 0
        assert onehot_store.store[20] == 1
        assert onehot_store.store[30] == 2
        np.testing.assert_array_equal(onehot_store.sequence, np.array(sequence))
        assert onehot_store.shape == (5, 3)  # (len(sequence), max(values) + 1)
        assert onehot_store.dtype == np.float32
        assert isinstance(onehot_store.indexer, OneHotStoreIndexer)

    def test_init_with_none_values(self):
        """Test initialization with None values (created from sequence)."""
        sequence = [10, 20, 30, 10, 20]

        onehot_store = OneHotStore(values=None, sequence=sequence)

        assert len(onehot_store.store) == 3
        # Values should be assigned in order they appear in sequence
        unique_indexes = np.unique(sequence)
        for i, idx in enumerate(unique_indexes):
            assert onehot_store.store[idx] == i

    def test_init_with_dtype(self):
        """Test initialization with a custom dtype."""
        sequence = [10, 20, 30, 10, 20]

        onehot_store = OneHotStore(sequence=sequence, dtype=np.int64)

        assert onehot_store.dtype == np.int64

    def test_from_sequence(self):
        """Test the from_sequence class method."""
        sequence = [10, 20, 30, 10, 20]

        onehot_store = OneHotStore.from_sequence(sequence)

        assert len(onehot_store.store) == 3
        np.testing.assert_array_equal(onehot_store.sequence, np.array(sequence))

        # Unique values from sequence should be indexes
        assert 10 in onehot_store.store
        assert 20 in onehot_store.store
        assert 30 in onehot_store.store

        # Values should be assigned in order they appear in sequence
        unique_indexes = np.unique(sequence)
        for i, idx in enumerate(unique_indexes):
            assert onehot_store.store[idx] == i

    def test_getitem(self):
        """Test the __getitem__ method."""
        sequence = [10, 20, 30, 10, 20]

        onehot_store = OneHotStore.from_sequence(sequence)
        subset = onehot_store[1:3]

        assert isinstance(subset, OneHotStore)
        np.testing.assert_array_equal(subset.sequence, np.array([20, 30]))
        assert len(subset.store) == 2
        assert 20 in subset.store
        assert 30 in subset.store

    def test_astype(self):
        """Test the astype method."""
        sequence = [10, 20, 30, 10, 20]

        onehot_store = OneHotStore.from_sequence(sequence)
        assert onehot_store.dtype == np.float32

        onehot_store.astype(np.int64)
        assert onehot_store.dtype == np.int64
