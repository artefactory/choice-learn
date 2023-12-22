"""Test the store module."""
from choice_learn.data.store import Store


def test_len_store():
    """Test the __len__ method of Store."""
    store = Store(values=[1, 2, 3, 4], sequence=[0, 1, 2, 3, 0, 1, 2, 3])
    assert len(store) == 8


def test_get_store_element():
    """Test the _get_store_element method of Store."""
    store = Store(values=[1, 2, 3, 4], sequence=[0, 1, 2, 3, 0, 1, 2, 3])
    assert store._get_store_element(0) == 1
    assert store._get_store_element([0, 1, 2]) == [1, 2, 3]


def test_store_batch():
    """Test the batch method of Store."""
    store = Store(values=[1, 2, 3, 4], sequence=[0, 1, 2, 3, 0, 1, 2, 3])
    assert store.batch[1] == 2
    assert store.batch[2:4] == [3, 4]
    assert store.batch[[2, 3, 6, 7]] == [3, 4, 3, 4]
