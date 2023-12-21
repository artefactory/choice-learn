"""Test the store module."""
from lib.data.store import Store


def test_len_store():
    """Test the __len__ method of Store."""
    store = Store(values=[1, 2, 3, 4], sequence=[0, 1, 2, 3, 0, 1, 2, 3])
    assert len(store) == 8
