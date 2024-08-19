"""Test Indexer + Storage."""

import numpy as np
import pytest

from choice_learn.data import FeaturesStorage

indexed_array = np.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0], [3.0, 4.0, 5.0], [5.0, 4.0, 3.0]])
indexed_dict = {0: [0.0, 1.0, 2.0], 1: [2.0, 1.0, 0.0], 2: [3.0, 4.0, 5.0], 3: [5.0, 4.0, 3.0]}


def test_storage_iloc():
    """Test __getitem__ with various inputs."""
    storage = FeaturesStorage(values=indexed_dict)

    assert (np.array(storage.batch[0]) == np.array([0.0, 1.0, 2.0])).all()
    assert (np.array(storage.batch[[1, 2]]) == np.array([[2.0, 1.0, 0.0], [3.0, 4.0, 5.0]])).all()
    assert (
        np.array(storage.batch[[[3, 1], [0, 2]]])
        == np.array([[[5.0, 4.0, 3.0], [2.0, 1.0, 0.0]], [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]])
    ).all()

    with pytest.raises(ValueError):
        storage.batch[0:4]
    with pytest.raises(KeyError):
        storage.batch[4]
