"""Test ChoiceDataset and Storage Indexers."""

import pytest

from choice_learn.data.indexer import Indexer


def test_dummy_indexer():
    """Dummy test for the base class Indexer."""
    with pytest.raises(TypeError):
        Indexer(list)
