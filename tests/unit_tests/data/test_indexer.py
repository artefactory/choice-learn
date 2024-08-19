"""Test ChoiceDataset and Storage Indexers."""

import numpy as np

from choice_learn.data.indexer import Indexer

def test_dummy_indexer():
    """Dummy test for the base class Indexer."""
    indexer = Indexer(list)
    indexer[0]
    assert True
