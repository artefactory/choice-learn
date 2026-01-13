"""Unit testing for included Open Source basket datasets loaders."""

import pandas as pd

from choice_learn.basket_models.data import TripDataset
from choice_learn.basket_models.datasets.bakery import load_bakery


def test_bakery_loader():
    """Test loading the Bakery dataset."""
    bakery = load_bakery(as_frame=True)
    assert isinstance(bakery, pd.DataFrame)
    assert len(bakery) == 75000

    bakery = load_bakery()
    assert isinstance(bakery, TripDataset)
    assert len(bakery) == 75000

    bakery = load_bakery(load_5_25_version=True)
    assert isinstance(bakery, TripDataset)
    assert len(bakery) == 67488
