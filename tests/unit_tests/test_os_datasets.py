"""Unit testing for included Open Source datasets loaders."""
import pandas as pd

from choice_learn.data import ChoiceDataset
from choice_learn.datasets import load_swissmetro


def test_swissmetro_loader():
    """Test loading the Swissmetro dataset."""
    swissmetro = load_swissmetro(as_frame=True)
    assert isinstance(swissmetro, pd.DataFrame)
    assert swissmetro.shape == (10728, 28)

    swissmetro = load_swissmetro()
    assert isinstance(swissmetro, ChoiceDataset)
    swissmetro = load_swissmetro(add_items_one_hot=True)
    assert isinstance(swissmetro, ChoiceDataset)
