"""Unit testing for Expedia loader."""

import numpy as np
import pandas as pd
import pytest

from choice_learn.data import ChoiceDataset
from choice_learn.datasets import load_expedia

def test_raise_filenotfound():
    """Test that error raised if no file exist"""
    with pytest.raises(FileNotFoundError):
        load_expedia()