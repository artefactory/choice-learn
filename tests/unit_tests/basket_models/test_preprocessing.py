"""Unit tests for basket preprocessing."""

import pandas as pd

from choice_learn.basket_models.basket_dataset.preprocessing import map_indexes


def test_map_indexes() -> None:
    """Test the map_indexes method."""
    data = pd.DataFrame(
        {
            "xxx": ["a", "b", "c", "a", "b", "c"],
            "yyy": [1, 2, 3, 1, 2, 3],
            "zzz": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        }
    )

    column_name = "xxx"
    index_start = 2

    assert map_indexes(data, column_name, index_start) == {
        "a": 2,
        "b": 3,
        "c": 4,
    }
