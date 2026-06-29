"""Unit testing for class ChoiceDataset."""

import numpy as np
import pandas as pd
import pytest

from choice_learn.data.choice_dataset import ChoiceDataset
from choice_learn.data.storage import FeaturesStorage

# We have two customers whose features are
# - Budget
# - Age
# Customer 1 bought item 1 at session 1 and item 2 at session 3
# Customer 2 bought item 3 at session 2

choices = [0, 2, 1]
available_items_by_choice = [
    [1, 1, 1],  # All items available at session 1
    [1, 1, 1],  # All items available at session 2
    [0, 1, 1],  # Item 1 not available at session 3
]

shared_features_by_choice = [
    [100, 20],  # session 1, customer 1 [budget, age]
    [200, 40],  # session 2, customer 2 [budget, age]
    [80, 20],  # session 3, customer 1 [budget, age]
]

items_features_by_choice = [
    [
        [100, 0],  # Session 1, Item 1 [price, promotion]
        [140, 0],  # Session 1, Item 2 [price, promotion]
        [200, 0],  # Session 1, Item 2 [price, promotion]
    ],
    [
        [100, 0],  # Session 2 Item 1 [price, promotion]
        [120, 1],  # Session 2, Item 2 [price, promotion]
        [200, 0],  # Session 2, Item 2 [price, promotion]
    ],
    [
        [0, 0],  # Session 3, Item 1 [price, promotion],
        # values do not really matter, but needs to exist for shapes sake
        [120, 1],  # Session 3, Item 2 [price, promotion]
        [180, 1],  # Session 3, Item 2 [price, promotion]
    ],
]


def test_raise_error():
    """Test various errors in instantiation."""
    with pytest.raises(ValueError):
        # Choices is None
        ChoiceDataset(
            shared_features_by_choice=shared_features_by_choice,
            items_features_by_choice=items_features_by_choice,
            available_items_by_choice=available_items_by_choice,
            choices=None,
        )
    with pytest.raises(ValueError):
        # wrong items features names shape
        ChoiceDataset(
            shared_features_by_choice=shared_features_by_choice,
            items_features_by_choice=(items_features_by_choice, items_features_by_choice),
            items_features_by_choice_names=(["ab", "bc"], ["cd"]),
            available_items_by_choice=available_items_by_choice,
            choices=choices,
        )
    with pytest.raises(ValueError):
        # wrong shared features names shape
        ChoiceDataset(
            shared_features_by_choice=(shared_features_by_choice, shared_features_by_choice),
            shared_features_by_choice_names=(["a", "b"], ["c", "d", "e"]),
            items_features_by_choice=items_features_by_choice,
            available_items_by_choice=available_items_by_choice,
            choices=choices,
        )

    with pytest.raises(ValueError):
        # wrong nb of items in items features shape
        ChoiceDataset(
            shared_features_by_choice=shared_features_by_choice,
            items_features_by_choice=(
                items_features_by_choice,
                np.array(
                    [
                        [
                            [100, 0, 2],
                            [140, 0, 2],
                            [200, 0, 2],
                            [200, 0, 2],
                        ],
                        [
                            [100, 0, 2],
                            [120, 1, 2],
                            [200, 0, 2],
                            [200, 0, 2],
                        ],
                        [
                            [0, 0, 2],
                            [120, 1, 2],
                            [180, 1, 2],
                            [180, 1, 2],
                        ],
                    ]
                ),
            ),
            available_items_by_choice=available_items_by_choice,
            choices=[0, 2, 1],
        )

    # wrong nb of shared features names
    with pytest.raises(ValueError):
        ChoiceDataset(
            shared_features_by_choice=shared_features_by_choice,
            items_features_by_choice=items_features_by_choice,
            available_items_by_choice=np.array([[1, 1], [0, 1], [1, 0]]),
            choices=[0, 2, 1],
        )

    # wrong nb of shared features names
    with pytest.raises(ValueError):
        ChoiceDataset(
            shared_features_by_choice=shared_features_by_choice,
            shared_features_by_choice_names=["a", "b", "c"],
            items_features_by_choice=items_features_by_choice,
            available_items_by_choice=available_items_by_choice,
            choices=[0, 2, 1],
        )
    with pytest.raises(ValueError):
        ChoiceDataset(
            shared_features_by_choice=(shared_features_by_choice, shared_features_by_choice),
            shared_features_by_choice_names=(["a", "b"], ["a", "b", "c"]),
            items_features_by_choice=items_features_by_choice,
            available_items_by_choice=available_items_by_choice,
            choices=[0, 2, 1],
        )
    with pytest.raises(ValueError):
        ChoiceDataset(
            shared_features_by_choice=shared_features_by_choice,
            items_features_by_choice=items_features_by_choice,
            items_features_by_choice_names=["ab", "bc", "cd"],
            available_items_by_choice=available_items_by_choice,
            choices=[0, 2, 1],
        )
    with pytest.raises(ValueError):
        ChoiceDataset(
            shared_features_by_choice=shared_features_by_choice,
            items_features_by_choice=(items_features_by_choice, items_features_by_choice),
            items_features_by_choice_names=(["ab", "bc"], ["ab", "bc", "cd"]),
            available_items_by_choice=available_items_by_choice,
            choices=[0, 2, 1],
        )
    with pytest.raises(ValueError):
        ChoiceDataset(
            shared_features_by_choice=shared_features_by_choice,
            items_features_by_choice=(items_features_by_choice, [0, 1]),
            items_features_by_choice_names=(["ab", "bc"], ["abc"]),
            available_items_by_choice=available_items_by_choice,
            choices=[0, 2, 1],
        )
    with pytest.raises(ValueError):
        ChoiceDataset(
            shared_features_by_choice=shared_features_by_choice,
            items_features_by_choice=(items_features_by_choice, [0, 1, 0]),
            items_features_by_choice_names=(["ab", "bc"], ["abc", "def"]),
            available_items_by_choice=available_items_by_choice,
            choices=[0, 2, 1],
            features_by_ids=[
                FeaturesStorage(
                    ids=[0, 1],
                    values=[[[0.0, 1.0], [1.0, 0], [1.0, 0]], [[2.0, 3.0], [3.0, 2.0], [3.0, 2.0]]],
                    name="abc",
                )
            ],
        )
    # shared features FeaturesStorage wrong key
    with pytest.raises(KeyError):
        ChoiceDataset(
            shared_features_by_choice=(shared_features_by_choice, [[0], [1], [2]]),
            items_features_by_choice=items_features_by_choice,
            shared_features_by_choice_names=(["ab", "bc"], ["abc"]),
            available_items_by_choice=available_items_by_choice,
            choices=[0, 2, 1],
            features_by_ids=[
                FeaturesStorage(
                    ids=[0, 1],
                    values=[[0.0, 1.0], [1.0, 0]],
                    name="abc",
                )
            ],
        )
    # Test instantiation with 1D FS for items_features
    dataset = ChoiceDataset(
        shared_features_by_choice=shared_features_by_choice,
        items_features_by_choice=(items_features_by_choice, [0, 1, 0]),
        items_features_by_choice_names=(["ab", "bc"], ["abc"]),
        available_items_by_choice=available_items_by_choice,
        choices=[0, 2, 1],
        features_by_ids=[
            FeaturesStorage(
                ids=[0, 1],
                values=[[[0.0, 1.0], [1.0, 0], [1.0, 0]], [[2.0, 3.0], [3.0, 2.0], [3.0, 2.0]]],
                name="abc",
            )
        ],
    )
    str(dataset)
    assert True

    # Test DF for items_features without choice_id
    with pytest.raises(ValueError):
        ChoiceDataset(
            shared_features_by_choice=shared_features_by_choice,
            items_features_by_choice=pd.DataFrame(
                {
                    "price": [100, 140, 200, 100, 120, 200, 0, 120, 180],
                    "promotion": [0, 0, 0, 0, 1, 0, 0, 1, 1],
                    "item_id": [0, 1, 2, 0, 1, 2, 0, 1, 2],
                }
            ),
            available_items_by_choice=available_items_by_choice,
            choices=[0, 2, 1],
        )

    with pytest.raises(ValueError):
        ChoiceDataset(
            shared_features_by_choice=shared_features_by_choice,
            items_features_by_choice=(items_features_by_choice, [2, 1, 0]),
            items_features_by_choice_names=(["ab", "bc"], ["abc"]),
            available_items_by_choice=available_items_by_choice,
            choices=[0, 2, 1],
            features_by_ids=[
                FeaturesStorage(
                    ids=[0, 1],
                    values=[[[0.0, 1.0], [1.0, 0], [1.0, 0]], [[2.0, 3.0], [3.0, 2.0], [3.0, 2.0]]],
                    name="abc",
                )
            ],
        )
    # Features Storage and no features names
    with pytest.raises(ValueError):
        ChoiceDataset(
            shared_features_by_choice=shared_features_by_choice,
            items_features_by_choice=(items_features_by_choice, [0, 1, 0]),
            available_items_by_choice=available_items_by_choice,
            choices=[0, 2, 1],
            features_by_ids=[
                FeaturesStorage(
                    ids=[0, 1],
                    values=[[[0.0, 1.0], [1.0, 0], [1.0, 0]], [[2.0, 3.0], [3.0, 2.0], [3.0, 2.0]]],
                    name="abc",
                )
            ],
        )
    with pytest.raises(ValueError):
        ChoiceDataset(
            shared_features_by_choice=(shared_features_by_choice, [[2], [1], [0]]),
            items_features_by_choice=(items_features_by_choice,),
            shared_features_by_choice_names=(["a", "b"], ["c"]),
            available_items_by_choice=available_items_by_choice,
            choices=[0, 2, 1],
            features_by_ids=[
                FeaturesStorage(
                    ids=[0, 1],
                    values=[[0.0, 1.0], [2.0, 3.0]],
                    name="abc",
                )
            ],
        )
    # Test fbid not FeaturesStorage
    with pytest.raises(ValueError):
        ChoiceDataset(
            shared_features_by_choice=shared_features_by_choice,
            shared_features_by_choice_names=["budget", "age"],
            items_features_by_choice=items_features_by_choice,
            items_features_by_choice_names=["price", "promotion"],
            available_items_by_choice=available_items_by_choice,
            choices=choices,
            features_by_ids=[np.array([1, 2, 3])],
        )


def test_instant_from_df():
    """Various tests for CD instantiation with a DF."""
    dataset = ChoiceDataset(
        shared_features_by_choice=pd.DataFrame(
            {"budget": [100, 200, 80], "age": [20, 40, 20], "choice_id": [0, 1, 2]}
        ),
        items_features_by_choice=pd.DataFrame(
            {
                "price": [100, 140, 200, 100, 120, 200, 0, 120, 180],
                "promotion": [0, 0, 0, 0, 1, 0, 0, 1, 1],
                "choice_id": [0, 0, 0, 1, 1, 1, 2, 2, 2],
                "item_id": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            }
        ),
        available_items_by_choice=pd.DataFrame(
            {
                "av": [1, 1, 1, 1, 1, 1, 0, 1, 1],
                "choice_id": [0, 0, 0, 1, 1, 1, 2, 2, 2],
                "item_id": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            }
        ),
        choices=pd.DataFrame({"choice_id": [0, 1, 2], "choices": [0, 2, 1]}),
    )
    example_dataset = ChoiceDataset(
        shared_features_by_choice=shared_features_by_choice,
        items_features_by_choice=items_features_by_choice,
        available_items_by_choice=available_items_by_choice,
        choices=choices,
    )

    assert (dataset.choices == example_dataset.choices).all()
    for value, ex_value in zip(
        dataset.shared_features_by_choice, example_dataset.shared_features_by_choice
    ):
        assert (value == ex_value).all()
    for value, ex_value in zip(
        dataset.items_features_by_choice, example_dataset.items_features_by_choice
    ):
        assert (value == ex_value).all()
    assert (dataset.available_items_by_choice == example_dataset.available_items_by_choice).all()

    dataset = ChoiceDataset(
        shared_features_by_choice=pd.DataFrame(
            {"budget": [100, 200, 80], "age": [20, 40, 20], "choice_id": [0, 1, 2]}
        ),
        items_features_by_choice=pd.DataFrame(
            {
                "price": [100, 140, 200, 100, 120, 200, 0, 120, 180],
                "promotion": [0, 0, 0, 0, 1, 0, 0, 1, 1],
                "choice_id": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            }
        ),
        available_items_by_choice=pd.DataFrame(
            {
                "av": [1, 1, 1, 1, 1, 1, 0, 1, 1],
                "choice_id": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            }
        ),
        choices=pd.DataFrame({"choice_id": [0, 1, 2], "choices": [0, 2, 1]}),
    )
    for value, ex_value in zip(
        dataset.items_features_by_choice, example_dataset.items_features_by_choice
    ):
        assert (value == ex_value).all()
    assert (dataset.available_items_by_choice == example_dataset.available_items_by_choice).all()

    # availabilities as pd.DataFrame without choice_id without item_id
    dataset = ChoiceDataset(
        shared_features_by_choice=pd.DataFrame(
            {"budget": [100, 200, 80], "age": [20, 40, 20], "choice_id": [0, 1, 2]}
        ),
        items_features_by_choice=pd.DataFrame(
            {
                "price": [100, 140, 200, 100, 120, 200, 0, 120, 180],
                "promotion": [0, 0, 0, 0, 1, 0, 0, 1, 1],
                "choice_id": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            }
        ),
        available_items_by_choice=pd.DataFrame(
            {
                "av": [1, 1, 1, 1, 1, 1, 0, 1, 1],
            }
        ),
        choices=pd.DataFrame({"choice_id": [0, 1, 2], "choices": [0, 2, 1]}),
    )
    assert (dataset.available_items_by_choice == example_dataset.available_items_by_choice).all()

    # choices as pd.Series
    series_dataset = ChoiceDataset(
        shared_features_by_choice=shared_features_by_choice,
        items_features_by_choice=items_features_by_choice,
        available_items_by_choice=available_items_by_choice,
        choices=pd.Series(choices),
    )
    assert (series_dataset.choices == np.array(choices)).all()

    # items_features as DF no availabilities

    dataset = ChoiceDataset(
        shared_features_by_choice=pd.DataFrame(
            {"budget": [100, 200, 80], "age": [20, 40, 20], "choice_id": [0, 1, 2]}
        ),
        items_features_by_choice=pd.DataFrame(
            {
                "price": [100, 140, 200, 100, 120, 200, 120, 180],
                "promotion": [0, 0, 0, 0, 1, 0, 1, 1],
                "choice_id": [0, 0, 0, 1, 1, 1, 2, 2],
                "item_id": [0, 1, 2, 0, 1, 2, 1, 2],
            }
        ),
        choices=pd.DataFrame({"choice_id": [0, 1, 2], "choices": [0, 2, 1]}),
    )
    assert (dataset.available_items_by_choice == example_dataset.available_items_by_choice).all()


def test_instantiate_len():
    """Test the __init__ method."""
    choices = [0, 2, 1]
    dataset = ChoiceDataset(
        shared_features_by_choice=shared_features_by_choice,
        items_features_by_choice=items_features_by_choice,
        available_items_by_choice=available_items_by_choice,
        choices=choices,
    )
    assert len(dataset) == 3


def test_fail_instantiate():
    """Tests shapes fail to instantiate."""
    choices = [0, 1]
    with pytest.raises(ValueError):
        ChoiceDataset(
            shared_features_by_choice=shared_features_by_choice,
            items_features_by_choice=items_features_by_choice,
            available_items_by_choice=available_items_by_choice,
            choices=choices,
        )


def test_fail_instantiate_10():
    """Tests shapes fail to instantiate."""
    available_items_by_choice = [
        [1, 1, 1],  # All items available at session 1
        [1, 1, 1],  # All items available at session 2
    ]
    with pytest.raises(ValueError):
        ChoiceDataset(
            shared_features_by_choice=shared_features_by_choice,
            items_features_by_choice=items_features_by_choice,
            available_items_by_choice=available_items_by_choice,
            choices=choices,
        )


def test_fail_instantiate_4():
    """Tests shapes fail to instantiate."""
    available_items_by_choice = [
        [1, 1],  # All items available at session 1
        [1, 1],  # All items available at session 2
        [1, 1],  # All items available at session 3
    ]
    with pytest.raises(ValueError):
        ChoiceDataset(
            shared_features_by_choice=shared_features_by_choice,
            items_features_by_choice=items_features_by_choice,
            available_items_by_choice=available_items_by_choice,
            choices=choices,
        )


def test_fail_instantiate_5():
    """Tests shapes fail to instantiate."""
    shared_features_by_choice = [
        [100, 20],  # session 1, customer 1 [budget, age]
        [200, 40],  # session 2, customer 2 [budget, age]
    ]
    with pytest.raises(ValueError):
        ChoiceDataset(
            shared_features_by_choice=shared_features_by_choice,
            items_features_by_choice=items_features_by_choice,
            available_items_by_choice=available_items_by_choice,
            choices=choices,
        )


def test_fail_instantiate_6():
    """Tests shapes fail to instantiate."""
    shared_features_by_choice = [
        [100],  # session 1, customer 1 [budget, age]
        [200, 40],  # session 2, customer 2 [budget, age]
        [80, 20],  # session 3, customer 1 [budget, age]
    ]
    with pytest.raises(ValueError):
        ChoiceDataset(
            shared_features_by_choice=shared_features_by_choice,
            items_features_by_choice=items_features_by_choice,
            available_items_by_choice=available_items_by_choice,
            choices=choices,
        )


def test_fail_instantiate_7():
    """Tests shapes fail to instantiate."""
    items_features_by_choice = [
        [
            [100, 0],  # Session 1, Item 1 [price, promotion]
            [140, 0],  # Session 1, Item 2 [price, promotion]
            [200, 0],  # Session 1, Item 2 [price, promotion]
        ],
        [
            [100, 0],  # Session 2 Item 1 [price, promotion]
            [120, 1],  # Session 2, Item 2 [price, promotion]
            [200, 0],  # Session 2, Item 2 [price, promotion]
        ],
    ]
    with pytest.raises(ValueError):
        ChoiceDataset(
            shared_features_by_choice=shared_features_by_choice,
            items_features_by_choice=items_features_by_choice,
            available_items_by_choice=available_items_by_choice,
            choices=choices,
        )


def test_fail_instantiate_8():
    """Tests shapes fail to instantiate."""
    items_features_by_choice = [
        [
            [100, 0],  # Session 1, Item 1 [price, promotion]
            [140, 0],  # Session 1, Item 2 [price, promotion]
        ],
        [
            [100, 0],  # Session 2 Item 1 [price, promotion]
            [120, 1],  # Session 2, Item 2 [price, promotion]
        ],
        [
            [100, 0],  # Session 3, Item 1 [price, promotion],
            # values do not really matter, but needs to exist for shapes sake
            [120, 1],  # Session 3, Item 2 [price, promotion]
        ],
    ]
    with pytest.raises(ValueError):
        ChoiceDataset(
            shared_features_by_choice=shared_features_by_choice,
            items_features_by_choice=items_features_by_choice,
            available_items_by_choice=available_items_by_choice,
            choices=choices,
        )


def test_fail_instantiate_9():
    """Tests shapes fail to instantiate."""
    choices = [0, 4, 2]  # choices higher than nb of items
    with pytest.raises(ValueError):
        ChoiceDataset(
            shared_features_by_choice=shared_features_by_choice,
            items_features_by_choice=items_features_by_choice,
            available_items_by_choice=available_items_by_choice,
            choices=choices,
        )


def test_shape():
    """Tests get shape methods."""
    dataset = ChoiceDataset(
        shared_features_by_choice=shared_features_by_choice,
        items_features_by_choice=items_features_by_choice,
        available_items_by_choice=available_items_by_choice,
        choices=choices,
    )

    assert dataset.get_n_items() == 3
    assert dataset.get_n_choices() == 3


def test_from_df():
    """Tests from_df method."""
    features_df = pd.DataFrame(
        {
            "item_id": [0, 1, 2, 0, 1, 2, 1, 2],
            "items_feat_1": [1, 2, 1.5, 1, 2, 1.5, 2, 1.5],
            "items_feat_2": [2, 4, 1.5, 2, 4, 1.5, 4, 1.5],
            "choice_id": [0, 0, 0, 1, 1, 1, 2, 2],
            "session_feat_1": [100, 100, 100, 200, 200, 200, 80, 80],
            "session_feat_2": [20, 20, 20, 40, 40, 40, 20, 20],
            "session_item_feat_1": [100, 140, 200, 100, 120, 200, 120, 180],
            "session_item_feat_2": [0, 0, 0, 0, 1, 0, 1, 1],
            "choice": [0, 0, 0, 2, 2, 2, 1, 1],
        }
    )
    cd_test = ChoiceDataset.from_single_long_df(
        features_df,
        shared_features_columns=["session_feat_1", "session_feat_2"],
        items_features_columns=["session_item_feat_1", "session_item_feat_2"],
        choice_format="items_id",
        choices_column="choice",
    )
    ground_truth_cd = ChoiceDataset(
        shared_features_by_choice=shared_features_by_choice,
        items_features_by_choice=items_features_by_choice,
        available_items_by_choice=available_items_by_choice,
        choices=choices,
    )
    assert (
        cd_test.shared_features_by_choice[0] == ground_truth_cd.shared_features_by_choice[0]
    ).all()
    assert (
        cd_test.items_features_by_choice[0].astype("float32")
        == ground_truth_cd.items_features_by_choice[0].astype("float32")
    ).all()
    assert (cd_test.available_items_by_choice == ground_truth_cd.available_items_by_choice).all()
    assert (cd_test.choices == ground_truth_cd.choices).all()

    cd_test = ChoiceDataset.from_single_long_df(
        features_df,
        shared_features_columns=["session_feat_1", "session_feat_2"],
        items_features_columns=None,
        choice_format="items_id",
        choices_column="choice",
    )
    assert cd_test.items_features_by_choice is None
    assert cd_test.items_features_by_choice_names is None

    features_df = pd.DataFrame(
        {
            "item_id": [0, 1, 2, 0, 1, 2, 1, 2],
            "items_feat_1": [1, 2, 1.5, 1, 2, 1.5, 2, 1.5],
            "items_feat_2": [2, 4, 1.5, 2, 4, 1.5, 4, 1.5],
            "choice_id": [0, 0, 0, 1, 1, 1, 2, 2],
            "session_feat_1": [100, 100, 100, 200, 200, 200, 80, 80],
            "session_feat_2": [20, 20, 20, 40, 40, 40, 20, 20],
            "session_item_feat_1": [100, 140, 200, 100, 120, 200, 120, 180],
            "session_item_feat_2": [0, 0, 0, 0, 1, 0, 1, 1],
            "choice": [1, 0, 0, 0, 0, 1, 1, 0],
        }
    )
    cd_test = ChoiceDataset.from_single_long_df(
        features_df,
        shared_features_columns=["session_feat_1", "session_feat_2"],
        items_features_columns=["session_item_feat_1", "session_item_feat_2"],
        choice_format="one_zero",
    )
    ground_truth_cd = ChoiceDataset(
        shared_features_by_choice=shared_features_by_choice,
        items_features_by_choice=items_features_by_choice,
        available_items_by_choice=available_items_by_choice,
        choices=choices,
    )
    assert (
        cd_test.shared_features_by_choice[0] == ground_truth_cd.shared_features_by_choice[0]
    ).all()
    assert (
        cd_test.items_features_by_choice[0].astype("float32")
        == ground_truth_cd.items_features_by_choice[0].astype("float32")
    ).all()
    assert (cd_test.available_items_by_choice == ground_truth_cd.available_items_by_choice).all()
    assert (cd_test.choices == ground_truth_cd.choices).all()

    with pytest.raises(ValueError):
        ChoiceDataset.from_single_long_df(
            features_df,
            shared_features_columns=["session_feat_1", "session_feat_2"],
            items_features_columns=["session_item_feat_1", "session_item_feat_2"],
            choice_format="items_index",
            choices_column="choice",
        )


def test_from_wide_df():
    """Diverse tests when instantiating from a single wide df."""
    wide_df = {
        "sh_1": [1.1, 2.2, 3.3],
        "sh_2": [11.1, 22.2, 33.3],
        "it_1_1": [0.4, 0.5, 0.6],
        "it_2_1": [0.7, 0.8, 0.9],
        "it_1_2": [1.4, 1.5, 1.6],
        "it_2_2": [1.7, 1.8, 1.9],
        "it_1_3": [2.4, 2.5, 2.6],
        "it_2_3": [2.7, 2.8, 2.9],
        "av_it_1": [1, 1, 1],
        "av_it_2": [1, 0, 1],
        "choice": ["it_1", "it_1", "it_2"],
    }
    dataset = ChoiceDataset.from_single_wide_df(
        df=pd.DataFrame(wide_df),
        items_id=["it_1", "it_2"],
        shared_features_columns=["sh_1", "sh_2"],
        items_features_suffixes=["1", "2", "3"],
        available_items_suffix=["av_it_1", "av_it_2"],
        choices_column="choice",
        choice_format="items_id",
    )
    assert (
        dataset.available_items_by_choice == np.array([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    ).all()
    assert (dataset.choices == np.array([0, 0, 1])).all()
    assert (
        dataset.shared_features_by_choice[0] == np.array([[1.1, 11.1], [2.2, 22.2], [3.3, 33.3]])
    ).all()
    assert dataset.shared_features_by_choice_names == (["sh_1", "sh_2"],)

    dataset = ChoiceDataset.from_single_wide_df(
        df=pd.DataFrame(wide_df),
        items_id=["it_1", "it_2"],
        shared_features_columns=None,
        items_features_suffixes=["1", "2", "3"],
        available_items_suffix=["av_it_1", "av_it_2"],
        choices_column="choice",
        choice_format="items_id",
    )
    assert dataset.shared_features_by_choice is None
    assert dataset.shared_features_by_choice_names is None
    assert dataset.items_features_by_choice_names == (["1", "2", "3"],)
    assert (
        dataset.items_features_by_choice
        == np.array(
            [
                [[0.4, 1.4, 2.4], [0.7, 1.7, 2.7]],
                [[0.5, 1.5, 2.5], [0.8, 1.8, 2.8]],
                [[0.6, 1.6, 2.6], [0.9, 1.9, 2.9]],
            ],
            dtype=np.float64,
        )
    ).all()

    with pytest.raises(ValueError):
        ChoiceDataset.from_single_wide_df(
            df=pd.DataFrame(wide_df),
            items_id=["it_1", "it_2"],
            shared_features_columns=None,
            items_features_suffixes=["1", "2", "3"],
            available_items_suffix=["av_it_1", "av_it_2"],
            available_items_prefix=["av_it_1", "av_it_2"],
            choices_column="choice",
            choice_format="items_id",
        )

    with pytest.raises(ValueError):
        ChoiceDataset.from_single_wide_df(
            df=pd.DataFrame(wide_df),
            items_id=["it_1", "it_2"],
            shared_features_columns=["sh_1", "sh_2"],
            items_features_suffixes=["1", "2", "3"],
            available_items_suffix=["av_it_1", "av_it_2", "av_it_3"],
            choices_column="choice",
            choice_format="items_id",
        )

    dataset = ChoiceDataset.from_single_wide_df(
        df=pd.DataFrame(wide_df),
        items_id=["it_1", "it_2"],
        shared_features_columns=["sh_1", "sh_2"],
        items_features_suffixes=["1", "2"],
        available_items_prefix=["av_it_1", "av_it_2"],
        choices_column="choice",
        choice_format="items_id",
    )
    assert (
        dataset.available_items_by_choice == np.array([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    ).all()
    assert (dataset.choices == np.array([0, 0, 1])).all()
    with pytest.raises(ValueError):
        ChoiceDataset.from_single_wide_df(
            df=pd.DataFrame(wide_df),
            items_id=["it_1", "it_2"],
            shared_features_columns=["sh_1", "sh_2"],
            items_features_suffixes=["1", "2"],
            available_items_prefix=["av_it_1", "av_it_2", "av_it_3"],
            choices_column="choice",
            choice_format="items_id",
        )
    dataset = ChoiceDataset.from_single_wide_df(
        df=pd.DataFrame(wide_df),
        items_id=["it_1", "it_2"],
        shared_features_columns=["sh_1", "sh_2"],
        items_features_suffixes=["1", "2"],
        available_items_prefix="av",
        choices_column="choice",
        choice_format="items_id",
    )
    assert (
        dataset.available_items_by_choice == np.array([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    ).all()
    assert (dataset.choices == np.array([0, 0, 1])).all()

    with pytest.raises(ValueError):
        ChoiceDataset.from_single_wide_df(
            df=pd.DataFrame(wide_df),
            items_id=None,
            shared_features_columns=["sh_1", "sh_2"],
            choices_column="choice",
            choice_format="items_id",
        )
    with pytest.raises(ValueError):
        wide_df_false = {
            "sh_1": [1.1, 2.2, 3.3],
            "sh_2": [11.1, 22.2, 33.3],
            "it_1_1": [0.4, 0.5, 0.6],
            "it_2_1": [0.7, 0.8, 0.9],
            "it_1_2": [1.4, 1.5, 1.6],
            "it_2_2": [1.7, 1.8, 1.9],
            "it_1_3": [2.4, 2.5, 2.6],
            "it_2_3": [2.7, 2.8, 2.9],
            "av_it_1": [1, 1, 1],
            "av_it_2": [1, 0, 1],
            "choice": ["it_3", "it_3", "it_4"],
        }
        ChoiceDataset.from_single_wide_df(
            df=pd.DataFrame(wide_df_false),
            items_id=["it_1", "it_2"],
            shared_features_columns=["sh_1", "sh_2"],
            items_features_suffixes=["1", "2"],
            available_items_prefix="av",
            choices_column="choice",
            choice_format="items_id",
        )

    extra_wide_df = {
        "sh_1": [1.1, 2.2, 3.3],
        "sh_2": [11.1, 22.2, 33.3],
        "it_1_1": [0.4, 0.5, 0.6],
        "it_2_1": [0.7, 0.8, 0.9],
        "it_1_2": [1.4, 1.5, 1.6],
        "it_2_2": [1.7, 1.8, 1.9],
        "it_1_3": [2.4, 2.5, 2.6],
        "it_2_3": [2.7, 2.8, 2.9],
        "1_it_1": [3.4, 3.5, 3.6],
        "1_it_2": [3.7, 3.8, 3.9],
        "2_it_1": [4.4, 4.5, 4.6],
        "2_it_2": [4.7, 4.8, 4.9],
        "3_it_1": [5.4, 5.5, 5.6],
        "3_it_2": [5.7, 5.8, 5.9],
        "av_it_1": [1, 1, 1],
        "av_it_2": [1, 0, 1],
        "choice": ["it_1", "it_1", "it_2"],
    }
    dataset = ChoiceDataset.from_single_wide_df(
        df=pd.DataFrame(extra_wide_df),
        items_id=["it_1", "it_2"],
        shared_features_columns=None,
        items_features_prefixes=["1", "2", "3"],
        items_features_suffixes=["1", "2", "3"],
        available_items_suffix=["av_it_1", "av_it_2"],
        choices_column="choice",
        choice_format="items_id",
    )
    assert dataset.shared_features_by_choice is None
    assert dataset.shared_features_by_choice_names is None
    assert dataset.items_features_by_choice_names == (["1", "2", "3", "1", "2", "3"],)
    assert (
        dataset.items_features_by_choice
        == np.array(
            [
                [
                    [3.4, 4.4, 5.4, 0.4, 1.4, 2.4],
                    [3.7, 4.7, 5.7, 0.7, 1.7, 2.7],
                ],
                [
                    [3.5, 4.5, 5.5, 0.5, 1.5, 2.5],
                    [3.8, 4.8, 5.8, 0.8, 1.8, 2.8],
                ],
                [
                    [3.6, 4.6, 5.6, 0.6, 1.6, 2.6],
                    [3.9, 4.9, 5.9, 0.9, 1.9, 2.9],
                ],
            ],
            dtype=np.float64,
        )
    ).all()


def test_summary():
    """Tests summary method."""
    dataset = ChoiceDataset(
        shared_features_by_choice=shared_features_by_choice,
        items_features_by_choice=items_features_by_choice,
        available_items_by_choice=available_items_by_choice,
        choices=choices,
    )
    dataset.summary()
    assert True


def test_getitem():
    """Tests getitem method."""
    dataset = ChoiceDataset(
        shared_features_by_choice=shared_features_by_choice,
        shared_features_by_choice_names=["budget", "age"],
        items_features_by_choice=items_features_by_choice,
        items_features_by_choice_names=["price", "promotion"],
        available_items_by_choice=available_items_by_choice,
        choices=choices,
    )

    sub_dataset = dataset[[0, 1]]
    assert (
        sub_dataset.shared_features_by_choice[0] == dataset.shared_features_by_choice[0][[0, 1]]
    ).all()
    assert (
        sub_dataset.items_features_by_choice[0] == dataset.items_features_by_choice[0][[0, 1]]
    ).all()
    assert (
        sub_dataset.available_items_by_choice == dataset.available_items_by_choice[[0, 1]]
    ).all()
    assert (sub_dataset.choices == dataset.choices[[0, 1]]).all()
    assert (sub_dataset.choices == [0, 2]).all()

    sliced_sub_dataset = dataset[:2]
    assert (
        sub_dataset.shared_features_by_choice[0] == sliced_sub_dataset.shared_features_by_choice[0]
    ).all()
    assert (
        sub_dataset.items_features_by_choice[0] == sliced_sub_dataset.items_features_by_choice[0]
    ).all()
    assert (sub_dataset.choices == sliced_sub_dataset.choices).all()

    sub_dataset = dataset[2]
    assert (sub_dataset.choices == [1]).all()
    assert sub_dataset.shared_features_by_choice_names[0] == ["budget", "age"]
    assert sub_dataset.items_features_by_choice_names[0] == ["price", "promotion"]

    dataset = ChoiceDataset(
        shared_features_by_choice=None,
        items_features_by_choice=None,
        available_items_by_choice=np.array([0, 0, 1]),
        choices=choices,
        features_by_ids=[
            FeaturesStorage(
                ids=[0, 1], values=[[1, 1, 1], [1, 0, 1]], name="available_items_by_choice"
            )
        ],
    )
    assert dataset.get_n_shared_features() == 0
    assert dataset.get_n_items_features() == 0
    sub_dataset = dataset[[0, 1]]
    assert sub_dataset.shared_features_by_choice is None
    assert sub_dataset.items_features_by_choice is None
    assert (
        sub_dataset.indexer.get_full_dataset()[2] == np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    ).all()

    # wrong nb of indexes in available_items_by_choice
    with pytest.raises(ValueError):
        ChoiceDataset(
            shared_features_by_choice=None,
            items_features_by_choice=None,
            available_items_by_choice=np.array([0, 0, 1, 0]),
            choices=choices,
            features_by_ids=[
                FeaturesStorage(
                    ids=[0, 1], values=[[1, 1, 1], [1, 0, 1]], name="available_items_by_choice"
                )
            ],
        )
    # missing indexes with availabilities as FS
    with pytest.raises(ValueError):
        ChoiceDataset(
            shared_features_by_choice=None,
            items_features_by_choice=None,
            available_items_by_choice=None,
            choices=choices,
            features_by_ids=[
                FeaturesStorage(
                    ids=[0, 1], values=[[1, 1, 1], [1, 0, 1]], name="available_items_by_choice"
                )
            ],
        )


def test_batch():
    """Tests the batch method."""
    dataset = ChoiceDataset(
        shared_features_by_choice=shared_features_by_choice,
        items_features_by_choice=items_features_by_choice,
        available_items_by_choice=available_items_by_choice,
        choices=choices,
    )
    batch = dataset.batch[[0, 1]]
    assert (batch[0] == shared_features_by_choice[:2]).all()
    assert (batch[1] == items_features_by_choice[:2]).all()
    assert (batch[2] == available_items_by_choice[:2]).all()
    assert (batch[3] == choices[:2]).all()

    sliced_batch = dataset.batch[:2]
    assert (batch[0] == sliced_batch[0]).all()
    assert (batch[1] == sliced_batch[1]).all()
    assert (batch[2] == sliced_batch[2]).all()
    assert (batch[3] == sliced_batch[3]).all()

    single_batch = dataset.batch[0]
    assert (batch[0][0] == single_batch[0]).all()
    assert (batch[1][0] == single_batch[1]).all()
    assert (batch[2][0] == single_batch[2]).all()
    assert (batch[3][0] == single_batch[3]).all()


def test_iter_batch():
    """Tests the iter_batch method."""
    dataset = ChoiceDataset(
        shared_features_by_choice=shared_features_by_choice,
        items_features_by_choice=items_features_by_choice,
        available_items_by_choice=available_items_by_choice,
        choices=choices,
    )
    for batch_nb, batch in enumerate(dataset.iter_batch(batch_size=2)):
        assert batch[0].shape[1] == 2
        assert batch[1].shape[1] == 3
        assert batch[1].shape[2] == 2
        assert batch[2].shape[1] == 3
        assert batch[3].shape[0] == 2 or batch[3].shape[0] == 1
    assert batch_nb == 1
    for batch_nb, (batch, weight) in enumerate(
        dataset.iter_batch(batch_size=2, sample_weight=np.array([1.0, 2.0, 0.8]))
    ):
        assert batch[0].shape[1] == 2
        assert batch[1].shape[1] == 3
        assert batch[1].shape[2] == 2
        assert batch[2].shape[1] == 3
        assert batch[3].shape[0] == 2 or batch[3].shape[0] == 1
        assert len(weight) == len(batch[3])
    assert batch_nb == 1


def test_filter():
    """Tests the filter method."""
    dataset = ChoiceDataset(
        shared_features_by_choice=shared_features_by_choice,
        items_features_by_choice=items_features_by_choice,
        available_items_by_choice=available_items_by_choice,
        choices=choices,
    )
    filtered_dataset = dataset.filter([True, False, True])
    assert len(filtered_dataset) == 2
    assert (
        filtered_dataset.shared_features_by_choice[0]
        == dataset.shared_features_by_choice[0][[0, 2]]
    ).all()
    assert (
        filtered_dataset.items_features_by_choice[0] == dataset.items_features_by_choice[0][[0, 2]]
    ).all()
    assert (
        filtered_dataset.available_items_by_choice == dataset.available_items_by_choice[[0, 2]]
    ).all()
    assert (filtered_dataset.choices == dataset.choices[[0, 2]]).all()
    assert (filtered_dataset.choices == [0, 1]).all()
