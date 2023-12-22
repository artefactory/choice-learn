"""Unit testing for class ChoiceDataset."""
import pandas as pd
import pytest

from choice_learn.data.choice_dataset import ChoiceDataset

items_features = [
    [1, 2],  # item 1 [size, weight]
    [2, 4],  # item 2 [size, weight]
    [1.5, 1.5],  # item 3 [size, weight]
]

# We have two customers whose features are
# - Budget
# - Age
# Customer 1 bought item 1 at session 1 and item 2 at session 3
# Customer 2 bought item 3 at session 2

choices = [0, 2, 1]
sessions_items_availabilities = [
    [1, 1, 1],  # All items available at session 1
    [1, 1, 1],  # All items available at session 2
    [0, 1, 1],  # Item 1 not available at session 3
]

sessions_features = [
    [100, 20],  # session 1, customer 1 [budget, age]
    [200, 40],  # session 2, customer 2 [budget, age]
    [80, 20],  # session 3, customer 1 [budget, age]
]

sessions_items_features = [
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


def test_instantiate_len():
    """Test the __init__ method."""
    choices = [0, 2, 1]
    dataset = ChoiceDataset(
        items_features=items_features,
        sessions_features=sessions_features,
        sessions_items_features=sessions_items_features,
        sessions_items_availabilities=sessions_items_availabilities,
        choices=choices,
    )
    assert len(dataset) == 3
    choices = [[0], [1, 2], [2, 1, 1, 1]]
    dataset = ChoiceDataset(
        items_features=items_features,
        sessions_features=sessions_features,
        sessions_items_features=sessions_items_features,
        sessions_items_availabilities=sessions_items_availabilities,
        choices=choices,
    )
    assert len(dataset) == 3


def test_fail_instantiate():
    """Tests shapes fail to instantiate."""
    choices = [0, 1]
    with pytest.raises(ValueError):
        ChoiceDataset(
            items_features=items_features,
            sessions_features=sessions_features,
            sessions_items_features=sessions_items_features,
            sessions_items_availabilities=sessions_items_availabilities,
            choices=choices,
        )


def test_fail_instantiate_2():
    """Tests shapes fail to instantiate."""
    items_features = [
        [1, 2],  # item 1 [size, weight]
        [2, 4],  # item 2 [size, weight]
        [1.5],  # item 3 [size, weight]
    ]
    with pytest.raises(ValueError):
        ChoiceDataset(
            items_features=items_features,
            sessions_features=sessions_features,
            sessions_items_features=sessions_items_features,
            sessions_items_availabilities=sessions_items_availabilities,
            choices=choices,
        )


def test_fail_instantiate_3():
    """Tests shapes fail to instantiate."""
    items_features = [
        [1, 2],  # item 1 [size, weight]
        [2, 4],  # item 2 [size, weight]
    ]
    with pytest.raises(ValueError):
        ChoiceDataset(
            items_features=items_features,
            sessions_features=sessions_features,
            sessions_items_features=sessions_items_features,
            sessions_items_availabilities=sessions_items_availabilities,
            choices=choices,
        )


def test_fail_instantiate_10():
    """Tests shapes fail to instantiate."""
    sessions_items_availabilities = [
        [1, 1, 1],  # All items available at session 1
        [1, 1, 1],  # All items available at session 2
    ]
    with pytest.raises(ValueError):
        ChoiceDataset(
            items_features=items_features,
            sessions_features=sessions_features,
            sessions_items_features=sessions_items_features,
            sessions_items_availabilities=sessions_items_availabilities,
            choices=choices,
        )


def test_fail_instantiate_4():
    """Tests shapes fail to instantiate."""
    sessions_items_availabilities = [
        [1, 1],  # All items available at session 1
        [1, 1],  # All items available at session 2
        [1, 1],  # All items available at session 3
    ]
    with pytest.raises(ValueError):
        ChoiceDataset(
            items_features=items_features,
            sessions_features=sessions_features,
            sessions_items_features=sessions_items_features,
            sessions_items_availabilities=sessions_items_availabilities,
            choices=choices,
        )


def test_fail_instantiate_5():
    """Tests shapes fail to instantiate."""
    sessions_features = [
        [100, 20],  # session 1, customer 1 [budget, age]
        [200, 40],  # session 2, customer 2 [budget, age]
    ]
    with pytest.raises(ValueError):
        ChoiceDataset(
            items_features=items_features,
            sessions_features=sessions_features,
            sessions_items_features=sessions_items_features,
            sessions_items_availabilities=sessions_items_availabilities,
            choices=choices,
        )


def test_fail_instantiate_6():
    """Tests shapes fail to instantiate."""
    sessions_features = [
        [100],  # session 1, customer 1 [budget, age]
        [200, 40],  # session 2, customer 2 [budget, age]
        [80, 20],  # session 3, customer 1 [budget, age]
    ]
    with pytest.raises(ValueError):
        ChoiceDataset(
            items_features=items_features,
            sessions_features=sessions_features,
            sessions_items_features=sessions_items_features,
            sessions_items_availabilities=sessions_items_availabilities,
            choices=choices,
        )


def test_fail_instantiate_7():
    """Tests shapes fail to instantiate."""
    sessions_items_features = [
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
            items_features=items_features,
            sessions_features=sessions_features,
            sessions_items_features=sessions_items_features,
            sessions_items_availabilities=sessions_items_availabilities,
            choices=choices,
        )


def test_fail_instantiate_8():
    """Tests shapes fail to instantiate."""
    sessions_items_features = [
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
            items_features=items_features,
            sessions_features=sessions_features,
            sessions_items_features=sessions_items_features,
            sessions_items_availabilities=sessions_items_availabilities,
            choices=choices,
        )


def test_fail_instantiate_9():
    """Tests shapes fail to instantiate."""
    choices = [0, 4, 2]  # choices higher than nb of items
    with pytest.raises(ValueError):
        ChoiceDataset(
            items_features=items_features,
            sessions_features=sessions_features,
            sessions_items_features=sessions_items_features,
            sessions_items_availabilities=sessions_items_availabilities,
            choices=choices,
        )


def test_shape():
    """Tests get shape methods."""
    dataset = ChoiceDataset(
        items_features=items_features,
        sessions_features=sessions_features,
        sessions_items_features=sessions_items_features,
        sessions_items_availabilities=sessions_items_availabilities,
        choices=choices,
    )

    assert dataset.get_num_items() == 3
    assert dataset.get_num_sessions() == 3
    assert dataset.get_num_choices() == 3


def test_from_df():
    """Tests from_df method."""
    features_df = pd.DataFrame(
        {
            "item_id": [0, 1, 2, 0, 1, 2, 1, 2],
            "items_feat_1": [1, 2, 1.5, 1, 2, 1.5, 2, 1.5],
            "items_feat_2": [2, 4, 1.5, 2, 4, 1.5, 4, 1.5],
            "session_id": [0, 0, 0, 1, 1, 1, 2, 2],
            "session_feat_1": [100, 100, 100, 200, 200, 200, 80, 80],
            "session_feat_2": [20, 20, 20, 40, 40, 40, 20, 20],
            "session_item_feat_1": [100, 140, 200, 100, 120, 200, 120, 180],
            "session_item_feat_2": [0, 0, 0, 0, 1, 0, 1, 1],
            "choice": [0, 0, 0, 2, 2, 2, 1, 1],
        }
    )
    cd_test = ChoiceDataset.from_single_df(
        features_df,
        items_features_columns=["items_feat_1", "items_feat_2"],
        sessions_features_columns=["session_feat_1", "session_feat_2"],
        sessions_items_features_columns=["session_item_feat_1", "session_item_feat_2"],
        choice_mode="item_id",
    )
    ground_truth_cd = ChoiceDataset(
        items_features=items_features,
        sessions_features=sessions_features,
        sessions_items_features=sessions_items_features,
        sessions_items_availabilities=sessions_items_availabilities,
        choices=choices,
    )
    assert (cd_test.items_features[0] == ground_truth_cd.items_features[0]).all()
    assert (cd_test.sessions_features[0] == ground_truth_cd.sessions_features[0]).all()
    assert (
        cd_test.sessions_items_features[0].astype("float32")
        == ground_truth_cd.sessions_items_features[0].astype("float32")
    ).all()
    assert (
        cd_test.sessions_items_availabilities == ground_truth_cd.sessions_items_availabilities
    ).all()
    assert (cd_test.choices == ground_truth_cd.choices).all()

    features_df = pd.DataFrame(
        {
            "item_id": [0, 1, 2, 0, 1, 2, 1, 2],
            "items_feat_1": [1, 2, 1.5, 1, 2, 1.5, 2, 1.5],
            "items_feat_2": [2, 4, 1.5, 2, 4, 1.5, 4, 1.5],
            "session_id": [0, 0, 0, 1, 1, 1, 2, 2],
            "session_feat_1": [100, 100, 100, 200, 200, 200, 80, 80],
            "session_feat_2": [20, 20, 20, 40, 40, 40, 20, 20],
            "session_item_feat_1": [100, 140, 200, 100, 120, 200, 120, 180],
            "session_item_feat_2": [0, 0, 0, 0, 1, 0, 1, 1],
            "choice": [1, 0, 0, 0, 0, 1, 1, 0],
        }
    )
    cd_test = ChoiceDataset.from_single_df(
        features_df,
        items_features_columns=["items_feat_1", "items_feat_2"],
        sessions_features_columns=["session_feat_1", "session_feat_2"],
        sessions_items_features_columns=["session_item_feat_1", "session_item_feat_2"],
        choice_mode="one_zero",
    )
    ground_truth_cd = ChoiceDataset(
        items_features=items_features,
        sessions_features=sessions_features,
        sessions_items_features=sessions_items_features,
        sessions_items_availabilities=sessions_items_availabilities,
        choices=choices,
    )
    assert (cd_test.items_features[0] == ground_truth_cd.items_features[0]).all()
    assert (cd_test.sessions_features[0] == ground_truth_cd.sessions_features[0]).all()
    assert (
        cd_test.sessions_items_features[0].astype("float32")
        == ground_truth_cd.sessions_items_features[0].astype("float32")
    ).all()
    assert (
        cd_test.sessions_items_availabilities == ground_truth_cd.sessions_items_availabilities
    ).all()
    assert (cd_test.choices == ground_truth_cd.choices).all()


def test_summary():
    """Tests summary method."""
    dataset = ChoiceDataset(
        items_features=items_features,
        sessions_features=sessions_features,
        sessions_items_features=sessions_items_features,
        sessions_items_availabilities=sessions_items_availabilities,
        choices=choices,
    )
    dataset.summary()
    assert True


def test_getitem():
    """Tests getitem method."""
    dataset = ChoiceDataset(
        items_features=items_features,
        sessions_features=sessions_features,
        sessions_items_features=sessions_items_features,
        sessions_items_availabilities=sessions_items_availabilities,
        choices=choices,
    )

    sub_dataset = dataset[[0, 1]]
    assert (sub_dataset.items_features[0] == dataset.items_features[0]).all()
    assert (sub_dataset.sessions_features[0] == dataset.sessions_features[0][[0, 1]]).all()
    assert (
        sub_dataset.sessions_items_features[0] == dataset.sessions_items_features[0][[0, 1]]
    ).all()
    assert (
        sub_dataset.sessions_items_availabilities == dataset.sessions_items_availabilities[[0, 1]]
    ).all()
    assert (sub_dataset.choices == dataset.choices[[0, 1]]).all()
    assert (sub_dataset.choices == [0, 2]).all()


def test_batch():
    """Tests the batch method."""
    dataset = ChoiceDataset(
        items_features=items_features,
        sessions_features=sessions_features,
        sessions_items_features=sessions_items_features,
        sessions_items_availabilities=sessions_items_availabilities,
        choices=choices,
    )
    batch = dataset.batch[[0, 1]]
    assert (batch[0] == items_features).all()
    assert (batch[1] == sessions_features[:2]).all()
    assert (batch[2] == sessions_items_features[:2]).all()
    assert (batch[3] == sessions_items_availabilities[:2]).all()
    assert (batch[4] == choices[:2]).all()

    sliced_batch = dataset.batch[:2]
    assert (batch[0] == sliced_batch[0]).all()
    assert (batch[1] == sliced_batch[1]).all()
    assert (batch[2] == sliced_batch[2]).all()
    assert (batch[3] == sliced_batch[3]).all()
    assert (batch[4] == sliced_batch[4]).all()

    single_batch = dataset.batch[0]
    assert (batch[0] == single_batch[0]).all()
    assert (batch[1][0] == single_batch[1]).all()
    assert (batch[2][0] == single_batch[2]).all()
    assert (batch[3][0] == single_batch[3]).all()
    assert (batch[4][0] == single_batch[4]).all()


def test_iter_batch():
    """Tests the iter_batch method."""
    dataset = ChoiceDataset(
        items_features=items_features,
        sessions_features=sessions_features,
        sessions_items_features=sessions_items_features,
        sessions_items_availabilities=sessions_items_availabilities,
        choices=choices,
    )
    for batch_nb, batch in enumerate(dataset.iter_batch(batch_size=2)):
        assert batch[0].shape[1] == 2
        assert batch[1].shape[1] == 2
        assert batch[2].shape[1] == 3
        assert batch[2].shape[2] == 2
        assert batch[3].shape[1] == 3
        assert batch[4].shape[0] == 2 or batch[4].shape[0] == 1
    assert batch_nb == 1
