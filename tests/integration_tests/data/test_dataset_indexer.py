"""Test Indexer + ChoiceDataset + Storage."""

import numpy as np
import pytest

from choice_learn.data import ChoiceDataset, FeaturesStorage, OneHotStorage


def test_cd_indexer():
    """Various indexations of a choice dataset."""
    shared_features = np.array([[0.0, 1.0], [2.0, 4.0]])
    items_features = np.array([[[0.5, 0.4], [0.2, 0.3]], [[0.4, 0.5], [0.3, 0.2]]])
    choices = [0, 1]

    # Tests with None as features
    dataset = ChoiceDataset(
        shared_features_by_choice=None, items_features_by_choice=items_features, choices=choices
    )
    assert dataset.batch[[0, 1]][0] is None
    full_dataset = dataset.indexer.get_full_dataset()
    assert full_dataset[0] is None
    assert (
        full_dataset[1]
        == np.array([[[0.5, 0.4], [0.2, 0.3]], [[0.4, 0.5], [0.3, 0.2]]], dtype=np.float32)
    ).all()

    dataset = ChoiceDataset(
        shared_features_by_choice=shared_features, items_features_by_choice=None, choices=choices
    )
    assert dataset.batch[[0, 1]][1] is None
    full_dataset = dataset.indexer.get_full_dataset()
    assert full_dataset[1] is None

    # Test with FeaturesStorage as availabilities
    dataset = ChoiceDataset(
        shared_features_by_choice=(shared_features,),
        items_features_by_choice=(items_features,),
        choices=choices,
    )
    full_dataset = dataset.indexer.get_full_dataset()
    assert isinstance(full_dataset[0], tuple)
    assert isinstance(full_dataset[1], tuple)
    assert len(full_dataset[0]) == 1
    assert len(full_dataset[1]) == 1
    assert (full_dataset[0][0] == shared_features.astype(np.float32)).all()
    assert (full_dataset[1][0] == items_features.astype(np.float32)).all()

    # Test with FeaturesStorage as availabilities
    dataset = ChoiceDataset(
        shared_features_by_choice=shared_features,
        items_features_by_choice=items_features,
        available_items_by_choice=np.array([0, 1]),
        choices=choices,
        features_by_ids=[
            FeaturesStorage(ids=[0, 1], values=[[1, 0], [0, 1]], name="available_items_by_choice")
        ],
    )
    assert (dataset.batch[[0, 1]][2] == np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)).all()
    full_dataset = dataset.indexer.get_full_dataset()
    assert (full_dataset[2] == np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)).all()

    # Test 1-D items_features with FeaturesStorage
    dataset = ChoiceDataset(
        shared_features_by_choice=shared_features,
        items_features_by_choice=np.array(["bc", "ab"]),
        items_features_by_choice_names=["abc"],
        choices=choices,
        features_by_ids=[
            FeaturesStorage(
                ids=["ab", "bc"],
                values=np.array(
                    [[[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]], [[4.4, 3.3, 5.5], [2.2, 1.1, 6.6]]],
                    dtype=np.float32,
                ),
                name="abc",
            )
        ],
    )
    assert (
        dataset.batch[0][1] == np.array([[4.4, 3.3, 5.5], [2.2, 1.1, 6.6]], dtype=np.float32)
    ).all()
    assert dataset.batch[[0, 1]][1].shape == (2, 2, 3)


def test_cd_indexer_w_fs():
    """Various indexations of a choice dataset with FeaturesStorage usage."""
    shared_features = np.array([[0.0, 1.0], [2.0, 4.0]])
    items_features = np.array([[[0.5, 0.4], [0.2, 0.3]], [[0.4, 0.5], [0.3, 0.2]]])
    choices = [0, 1]
    # Test FeaturesStorage & features conjoint use
    dataset = ChoiceDataset(
        shared_features_by_choice=shared_features,
        items_features_by_choice=[
            [[0.0, 3.3, 99.0], [1.0, 3.3, 100.0]],
            [[2.0, 2.2, 101.0], [3.0, 3.3, 102.0]],
        ],
        items_features_by_choice_names=["aaa", "abc", "ddd"],
        choices=choices,
        features_by_ids=[
            FeaturesStorage(
                ids=[2.2, 3.3],
                values=np.array([[4.4, 3.3, 5.5], [2.2, 1.1, 6.6]], dtype=np.float32),
                name="abc",
            )
        ],
    )
    assert (
        dataset.batch[[1, 0]][1]
        == np.array(
            [
                [[2.0, 4.4, 3.3, 5.5, 101.0], [3.0, 2.2, 1.1, 6.6, 102.0]],
                [[0.0, 2.2, 1.1, 6.6, 99.0], [1.0, 2.2, 1.1, 6.6, 100.0]],
            ],
            dtype=np.float32,
        )
    ).all()
    full_dataset = dataset.indexer.get_full_dataset()
    assert (
        full_dataset[1]
        == np.array(
            [
                [[0.0, 2.2, 1.1, 6.6, 99.0], [1.0, 2.2, 1.1, 6.6, 100.0]],
                [[2.0, 4.4, 3.3, 5.5, 101.0], [3.0, 2.2, 1.1, 6.6, 102.0]],
            ],
            dtype=np.float32,
        )
    ).all()

    # Test FeaturesStorage & features conjoint use
    dataset = ChoiceDataset(
        shared_features_by_choice=shared_features,
        items_features_by_choice=[[[0.0, 3.3], [1.0, 3.3]], [[2.0, 2.2], [3.0, 3.3]]],
        items_features_by_choice_names=["aaa", "abc"],
        choices=choices,
        features_by_ids=[
            FeaturesStorage(
                ids=[2.2, 3.3],
                values=np.array([[4.4, 3.3, 5.5], [2.2, 1.1, 6.6]], dtype=np.float32),
                name="abc",
            )
        ],
    )
    assert (
        dataset.batch[[0, 1]][1]
        == np.array(
            [
                [[0.0, 2.2, 1.1, 6.6], [1.0, 2.2, 1.1, 6.6]],
                [[2.0, 4.4, 3.3, 5.5], [3.0, 2.2, 1.1, 6.6]],
            ],
            dtype=np.float32,
        )
    ).all()

    with pytest.raises(NotImplementedError):
        dataset.batch["abc"]

    # Test FeaturesStorage & features conjoint use
    dataset = ChoiceDataset(
        shared_features_by_choice=[[0.0, 3.3, 99.0], [1.0, 2.2, 100.0]],
        items_features_by_choice=items_features,
        shared_features_by_choice_names=["aaa", "abc", "ddd"],
        choices=choices,
        features_by_ids=[
            FeaturesStorage(
                ids=[2.2, 3.3],
                values=np.array([[4.4, 3.3, 5.5], [2.2, 1.1, 6.6]], dtype=np.float32),
                name="abc",
            )
        ],
    )
    full_dataset = dataset.indexer.get_full_dataset()
    assert (
        full_dataset[0]
        == np.array([[0.0, 2.2, 1.1, 6.6, 99.0], [1.0, 4.4, 3.3, 5.5, 100.0]], dtype=np.float32)
    ).all()


def test_batch():
    """Test specific usecase of batching that was failing."""
    storage = OneHotStorage(ids=[0, 1, 2, 3], name="id")
    shared_storage = OneHotStorage(ids=[0, 1, 2, 3], name="shared_id")

    items_features = np.array(
        [
            [
                [2, 2, 2, 2],
                [2, 2, 2, 3],
            ],
            [
                [2, 2, 3, 2],
                [3, 2, 2, 2],
            ],
            [[3, 2, 2, 2], [2, 3, 2, 2]],
        ]
    )

    items_features_ids = np.array(
        [
            [[0], [1]],
            [[3], [2]],
            [[0], [1]],
        ]
    )

    shared_features = np.array([[2, 1], [3, 4], [9, 4]])
    shared_features_ids = np.array([[0], [1], [2]])

    choices = np.array([0, 1, 1])

    dataset = ChoiceDataset(
        shared_features_by_choice=(shared_features, shared_features_ids),
        shared_features_by_choice_names=(["shared_a", "shared_b"], ["shared_id"]),
        items_features_by_choice=(items_features, items_features_ids),
        items_features_by_choice_names=(["a", "b", "c", "d"], ["id"]),
        choices=choices,
        features_by_ids=[storage, shared_storage],
    )

    batch = dataset.get_choices_batch(0)
    assert (batch[0][0] == np.array([2, 1])).all()
    assert (batch[0][1] == np.array([1, 0, 0, 0])).all()

    assert (batch[1][0] == np.array([[2, 2, 2, 2], [2, 2, 2, 3]])).all()
    assert (batch[1][1] == np.array([[1, 0, 0, 0], [0, 1, 0, 0]])).all()

    assert (batch[2] == np.array([1.0, 1.0])).all()
    assert batch[3] == 0

    batch = dataset.batch[0]
    assert (batch[0][0] == np.array([2, 1])).all()
    assert (batch[0][1] == np.array([1, 0, 0, 0])).all()

    assert (batch[1][0] == np.array([[2, 2, 2, 2], [2, 2, 2, 3]])).all()
    assert (batch[1][1] == np.array([[1, 0, 0, 0], [0, 1, 0, 0]])).all()

    assert (batch[2] == np.array([1.0, 1.0])).all()
    assert batch[3] == 0

    batch = dataset.get_choices_batch([1, 2])
    assert (batch[0][0] == np.array([[3, 4], [9, 4]])).all()
    assert (batch[0][1] == np.array([[0, 1, 0, 0], [0, 0, 1, 0]])).all()

    assert (
        batch[1][0] == np.array([[[2, 2, 3, 2], [3, 2, 2, 2]], [[3, 2, 2, 2], [2, 3, 2, 2]]])
    ).all()
    assert (
        batch[1][1] == np.array([[[0, 0, 0, 1], [0, 0, 1, 0]], [[1, 0, 0, 0], [0, 1, 0, 0]]])
    ).all()

    assert (batch[2] == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
    assert (batch[3] == np.array([1, 1])).all()

    batch = dataset.batch[[1, 2]]
    assert (batch[0][0] == np.array([[3, 4], [9, 4]])).all()
    assert (batch[0][1] == np.array([[0, 1, 0, 0], [0, 0, 1, 0]])).all()

    assert (
        batch[1][0] == np.array([[[2, 2, 3, 2], [3, 2, 2, 2]], [[3, 2, 2, 2], [2, 3, 2, 2]]])
    ).all()
    assert (
        batch[1][1] == np.array([[[0, 0, 0, 1], [0, 0, 1, 0]], [[1, 0, 0, 0], [0, 1, 0, 0]]])
    ).all()

    assert (batch[2] == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
    assert (batch[3] == np.array([1, 1])).all()


def test_batch_2():
    """Test specific usecase of batching that was failing."""
    storage = OneHotStorage(ids=[0, 1, 2, 3], name="id")
    shared_storage = OneHotStorage(ids=[0, 1, 2, 3], name="shared_id")
    mixed_shared_storage = FeaturesStorage(
        ids=[0, 1, 2, 3], values=[[10], [20], [30], [40]], name="mixed_shared_id"
    )

    items_features = np.array(
        [
            [
                [2, 2, 2, 2],
                [2, 2, 2, 3],
            ],
            [
                [2, 2, 3, 2],
                [3, 2, 2, 2],
            ],
            [[3, 2, 2, 2], [2, 3, 2, 2]],
        ]
    )

    items_features_ids = np.array(
        [
            [[0], [1]],
            [[3], [2]],
            [[0], [1]],
        ]
    )

    shared_features = np.array([[2, 3, 1], [3, 2, 4], [9, 1, 4]])
    shared_features_ids = np.array([[0], [1], [2]])

    choices = np.array([0, 1, 1])

    dataset = ChoiceDataset(
        shared_features_by_choice=(shared_features, shared_features_ids),
        shared_features_by_choice_names=(
            ["shared_a", "mixed_shared_id", "shared_b"],
            ["shared_id"],
        ),
        items_features_by_choice=(items_features, items_features_ids),
        items_features_by_choice_names=(["a", "b", "c", "d"], ["id"]),
        choices=choices,
        features_by_ids=[storage, shared_storage, mixed_shared_storage],
    )

    batch = dataset.get_choices_batch(0)
    assert (batch[0][0] == np.array([2, 40, 1])).all()
    assert (batch[0][1] == np.array([1, 0, 0, 0])).all()

    assert (batch[1][0] == np.array([[2, 2, 2, 2], [2, 2, 2, 3]])).all()
    assert (batch[1][1] == np.array([[1, 0, 0, 0], [0, 1, 0, 0]])).all()

    assert (batch[2] == np.array([1.0, 1.0])).all()
    assert batch[3] == 0

    batch = dataset.batch[0]
    assert (batch[0][0] == np.array([2, 40, 1])).all()
    assert (batch[0][1] == np.array([1, 0, 0, 0])).all()

    assert (batch[1][0] == np.array([[2, 2, 2, 2], [2, 2, 2, 3]])).all()
    assert (batch[1][1] == np.array([[1, 0, 0, 0], [0, 1, 0, 0]])).all()

    assert (batch[2] == np.array([1.0, 1.0])).all()
    assert batch[3] == 0

    batch = dataset.get_choices_batch([1, 2])
    assert (batch[0][0] == np.array([[3, 30, 4], [9, 20, 4]])).all()
    assert (batch[0][1] == np.array([[0, 1, 0, 0], [0, 0, 1, 0]])).all()

    assert (
        batch[1][0] == np.array([[[2, 2, 3, 2], [3, 2, 2, 2]], [[3, 2, 2, 2], [2, 3, 2, 2]]])
    ).all()
    assert (
        batch[1][1] == np.array([[[0, 0, 0, 1], [0, 0, 1, 0]], [[1, 0, 0, 0], [0, 1, 0, 0]]])
    ).all()

    assert (batch[2] == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
    assert (batch[3] == np.array([1, 1])).all()

    batch = dataset.batch[[1, 2]]
    assert (batch[0][0] == np.array([[3, 30, 4], [9, 20, 4]])).all()
    assert (batch[0][1] == np.array([[0, 1, 0, 0], [0, 0, 1, 0]])).all()

    assert (
        batch[1][0] == np.array([[[2, 2, 3, 2], [3, 2, 2, 2]], [[3, 2, 2, 2], [2, 3, 2, 2]]])
    ).all()
    assert (
        batch[1][1] == np.array([[[0, 0, 0, 1], [0, 0, 1, 0]], [[1, 0, 0, 0], [0, 1, 0, 0]]])
    ).all()

    assert (batch[2] == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
    assert (batch[3] == np.array([1, 1])).all()
