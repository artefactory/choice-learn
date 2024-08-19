"""Test Indexer + ChoiceDataset + Storage."""

import numpy as np

from choice_learn.data import ChoiceDataset, FeaturesStorage, OneHotStorage 


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
