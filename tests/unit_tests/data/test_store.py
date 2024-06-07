"""Test the store module."""
import numpy as np
import pandas as pd

from choice_learn.data.storage import FeaturesStorage, OneHotStorage


def test_len_store():
    """Test the __len__ method of Storage."""
    features = {"customerA": [1, 2], "customerB": [4, 5], "customerC": [7, 8]}
    storage = FeaturesStorage(
        values=features, values_names=["age", "income", "children_nb"], name="customers"
    )
    assert len(storage) == 3
    assert storage.shape == (3, 2)


def test_get_store_element():
    """Test the _get_store_element method of Store."""
    features = {"customerA": [1, 2], "customerB": [4, 5], "customerC": [7, 8]}
    storage = FeaturesStorage(
        values=features, values_names=["age", "income", "children_nb"], name="customers"
    )
    assert (storage.get_element_from_index(0) == np.array([1, 2])).all()
    assert (storage.get_element_from_index([0, 1, 2]) == np.array([[1, 2], [4, 5], [7, 8]])).all()


def test_store_batch():
    """Test the batch method of Store."""
    features = {"customerA": [1, 2], "customerB": [4, 5], "customerC": [7, 8]}
    storage = FeaturesStorage(
        values=features, values_names=["age", "income", "children_nb"], name="customers"
    )
    assert (storage.batch["customerA"] == np.array([1, 2])).all()
    assert (
        storage.batch[["customerA", "customerC", "customerA", "customerC"]]
        == np.array([[1, 2], [7, 8], [1, 2], [7, 8]])
    ).all()


def test_featuresstore_instantiation():
    """Test the instantiation of FeaturesStore."""
    features = {"customerA": [1, 2], "customerB": [4, 5], "customerC": [7, 8]}
    storage = FeaturesStorage(
        values=features, values_names=["age", "income", "children_nb"], name="customers"
    )

    for k, v in storage.storage.items():
        assert (
            v
            == {
                "customerA": np.array([1, 2]),
                "customerB": np.array([4, 5]),
                "customerC": np.array([7, 8]),
            }[k]
        ).all()


def test_featuresstore_instantiation_indexless():
    """Test the instantiation of FeaturesStore."""
    features = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    ids = ["customerA", "customerB", "customerC"]

    storage = FeaturesStorage(
        ids=ids, values=features, values_names=["age", "income", "children_nb"], name="customers"
    )
    assert storage.shape == (3, 3)
    for k, v in storage.storage.items():
        assert (
            v
            == {
                "customerA": np.array([1, 2, 3]),
                "customerB": np.array([4, 5, 6]),
                "customerC": np.array([7, 8, 9]),
            }[k]
        ).all()


def test_featuresstore_instantiation_from_list():
    """Test the instantiation of FeaturesStore."""
    features = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    storage = FeaturesStorage(
        values=features, values_names=["age", "income", "children_nb"], name="customers"
    )
    storage.batch[[0, 2, 0, 2]]
    assert storage.shape == (3, 3)
    assert (
        storage.batch[[0, 2, 0, 2]] == np.array([[1, 2, 3], [7, 8, 9], [1, 2, 3], [7, 8, 9]])
    ).all()


def test_array_store_with_ids():
    """Test the instantiation of FeaturesStore."""
    features = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    storage = FeaturesStorage(
        ids=[0, 1, 2],
        values=features,
        values_names=["age", "income", "children_nb"],
        name="customers",
    )
    storage.batch[[0, 2, 0, 2]]
    assert storage.shape == (3, 3)
    assert (
        storage.batch[[0, 2, 0, 2]] == np.array([[1, 2, 3], [7, 8, 9], [1, 2, 3], [7, 8, 9]])
    ).all()


def test_array_store_with_mixed_ids():
    """Test the instantiation of FeaturesStore."""
    features = [[1, 2, 3], [7, 8, 9], [4, 5, 6]]

    storage = FeaturesStorage(
        ids=[0, 2, 1],
        values=features,
        values_names=["age", "income", "children_nb"],
        name="customers",
    )
    storage.batch[[0, 2, 0, 2]]
    assert storage.shape == (3, 3)
    assert (
        storage.batch[[0, 2, 0, 2]] == np.array([[1, 2, 3], [7, 8, 9], [1, 2, 3], [7, 8, 9]])
    ).all()


def test_featuresstore_instantiation_fromdict():
    """Test the instantiation of FeaturesStore."""
    features = {
        "age": [1, 4, 7],
        "income": [2, 5, 8],
        "children_nb": [3, 6, 9],
        "id": ["customerA", "customerB", "customerC"],
    }
    features = pd.DataFrame(features)
    storage = FeaturesStorage(values=features, name="customers")
    assert storage.shape == (3, 3)
    for k, v in storage.storage.items():
        assert (
            v
            == {
                "customerA": np.array([1, 2, 3]),
                "customerB": np.array([4, 5, 6]),
                "customerC": np.array([7, 8, 9]),
            }[k]
        ).all()


def test_featuresstore_instantiation_fromdf():
    """Test the instantiation of FeaturesStore."""
    features = {"age": [1, 4, 7], "income": [2, 5, 8], "children_nb": [3, 6, 9]}
    features = pd.DataFrame(features, index=["customerA", "customerB", "customerC"])
    storage = FeaturesStorage(values=features, name="customers")
    assert storage.shape == (3, 3)
    for k, v in storage.storage.items():
        assert (
            v
            == {
                "customerA": np.array([1, 2, 3]),
                "customerB": np.array([4, 5, 6]),
                "customerC": np.array([7, 8, 9]),
            }[k]
        ).all()


def test_featuresstore_getitem():
    """Test the __getitem__ method of FeaturesStore."""
    features = {"customerA": [1, 2], "customerB": [4, 5], "customerC": [7, 8]}
    storage = FeaturesStorage(
        values=features, values_names=["age", "income", "children_nb"], name="customers"
    )
    sub_storage = storage[["customerA", "customerC"]]
    assert sub_storage.shape == (2, 2)
    for k, v in {"customerA": np.array([1, 2]), "customerC": np.array([7, 8])}.items():
        print(v, sub_storage.storage[k])
        assert (v == sub_storage.storage[k]).all()


def test_onehotstore_instantiation():
    """Test the instantiation of OneHotStore."""
    ids = [0, 1, 2, 3, 4]
    values = [4, 3, 2, 1, 0]
    storage = OneHotStorage(ids=ids, values=values, name="OneHotTest")
    assert storage.shape == (5, 5)
    assert storage.storage == {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}


def test_onehotstore_instantiation_from_sequence():
    """Test the instantiation; from_sequence of OneHotStore."""
    values = [4, 3, 2, 1, 0]
    storage = OneHotStorage(values=values, name="OneHotTest")
    assert (
        storage.batch[[0, 2, 4]] == np.array([[0, 0, 0, 0, 1], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0]])
    ).all()
    assert storage.storage == {4: 0, 3: 1, 2: 2, 1: 3, 0: 4}


def test_onehotstore_instantiation_from_ids():
    """Test the instantiation; from_sequence of OneHotStore."""
    ids = [0, 1, 2, 3, 4]
    storage = OneHotStorage(ids=ids, name="OneHotTest")
    assert (
        storage.batch[[0, 2, 4]] == np.array([[1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1]])
    ).all()
    assert storage.storage == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}


def test_onehotstore_instantiation_from_dict():
    """Test the instantiation; from_sequence of OneHotStore."""
    ids = [0, 1, 2, 3, 4]
    values = [4, 3, 2, 1, 0]
    values_dict = {k: v for k, v in zip(ids, values)}
    storage = OneHotStorage(values=values_dict, name="OneHotTest")
    assert (
        storage.batch[[0, 2, 4]] == np.array([[0, 0, 0, 0, 1], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0]])
    ).all()
    assert storage.storage == {4: 0, 3: 1, 2: 2, 1: 3, 0: 4}


def test_onehotstore_getitem():
    """Test the getitem of OneHotStore."""
    ids = [0, 1, 2, 3, 4]
    values = [4, 3, 2, 1, 0]
    storage = OneHotStorage(ids=ids, values=values, name="OneHotTest")
    assert (
        storage.batch[[0, 2, 4]] == np.array([[0, 0, 0, 0, 1], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0]])
    ).all()
    assert storage.get_element_from_index(0) == 4


def test_fail_instantiation():
    """Testing failed instantiation."""
    try:
        _ = OneHotStorage(name="OneHotTest")
        assert False
    except ValueError:
        assert True
