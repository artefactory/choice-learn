"""Test the store module."""
from choice_learn.data.store import FeaturesStore, OneHotStore, Store


def test_len_store():
    """Test the __len__ method of Store."""
    store = Store(values=[1, 2, 3, 4], sequence=[0, 1, 2, 3, 0, 1, 2, 3])
    assert len(store) == 8


def test_get_store_element():
    """Test the _get_store_element method of Store."""
    store = Store(values=[1, 2, 3, 4], sequence=[0, 1, 2, 3, 0, 1, 2, 3])
    assert store._get_store_element(0) == 1
    assert store._get_store_element([0, 1, 2]) == [1, 2, 3]


def test_store_batch():
    """Test the batch method of Store."""
    store = Store(values=[1, 2, 3, 4], sequence=[0, 1, 2, 3, 0, 1, 2, 3])
    assert store.batch[1] == 2
    assert store.batch[2:4] == [3, 4]
    assert store.batch[[2, 3, 6, 7]] == [3, 4, 3, 4]


def test_featuresstore_instantiation():
    """Test the instantiation of FeaturesStore."""
    store = FeaturesStore(
        values=[[10, 10], [4, 4], [2, 2], [8, 8]],
        sequence=[0, 1, 2, 3, 0, 1, 2, 3],
        indexes=[0, 1, 2, 3],
    )
    assert store.shape == (8, 2)
    assert [store.sequence[i] == [0, 1, 2, 3, 0, 1, 2, 3][i] for i in range(8)]
    assert store.store == {0: [10, 10], 1: [4, 4], 2: [2, 2], 3: [8, 8]}


def test_featuresstore_instantiation_indexless():
    """Test the instantiation of FeaturesStore."""
    store = FeaturesStore(
        values=[[10, 10], [4, 4], [2, 2], [8, 8]], sequence=[0, 1, 2, 3, 0, 1, 2, 3]
    )
    assert store.shape == (8, 2)
    assert [store.sequence[i] == [0, 1, 2, 3, 0, 1, 2, 3][i] for i in range(8)]
    assert store.store == {0: [10, 10], 1: [4, 4], 2: [2, 2], 3: [8, 8]}


def test_featuresstore_instantiation_from_list():
    """Test the instantiation of FeaturesStore."""
    store = FeaturesStore.from_list(
        values_list=[[10, 10], [4, 4], [2, 2], [8, 8]], sequence=[0, 1, 2, 3, 0, 1, 2, 3]
    )
    assert store.shape == (8, 2)
    assert [store.sequence[i] == [0, 1, 2, 3, 0, 1, 2, 3][i] for i in range(8)]
    assert store.store == {0: [10, 10], 1: [4, 4], 2: [2, 2], 3: [8, 8]}


def test_featuresstore_instantiation_fromdict():
    """Test the instantiation of FeaturesStore."""
    store = FeaturesStore.from_dict(
        values_dict={0: [10, 10], 1: [4, 4], 2: [2, 2], 3: [8, 8]},
        sequence=[0, 1, 2, 3, 0, 1, 2, 3],
    )
    assert store.shape == (8, 2)
    assert [store.sequence[i] == [0, 1, 2, 3, 0, 1, 2, 3][i] for i in range(8)]
    assert store.store == {0: [10, 10], 1: [4, 4], 2: [2, 2], 3: [8, 8]}


def test_featuresstore_getitem():
    """Test the __getitem__ method of FeaturesStore."""
    store = FeaturesStore.from_dict(
        values_dict={0: [10, 10], 1: [4, 4], 2: [2, 2], 3: [8, 8]},
        sequence=[0, 1, 2, 3, 0, 1, 2, 3],
    )
    sub_store = store[0:3]
    assert sub_store.shape == (3, 2)
    assert [sub_store.sequence[i] == [0, 1, 2][i] for i in range(3)]
    assert sub_store.store == {0: [10, 10], 1: [4, 4], 2: [2, 2]}


def test_onehotstore_instantiation():
    """Test the instantiation of OneHotStore."""
    indexes = [0, 1, 2, 4]
    values = [0, 1, 2, 3]
    sequence = [0, 1, 2, 4, 0, 1, 2, 4]
    store = OneHotStore(indexes=indexes, values=values, sequence=sequence)
    assert store.shape == (8, 4)
    assert [store.sequence[i] == [0, 1, 2, 4, 0, 1, 2, 4][i] for i in range(8)]
    assert store.store == {0: 0, 1: 1, 2: 2, 4: 3}


def test_onehotstore_instantiation_from_sequence():
    """Test the instantiation; from_sequence of OneHotStore."""
    sequence = [0, 1, 2, 3, 0, 1, 2, 3]
    store = OneHotStore.from_sequence(sequence=sequence)
    assert store.shape == (8, 4)
    assert [store.sequence[i] == [0, 1, 2, 3, 0, 1, 2, 3][i] for i in range(8)]
    assert store.store == {0: 0, 1: 1, 2: 2, 3: 3}


def test_onehotstore_getitem():
    """Test the getitem of OneHotStore."""
    indexes = [0, 1, 2, 4]
    values = [0, 1, 2, 3]
    sequence = [0, 1, 2, 4, 0, 1, 2, 4]
    store = OneHotStore(indexes=indexes, values=values, sequence=sequence)
    sub_store = store[0:3]
    assert sub_store.shape == (3, 3)
    assert [
        sub_store.sequence[i] == [0, 1, 2, 3, 0, 1, 2, 3][i] for i in range(len(sub_store.sequence))
    ]
    assert sub_store.store == {
        0: 0,
        1: 1,
        2: 2,
    }


def test_onehotstore_batch():
    """Test the getitem of OneHotStore."""
    indexes = [0, 1, 2, 4]
    values = [0, 1, 2, 3]
    sequence = [0, 1, 2, 4, 0, 1, 2, 4]
    store = OneHotStore(indexes=indexes, values=values, sequence=sequence)

    batch = store.batch[0]
    assert (batch == [1, 0, 0, 0]).all()

    batch = store.batch[0:4]
    assert (batch == [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).all()

    batch = store.batch[[3, 6, 7]]
    assert (batch == [[0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]).all()
