"""Different classes to optimize RAM usage with repeated features over time."""
import numpy as np

from choice_learn.data.indexer import OneHotStoreIndexer, StoreIndexer


class Store(object):
    """Class to keep OneHotStore and FeaturesStore with same parent."""

    def __init__(self, indexes=None, values=None, sequence=None, name=None, indexer=StoreIndexer):
        """Build the store.

        Parameters
        ----------
        indexes : array_like or None
            list of indexes of features to store. If None is given, indexes are created from
            apparition order of values
        values : array_like
            list of values of features to store
        sequence : array_like
            sequence of apparitions of the features
        name: string, optional
            name of the features store -- not used at the moment
        """
        if indexes is None:
            indexes = list(range(len(values)))
        self.store = {k: v for (k, v) in zip(indexes, values)}
        self.sequence = np.array(sequence)
        self.name = name

        if sequence is not None and values is not None:
            try:
                width = len(values[0])
            except TypeError:
                width = 1
            self.shape = (len(sequence), width)

        self.indexer = indexer(self)

    def _get_store_element(self, index):
        """Getter method over self.sequence.

        Returns the features stored at index index. Compared to __getitem__, it does take
        the index-th element of sequence but the index-th element of the store.

        Parameters
        ----------
        index : (int, list, slice)
            index argument of the feature

        Returns
        -------
        array_like
            features corresponding to the index index in self.store
        """
        if isinstance(index, list):
            return [self.store[i] for i in index]
        # else:
        return self.store[index]

    def __len__(self):
        """Return the length of the sequence of apparition of the features."""
        return len(self.sequence)

    @property
    def batch(self):
        """Indexing attribute."""
        return self.indexer


class FeaturesStore(Store):
    """Base class to store features and a sequence of apparitions.

    Mainly useful when features are repeated frequently over the sequence.
    An example would be to store the features of a customers (supposing that the same customers come
    several times over the work sequence) and to save which customer is concerned for each choice.

    Attributes
    ----------
    store : dict
        Dictionary stocking features that can be called from indexes: {index: features}
    shape : tuple
        shape of the features store: (sequence_length, features_number)
    sequence : array_like
        List of elements of indexes representing the sequence of apparitions of the features
    name: string, optional
        name of the features store -- not used at the moment
    dtype: type
        type of the features
    """

    @classmethod
    def from_dict(cls, values_dict, sequence):
        """Instantiate the FeaturesStore from a dictionary of values.

        Parameters
        ----------
        values_dict : dict
            dictionary of values to store, {index: value}
        sequence : array_like
            sequence of apparitions of the features

        Returns
        -------
        FeaturesStore created from the values in the dictionnary
        """
        # Check uniform shape of values
        return cls(
            indexes=list(values_dict.keys()), values=list(values_dict.values()), sequence=sequence
        )

    @classmethod
    def from_list(cls, values_list, sequence):
        """Instantiate the FeaturesStore from a list of values.

        Creates indexes for each value

        Parameters
        ----------
        values_list : list
            List of values to store
        sequence : array_like
            sequence of apparitions of the features

        Returns
        -------
        FeaturesStore
        """
        # Check uniform shape of list
        # Useful ? To rethink...
        return cls(indexes=list(range(len(values_list))), values=values_list, sequence=sequence)

    def __getitem__(self, sequence_index):
        """Subsets self with sequence_index.

        Parameters
        ----------
        sequence_index : (int, list, slice)
            index position of the sequence

        Returns
        -------
        array_like
            features corresponding to the sequence_index-th position of sequence
        """
        if isinstance(sequence_index, int):
            sequence_index = [sequence_index]
        new_sequence = self.sequence[sequence_index]
        store = {}
        for k, v in self.store.items():
            if k in new_sequence:
                store[k] = v
            else:
                print(f"Key {k} of store with value {v} not in sequence anymore")

        return FeaturesStore.from_dict(store, new_sequence)

    def astype(self, dtype):
        """Change the dtype of the features.

        The type of the features should implement the astype method.
        Typically, should work like np.ndarrays.

        Parameters
        ----------
        dtype : str or type
            type to set the features as
        """
        for k, v in self.store.items():
            self.store[k] = v.astype(dtype)


class OneHotStore(Store):
    """Specific FeaturesStore for one hot features storage.

    Inherits from FeaturesStore.
    For example can be used to store a OneHot representation of the days of week.

    Has the same attributes as FeaturesStore, only differs whit some One-Hot optimized methods.
    """

    def __init__(
        self,
        indexes=None,
        values=None,
        sequence=None,
        name=None,
        dtype=np.float32,
    ):
        """Build the OneHot features store.

        Parameters
        ----------
        indexes : array_like or None
            list of indexes of features to store. If None is given, indexes are created from
            apparition order of values
        values : array_like or None
            list of values of features to store that must be One-Hot. If None given they are created
            from order of apparition in sequence
        sequence : array_like
            sequence of apparitions of the features
        name: string, optional
            name of the features store -- not used at the moment
        """
        self.name = name
        self.sequence = np.array(sequence)

        if values is None:
            self = self.from_sequence(sequence)
        else:
            self.store = {k: v for (k, v) in zip(indexes, values)}
            self.shape = (len(sequence), np.max(values) + 1)

        self.dtype = dtype
        self.indexer = OneHotStoreIndexer(self)

    @classmethod
    def from_sequence(cls, sequence):
        """Create a OneHotFeatureStore from a sequence of apparition.

        One Hot vector are created from the order of apparition in the sequence: feature vectors
        created have a length of the number of different values in the sequence and the 1 is
        positioned in order of first appartitions in the sequence.

        Parameters
        ----------
        sequence : array-like
            Sequence of apparitions of values, or indexes. Will be used to index self.store

        Returns
        -------
        FeatureStore
            Created from the sequence.
        """
        all_indexes = np.unique(sequence)
        values = np.arange(len(all_indexes))
        return cls(indexes=all_indexes, values=values, sequence=sequence)

    def __getitem__(self, sequence_index):
        """Get an element at sequence_index-th position of self.sequence.

        Parameters
        ----------
        sequence_index : (int, list, slice)
            index from sequence of element to get

        Returns
        -------
        np.ndarray
            OneHot features corresponding to the sequence_index-th position of sequence
        """
        if isinstance(sequence_index, int):
            sequence_index = [sequence_index]
        new_sequence = self.sequence[sequence_index]
        store = {}
        for k, v in self.store.items():
            if k in new_sequence:
                store[k] = v
            else:
                print(f"Key {k} of store with value {v} not in sequence anymore")

        return OneHotStore(
            indexes=list(store.keys()), values=list(store.values()), sequence=new_sequence
        )

    def astype(self, dtype):
        """Change (mainly int or float) type of returned OneHot features vectors.

        Parameters
        ----------
        dtype : type
            Type to set the features as
        """
        self.dtype = dtype
