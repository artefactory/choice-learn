"""Different classes to optimize RAM usage with repeated features over time."""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from choice_learn.data.indexer import StorageIndexer

class Storage(ABC):
    """Class to keep OneHotStore and FeaturesStore with same parent."""

    def __init__(self, features_to_store):
        self.features_to_store = features_to_store

    @abstractmethod
    def __getitem__(self, keys):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @property
    def batch(self):
        pass


class FeaturesStorage(Storage):
    """Class to keep OneHotStore and FeaturesStore with same parent."""

    def __init__(self, ids=None, values=None, values_names=None, name=None, indexer=StorageIndexer):
        """Builds the store.

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
        if isinstance(values, dict):
            storage = values
            lengths = []
            for k, v in storage.items():
                assert (isinstance(v, np.ndarray) | isinstance(v, list))
                assert len(np.array(v).shape) == 1
                lengths.append(len(v))
            assert len(set(lengths)) == 1

        elif isinstance(values, pd.DataFrame):
            if values_names is not None:
                print("Warning: values_names is ignored when values is a DataFrame")
            if "id" in values.columns:
                values = values.set_index("id")
            values_names = values.columns
            storage = {k: v.to_numpy() for (k, v) in values.iterrows()}
        elif isinstance(values, list) or isinstance(values, np.ndarray):
            if ids is None:
                ids = list(range(len(values)))
            storage = {k: v for (k, v) in zip(ids, values)}
        else:
            raise ValueError("values must be a dict, a DataFrame, a list or a numpy array")

        self.storage = storage
        self.values_names = values_names
        self.name = name

        self.shape = (len(self), len(next(iter(self.storage.values()))))
        self.indexer = indexer(self)

    def _get_store_element(self, index):
        """Getter method over self.sequence.

        Returns the features stored at index index. Compared to __getitem__, it does take
        the index-th element of sequence but the index-th element of the store.

        Parameters
        ----------
        index : (int, list, slice)
            index argument of the feature

        Returns:
        --------
        array_like
            features corresponding to the index index in self.store
        """
        if isinstance(index, list):
            return [self.store[i] for i in index]
        # else:
        return self.store[index]

    def __len__(self):
        """Returns the length of the sequence of apparition of the features."""
        return len(self.storage)

    def __getitem__(self, keys):
        """_summary_.

        Parameters
        ----------
        keys : _type_
            _description_
        """
        sub_storage = {k: self.storage[k] for k in keys}
        return FeaturesStorage(values=sub_storage, values_names=self.values_names, name=self.name)

    @property
    def batch(self):
        """Indexing attribute."""
        return self.indexer
