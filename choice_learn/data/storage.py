"""Different classes to optimize RAM usage with repeated features over time."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from choice_learn.data.indexer import ArrayStorageIndexer, OneHotStorageIndexer, StorageIndexer


class Storage(ABC):
    """Parent Class to have OneHotStorage and FeaturesStorage with same parent."""

    def __init__(self, features_to_store):
        """Instantiate the storage.

        Parameters
        ----------
        features_to_store : object
            Object to store
        """
        self.features_to_store = features_to_store

    @abstractmethod
    def __getitem__(self, keys):
        """Access an element. To be implemented in children classes.

        Parameters
        ----------
        keys : float, int, str or list of
            values among indexes of the stiage
        """
        pass

    @abstractmethod
    def get_element_from_index(self, index):
        """Getter method with index (int).

        Returns the features stored witg the index-th ID.

        Parameters
        ----------
        index : (int, list, slice)
            index argument of the feature

        Returns
        -------
        array_like
            features corresponding to the index index in self.storage
        """
        pass

    @abstractmethod
    def __len__(self):
        """Return the length of the sequence of apparition of the features."""
        pass

    @property
    def batch(self):
        """Indexing method."""
        pass

    def __str__(self):
        """Return string representation method.

        Returns
        -------
        str
            Description of the storage.
        """
        return f"FeatureStorage with name {self.name}"


class FeaturesStorage(object):
    """Base FeaturesStorage class that redirects toward the right one."""

    def __new__(
        cls,
        ids=None,
        values=None,
        values_names=None,
        name=None,
        as_one_hot=False,
    ):
        """Redirects toward the right object.

        Parameters
        ----------
        ids : Iterable, optional
            IDs to references features with, by default None
        values : Iterable, optional
            Features to be stored, by default None
        values_names : list, optional
            List of names for the features to be stored, by default None
        name : str, optional
            Name of the FeaturesStorage, to be matched in ChoiceDataset, by default None
        as_one_hot: bool
            Whether features are OneHot representations or not.

        Returns
        -------
        FeaturesStorage
            One of ArrayStorage, DictStorage or OneHotStorage
        """
        if as_one_hot:
            return OneHotStorage(ids=ids, values=values, name=name)
        if ids is None and (isinstance(values, np.ndarray) or isinstance(values, list)):
            return ArrayStorage(values=values, values_names=values_names, name=name)

        if ids is not None:
            check_ids = np.in1d(ids, np.arange(len(ids))).all()
            if check_ids:
                values = np.array(values)[np.argsort(ids)]
                return ArrayStorage(values=values, values_names=values_names, name=name)

        return DictStorage(
            ids=ids, values=values, values_names=values_names, name=name, indexer=StorageIndexer
        )


class DictStorage(Storage):
    """Function to store features with ids."""

    def __init__(
        self,
        ids=None,
        values=None,
        values_names=None,
        name=None,
        indexer=StorageIndexer,
    ):
        """Build the store.

        Parameters
        ----------
        ids : array_like or None
            list of ids of features to store. If None is given, ids are created from
            apparition order of values
        values : array_like
            list of values of features to store
        values_names : array_like
            Iterable of str indicating the name of the features. Must be same length as values.
        name: string, optional
            name of the features store
        """
        if isinstance(values, dict):
            storage = values
            lengths = []
            for k, v in storage.items():
                if not isinstance(v, np.ndarray) | isinstance(v, list):
                    raise ValueError("values must be a dict of np.ndarray or list")
                if not len(np.array(v).shape) == 1:
                    raise ValueError(
                        "values (features) must be a dict of np.ndarray or list of 1D arrays"
                    )
                lengths.append(len(v))
                if isinstance(v, list):
                    storage[k] = np.array(v)
            if not len(set(lengths)) == 1:
                raise ValueError("values (dict values) must all have same length")
            if ids is not None:
                print("Warning: ids is ignored when values is a dict")

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
            storage = {k: np.array(v) for (k, v) in zip(ids, values)}
        else:
            raise ValueError("values must be a dict, a DataFrame, a list or a numpy array")

        self.storage = storage
        self.values_names = values_names
        self.name = name

        self.shape = (len(self), len(next(iter(self.storage.values()))))
        self.indexer = indexer(self)

    def get_element_from_index(self, index):
        """Getter method with index (int).

        Returns the features stored witg the index-th ID.

        Parameters
        ----------
        index : (int, list, slice)
            index argument of the feature

        Returns
        -------
        array_like
            features corresponding to the index index in self.storage
        """
        if isinstance(index, int):
            index = [index]
        keys = [list(self.storage.keys())[i] for i in index]
        return self.batch[keys]

    def __len__(self):
        """Return the length of the sequence of apparition of the features."""
        return len(self.storage)

    def __getitem__(self, id_keys):
        """Subset FeaturesStorage, keeping only features which id is in keys.

        Parameters
        ----------
        id_keys : Iterable
            List of ids to keep.

        Returns
        -------
        FeaturesStorage
            Subset of the FeaturesStorage, with only the features whose id is in id_keys
        """
        if not isinstance(id_keys, list):
            id_keys = [id_keys]
        sub_storage = {k: self.storage[k] for k in id_keys}
        return FeaturesStorage(values=sub_storage, values_names=self.values_names, name=self.name)

    def get_storage_type(self):
        """Functions to access stored elements dtypes.

        Returns
        -------
        tuple
            tuple of dtypes of the stored elements, as returned by np.dtype
        """
        element = self.get_element_from_index(0)
        return element.dtype

    @property
    def batch(self):
        """Indexing attribute."""
        return self.indexer


class ArrayStorage(Storage):
    """Function to store features with ids as NumPy Array."""

    def __init__(self, values=None, values_names=None, name=None):
        """Build the store.

        Parameters
        ----------
        values : array_like
            list of values of features to store
        values_names : array_like
            Iterable of str indicating the name of the features. Must be same length as values.
        name: string, optional
            name of the features store
        """
        if isinstance(values, list):
            values = np.array(values)
        elif not isinstance(values, np.ndarray):
            raise ValueError("ArrayStorage Values must be a list or a numpy array")
        # self.storage = storage
        self.values_names = values_names
        self.name = name

        self.shape = values.shape
        self.storage = values
        self.indexer = ArrayStorageIndexer(self)

    def get_element_from_index(self, index):
        """Getter method with index (int).

        Returns the features stored witg the index-th ID.

        Parameters
        ----------
        index : (int, list, slice)
            index argument of the feature

        Returns
        -------
        array_like
            features corresponding to the index index in self.storage
        """
        return self.batch[index]

    def __len__(self):
        """Return the length of the sequence of apparition of the features."""
        return self.shape[0]

    def __getitem__(self, id_keys):
        """Subset FeaturesStorage, keeping only features which id is in keys.

        Parameters
        ----------
        id_keys : Iterable
            List of ids to keep.

        Returns
        -------
        FeaturesStorage
            Subset of the FeaturesStorage, with only the features whose id is in id_keys
        """
        if not isinstance(id_keys, list):
            id_keys = [id_keys]
        return ArrayStorage(
            values=self.batch[id_keys], values_names=self.values_names, name=self.name
        )

    def get_storage_type(self):
        """Functions to access stored elements dtypes.

        Returns
        -------
        tuple
            tuple of dtypes of the stored elements, as returned by np.dtype
        """
        element = self.get_element_from_index(0)
        return element.dtype

    @property
    def batch(self):
        """Indexing attribute."""
        return self.indexer


class OneHotStorage(Storage):
    """Specific Storage for one hot features storage.

    Inherits from Storage.
    For example can be used to store a OneHot representation of the days of week.

    Has the same attributes as FeaturesStoage, only differs whit some One-Hot optimized methods.
    It only stores the indexes of the features, and creates the OneHot matrix
    when needed, using .batch[].
    """

    def __init__(
        self, ids=None, values=None, name=None, dtype=np.uint8, indexer=OneHotStorageIndexer
    ):
        """Build the store.

        Parameters
        ----------
        ids : array_like or None
            list of ids of features to store. If None is given, ids are created from
            apparition order of values
        values : array_like
            list of values of features to store
        dtype: type
            type for One Hot representation, usually int or float, default is np.uint8
        name: string, optional
            name of the features store
        """
        if isinstance(values, dict):
            storage = values
            for k, v in storage.items():
                if not isinstance(v, int):
                    raise ValueError(
                        """values of values dict must be int as
                        they are indexes of the one hot vector ones."""
                    )
            length = np.max(list(storage.values())) + 1
            if ids is not None:
                print("Warning: ids is ignored when values is a dict")

        elif isinstance(values, list) or isinstance(values, np.ndarray):
            if ids is None:
                ids = list(range(len(values)))
            storage = {k: int(v) for (k, v) in zip(ids, values)}
            length = np.max(values) + 1

        elif values is None:
            if ids is None:
                raise ValueError("ids or values must be given, both are None")
            value = 0
            storage = {}
            for id in ids:
                storage[id] = value
                value += 1
            length = value
        else:
            raise ValueError("values must be a dict, a DataFrame, a list or a numpy array")

        self.storage = storage
        self.name = name

        self.shape = (len(self), length)
        self.dtype = dtype
        self.indexer = indexer(self)

    def __len__(self):
        """Return the length of the sequence of apparition of the features."""
        return len(self.storage)

    def __getitem__(self, id_keys):
        """Subset FeaturesStorage, keeping only features which id is in keys.

        Parameters
        ----------
        id_keys : Iterable
            List of ids to keep.

        Returns
        -------
        OneHotStorage
            Subset of the OneHotStorage, with only the features whose id is in id_keys
        """
        if isinstance(id_keys, int):
            id_keys = [id_keys]
        sub_storage = {k: self.storage[k] for k in id_keys}

        return OneHotStorage(values=sub_storage, name=self.name, dtype=self.dtype)

    def astype(self, dtype):
        """Change (mainly int or float) type of returned OneHot features vectors.

        Parameters
        ----------
        dtype : type
            Type to set the features as
        """
        self.dtype = dtype

    def get_element_from_index(self, index):
        """Getter method with index (int).

        Returns the features stored witg the index-th ID.

        Parameters
        ----------
        index : (int, list, slice)
            index argument of the feature

        Returns
        -------
        array_like
            features corresponding to the index index in self.storage
        """
        keys = list(self.storage.keys())[index]
        return self.storage[keys]

    def get_storage_type(self):
        """Functions to access stored elements dtypes.

        Returns
        -------
        type
            tuple of dtypes of the stored elements, as returned by np.dtype
        """
        return self.dtype

    @property
    def batch(self):
        """Indexing attribute."""
        return self.indexer
