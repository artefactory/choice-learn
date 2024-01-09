"""Different classes to optimize RAM usage with repeated features over time."""
import numpy as np
import panda as pd


class Storage(object):
    """Class to keep OneHotStore and FeaturesStore with same parent."""

    def __init__(self, ids=None, values=None, values_names=None, name=None):
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
            pass
        elif isinstance(values, pd.DataFrame):
            if values_names is not None:
                print("Warning: values_names is ignored when values is a DataFrame")
            if "id" in values.columns:
                values = values.set_index("id")
            values_names = values.columns
            storage = {k: v.to_numpy() for (k, v) in values.iterrows()}
        elif isinstance(values, list):
            pass
        elif isinstance(values, np.ndarray):
            pass
        else:
            raise ValueError("values must be a dict, a DataFrame, a list or a numpy array")

        if ids is None:
            ids = list(range(len(values)))

        self.storage = {k: v for (k, v) in zip(ids, values)}
        self.name = name

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
        return len(self.sequence)

    @property
    def batch(self):
        """Indexing attribute."""
        return self.indexer
