"""Indexer classes for data classes."""
from abc import abstractmethod
from collections.abc import Iterable

import numpy as np


class Indexer(object):
    """Base class for Indexer."""

    def __init__(self, indexed_object):
        """Instanciate an Indexer object.

        Parameters
        ----------
        indexed_object : object
            object to be indexed.
        """
        self.indexed_object = indexed_object

    @abstractmethod
    def __getitem__(self, index):
        """Main method to be coded for children classes.

        Parameters
        ----------
        index : int, slice, list
            index(es) of elements of self.indexed_object to be returned.
        """
        pass


class StoreIndexer(Indexer):
    """Class for Ilocing/Batching FeaturesStore."""

    def __init__(self, store):
        """StoreIndexer constructor.

        Parameters
        ----------
        store : choice_modeling.data.store.FeaturesStore
            Store object to be indexed.
        """
        self.store = store

    def __getitem__(self, sequence_index):
        """Returns the features appearing at the sequence_index-th position of sequence.

        Parameters
        ----------
        sequence_index : (int, list, slice)
            index position of the sequence

        Returns:
        --------
        array_like
            features corresponding to the sequence_index-th position of sequence
        """
        if isinstance(sequence_index, list):
            return [self.store.store[self.store.sequence[i]] for i in sequence_index]
        if isinstance(sequence_index, slice):
            return [
                self.store.store[self.store.sequence[i]]
                for i in range(*sequence_index.indices(len(self.store.sequence)))
            ]
        return self.store.store[self.store.sequence[sequence_index]]


class StorageIndexer(Indexer):
    """Class for Ilocing/Batching FeaturesStorage."""

    def __init__(self, storage):
        """StoreIndexer constructor.

        Parameters
        ----------
        store : choice_modeling.data.store.FeaturesStore
            Store object to be indexed.
        """
        self.storage = storage

    def __getitem__(self, sequence_keys):
        """Returns the features appearing at the sequence_index-th position of sequence.

        Parameters
        ----------
        sequence_index : (int, list, slice)
            index position of the sequence

        Returns:
        --------
        array_like
            features corresponding to the sequence_index-th position of sequence
        """
        if isinstance(sequence_keys, Iterable):
            return np.array([self.storage.storage[key] for key in sequence_keys])
        if isinstance(sequence_keys, slice):
            raise ValueError("Slicing is not supported for storage")
        return np.array(self.storage.storage[sequence_keys])


class OneHotStoreIndexer(Indexer):
    """Class for Ilocing OneHotStore."""

    def __init__(self, store):
        """OneHotStoreIndexer constructor.

        Parameters
        ----------
        store : choice_modeling.data.store.OneHotStore
            OneHotStore object to be indexed.
        """
        self.store = store

        self.shape = (len(self.store.sequence), np.max(list(self.store.store.values())) + 1)

    def __getitem__(self, sequence_index):
        """Main method to get an element at sequence_index-th position of self.sequence.

        Parameters
        ----------
        sequence_index : (int, list, slice)
            index from sequence of element to get

        Returns:
        --------
        np.ndarray
            OneHot features corresponding to the sequence_index-th position of sequence
        """
        if isinstance(sequence_index, list):
            # Construction of the OneHot vector from the index of the 1 value
            one_hot = np.zeros((len(sequence_index), self.shape[1]))
            for i, j in enumerate(sequence_index):
                one_hot[i, self.store.store[self.store.sequence[j]]] = 1
            return one_hot.astype(self.store.dtype)
        if isinstance(sequence_index, slice):
            return self[list(range(*sequence_index.indices(len(self.store.sequence))))]
        # else:
        one_hot = np.zeros(self.shape[1])
        one_hot[self.store.store[self.store.sequence[sequence_index]]] = 1
        return one_hot.astype(self.store.dtype)


class ChoiceDatasetIndexer(Indexer):
    """Indexing class for ChoiceDataset."""

    def __init__(self, choice_dataset):
        """Instanciate a ChoiceDatasetIndexer object.

        Parameters
        ----------
        choice_dataset : choce_modeling.data.dataset.ChoiceDataset
            Dataset to be indexed.
        """
        self.choice_dataset = choice_dataset

    def _get_fixed_items_features(self):
        """Method to access items features of the ChoiceDataset.

        Returns:
        --------
        tuple of np.ndarray or np.ndarray
            items_features of the ChoiceDataset
        """
        if self.choice_dataset.fixed_items_features is None:
            items_features = None
        else:
            items_features = tuple(
                items_feature.astype(self.choice_dataset._return_types[0][i])
                for i, items_feature in enumerate(self.choice_dataset.fixed_items_features)
            )
            # items_features were not given as a tuple, so we return do not return it as a tuple
            # if not self.choice_dataset._return_items_features_tuple:
            #     items_features = items_features[0]

        return items_features

    def _get_contexts_features(self, contexts_indexes):
        """Method to access sessions features of the ChoiceDataset.

        Parameters
        ----------
        sessions_indexes : list of ints or int
            indexes of the sessions to return

        Returns:
        --------
        tuple of np.ndarray or np.ndarray
            items_features of the ChoiceDataset
        """
        if self.choice_dataset.contexts_features is None:
            contexts_features = None
        else:
            contexts_features = []
            for i, contexts_feature in enumerate(self.choice_dataset.contexts_features):
                if hasattr(contexts_feature, "batch"):
                    contexts_features.append(
                        contexts_feature.batch[contexts_indexes].astype(
                            self.choice_dataset._return_types[1][i]
                        )
                    )
                else:
                    contexts_features.append(
                        np.stack(contexts_feature[contexts_indexes], axis=0).astype(
                            self.choice_dataset._return_types[1][i]
                        )
                    )
            # sessions_features were not given as a tuple, so we return do not return it as a tuple
            # if not self.choice_dataset._return_contexts_features_tuple:
            #     contexts_features = contexts_feature[0]
            # else:
            #     contexts_features = tuple(contexts_features)
        return contexts_features

    def _get_contexts_items_features(self, contexts_indexes):
        """Method to access sessions items features of the ChoiceDataset.

        Parameters
        ----------
        sessions_indexes : list of ints or int
            indexes of the sessions to return

        Returns:
        --------
        tuple of np.ndarray or np.ndarray
            items_features of the ChoiceDataset
        """
        if self.choice_dataset.contexts_items_features is None:
            return None
        contexts_items_features = []
        for i, contexts_items_feature in enumerate(self.choice_dataset.contexts_items_features):
            if hasattr(contexts_items_feature, "iloc"):
                contexts_items_features.append(
                    contexts_items_feature.iloc[contexts_indexes].astype(self._return_types[2][i])
                )
            else:
                contexts_items_features.append(
                    np.stack(contexts_items_feature[contexts_indexes], axis=0).astype(
                        self.choice_dataset._return_types[2][i]
                    )
                )
        # sessions_items_features were not given as a tuple, thus we do not return it as a tuple
        # if self.choice_dataset._return_contexts_items_features_tuple:
        #     contexts_items_features = tuple(contexts_items_features)
        # else:
        #     contexts_items_features = contexts_items_features[0]
        return contexts_items_features

    def __getitem__(self, choices_indexes):
        """Method to access data within the ChoiceDataset from its index.

        One index corresponds to a choice within a session.
        Return order:
            - Fixed item features
            - Session features
            - Session item features
            - Items availabilities
            - Choice

        Parameters
        ----------
        index : int or list of int or slice
            indexes of the choices (that will be mapped to choice & session indexes) to return

        """
        if isinstance(choices_indexes, list):
            fixed_items_features = self._get_fixed_items_features()

            # Get the session indexes
            contexts_features = self._get_contexts_features(choices_indexes)
            contexts_items_features = self._get_contexts_items_features(choices_indexes)

            if self.choice_dataset.contexts_items_availabilities is None:
                contexts_items_availabilities = None
            else:
                if hasattr(self.choice_dataset.contexts_items_availabilities, "batch"):
                    contexts_items_availabilities = (
                        self.choice_dataset.contexts_items_availabilities.batch[
                            choices_indexes
                        ].astype(self.choice_dataset._return_types[3])
                    )
                else:
                    contexts_items_availabilities = (
                        self.choice_dataset.contexts_items_availabilities[choices_indexes].astype(
                            self.choice_dataset._return_types[3]
                        )
                    )

            for indexes, func in self.choice_dataset.fixed_items_features_map:
                fixed_items_features[indexes[0]][:, indexes[1] : indexes[1] + 1] = func[
                    fixed_items_features[indexes[0]][:, indexes[1]]
                ]
            for indexes, func in self.choice_dataset.contexts_features_map:
                contexts_features[indexes[0]][:, indexes[1] : indexes[1] + 1] = func[
                    contexts_features[indexes[0]][:, indexes[1]]
                ]
            for indexes, func in self.choice_dataset.contexts_items_features_map:
                contexts_items_features[indexes[0]][:, :, indexes[1] : indexes[1] + 1] = func[
                    contexts_items_features[indexes[0]][:, :, indexes[1]]
                ]
            # items_features were not given as a tuple, so we return do not return it as a tuple
            if not self.choice_dataset._return_items_features_tuple:
                fixed_items_features = fixed_items_features[0]
            if not self.choice_dataset._return_contexts_features_tuple:
                contexts_features = contexts_features[0]
            # sessions_items_features were not given as a tuple, so we return do not return
            # it as a tuple
            if not self.choice_dataset._return_contexts_items_features_tuple:
                contexts_items_features = contexts_items_features[0]

            choices = self.choice_dataset.choices[choices_indexes].astype(
                self.choice_dataset._return_types[4]
            )

            return (
                fixed_items_features,
                contexts_features,
                contexts_items_features,
                contexts_items_availabilities,
                choices,
            )

        if isinstance(choices_indexes, slice):
            return self.__getitem__(
                list(range(*choices_indexes.indices(self.choice_dataset.choices.shape[0])))
            )

        if isinstance(choices_indexes, int):
            fixed_items_features = self._get_fixed_items_features()
            # Get the session indexes

            contexts_features = self._get_contexts_features(choices_indexes)
            contexts_items_features = self._get_contexts_items_features(choices_indexes)

            if self.choice_dataset.contexts_items_availabilities is None:
                contexts_items_availabilities = None
            else:
                if hasattr(self.choice_dataset.contexts_items_availabilities, "batch"):
                    contexts_items_availabilities = (
                        self.choice_dataset.contexts_items_availabilities.iloc[
                            choices_indexes
                        ].astype(self.choice_dataset._return_types[3])
                    )
                else:
                    contexts_items_availabilities = (
                        self.choice_dataset.contexts_items_availabilities[choices_indexes].astype(
                            self.choice_dataset._return_types[3]
                        )
                    )
            for indexes, func in self.choice_dataset.fixed_items_features_map:
                fixed_items_features[indexes[0]][:, indexes[1] : indexes[1] + 1] = func[
                    fixed_items_features[indexes[0]][:, indexes[1]]
                ]
            for indexes, func in self.choice_dataset.contexts_features_map:
                contexts_features[indexes[0]][indexes[1] : indexes[1] + 1] = func[
                    contexts_features[indexes[0]][indexes[1]]
                ]
            for indexes, func in self.choice_dataset.contexts_items_features_map:
                contexts_items_features[indexes[0]][:, indexes[1] : indexes[1] + 1] = func[
                    contexts_items_features[indexes[0]][:, indexes[1]]
                ]

            # items_features were not given as a tuple, so we return do not return it as a tuple
            if not self.choice_dataset._return_items_features_tuple:
                fixed_items_features = fixed_items_features[0]
            if not self.choice_dataset._return_contexts_features_tuple:
                contexts_features = contexts_features[0]
            # sessions_items_features were not given as a tuple, so we return do not return
            # it as a tuple
            if not self.choice_dataset._return_contexts_items_features_tuple:
                contexts_items_features = contexts_items_features[0]

            choice = self.choice_dataset.choices[choices_indexes].astype(
                self.choice_dataset._return_types[4]
            )

            return (
                fixed_items_features,
                contexts_features,
                contexts_items_features,
                contexts_items_availabilities,
                choice,
            )
        raise NotImplementedError
