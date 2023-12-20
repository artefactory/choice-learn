from abc import abstractmethod

import numpy as np
import pandas as pd


class Indexer(object):
    def __init__(self, indexed_object):
        self.indexed_object = indexed_object

    @abstractmethod
    def __getitem__(self, index):
        pass


class StoreIndexer(Indexer):
    """Class for Ilocing FeaturesStore

    Parameters
    ----------
    TBD
    """

    def __init__(self, store):
        self.store = store

    def __getitem__(self, sequence_index):
        """
        Returns the features corresponding appearing at the sequence_index-th position of sequence

        Parameters
        ----------
        sequence_index : (int, list, slice)
            index position of the sequence

        Returns
        -------
        array_like
            features corresponding to the sequence_index-th position of sequence
        """
        if isinstance(sequence_index, list):
            return [self.store.store[self.store.sequence[i]] for i in sequence_index]
        elif isinstance(sequence_index, slice):
            return [
                self.store.store[self.store.sequence[i]]
                for i in range(*sequence_index.indices(len(self.sequence)))
            ]
        return self.store.store[self.store.sequence[sequence_index]]


class OneHotStoreIndexer(Indexer):
    """Class for Ilocing OneHotStore

    Parameters
    ----------
    TBD
    """

    def __init__(self, store):
        self.store = store

        self.shape = (len(self.store.sequence), np.max(list(self.store.store.values())) + 1)

    def __getitem__(self, sequence_index):
        """Main method to get an element at sequence_index-th position of self.sequence.

        Parameters
        ----------
        sequence_index : (int, list, slice)
            index from sequence of element to get

        Returns
        -------
        np.ndarray
            OneHot features corresponding to the sequence_index-th position of sequence
        """
        if isinstance(sequence_index, list):
            # Construction of the OneHot vector from the index of the 1 value
            one_hot = np.zeros((len(sequence_index), self.shape[1]))
            for i, j in enumerate(sequence_index):
                one_hot[i, self.store.store[self.store.sequence[j]]] = 1
            return one_hot.astype(self.store.dtype)
        else:
            one_hot = np.zeros(self.shape[1])
            one_hot[self.store.store[self.store.sequence[sequence_index]]] = 1
            return one_hot.astype(self.store.dtype)


class ChoiceDatasetIndexer(Indexer):
    """Indexing class for ChoiceDataset

    Parameters
    ----------
    object : _type_
        _description_
    """

    def __init__(self, choice_dataset):
        self.choice_dataset = choice_dataset

    def _get_items_features(self):
        if self.choice_dataset.items_features is None:
            items_features = None
        else:
            items_features = tuple(
                items_feature.astype(self.choice_dataset._return_types[0][i])
                for i, items_feature in enumerate(self.choice_dataset.items_features)
            )
            # items_features were not given as a tuple, so we return do not return it as a tuple
            if not self.choice_dataset._return_items_features_tuple:
                items_features = items_features[0]

        return items_features

    def _get_sessions_features(self, sessions_indexes):
        if self.choice_dataset.sessions_features is None:
            sessions_features = None
        else:
            sessions_features = []
            for i, sessions_feature in enumerate(self.choice_dataset.sessions_features):
                if hasattr(sessions_feature, "iloc"):
                    sessions_features.append(
                        sessions_feature.iloc[sessions_indexes].astype(
                            self.choice_dataset._return_types[1][i]
                        )
                    )
                else:
                    sessions_features.append(
                        np.stack(sessions_feature[sessions_indexes], axis=0).astype(
                            self.choice_dataset._return_types[1][i]
                        )
                    )
            # sessions_features were not given as a tuple, so we return do not return it as a tuple
            if not self.choice_dataset._return_sessions_features_tuple:
                sessions_features = sessions_features[0]
            else:
                sessions_features = tuple(sessions_features)
        return sessions_features

    def _get_sessions_items_features(self, sessions_indexes):
        if self.choice_dataset.sessions_items_features is None:
            sessions_items_features = None
        else:
            sessions_items_features = []
            for i, sessions_items_feature in enumerate(self.choice_dataset.sessions_items_features):
                if hasattr(sessions_items_feature, "iloc"):
                    sessions_items_features.append(
                        sessions_items_feature.iloc[sessions_indexes].astype(
                            self._return_types[2][i]
                        )
                    )
                else:
                    sessions_items_features.append(
                        np.stack(sessions_items_feature[sessions_indexes], axis=0).astype(
                            self.choice_dataset._return_types[2][i]
                        )
                    )
            # sessions_items_features were not given as a tuple, so we return do not return it as a tuple
            if self.choice_dataset._return_sessions_items_features_tuple:
                sessions_items_features = tuple(sessions_items_features)
            else:
                sessions_items_features = sessions_items_features[0]
            return sessions_items_features

    def __getitem__(self, choice_index):
        """
        Method to access data within the ChoiceDataset from its index.
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
        if isinstance(choice_index, list):
            items_features = self._get_items_features()
            # Get the session indexes
            sessions_indexes = [self.choice_dataset.indexes[i] for i in choice_index]

            sessions_features = self._get_sessions_features(sessions_indexes)
            sessions_items_features = self._get_sessions_items_features(sessions_indexes)

            if self.choice_dataset.sessions_items_availabilities is None:
                sessions_items_availabilities = None
            else:
                if hasattr(self.choice_dataset.sessions_items_availabilities, "iloc"):
                    sessions_items_availabilities = (
                        self.choice_dataset.sessions_items_availabilities.iloc[
                            sessions_indexes
                        ].astype(self.choice_dataset._return_types[3])
                    )
                else:
                    sessions_items_availabilities = (
                        self.choice_dataset.sessions_items_availabilities[sessions_indexes].astype(
                            self.choice_dataset._return_types[3]
                        )
                    )

            choice = self.choice_dataset.choices[choice_index].astype(
                self.choice_dataset._return_types[4]
            )

            return (
                items_features,
                sessions_features,
                sessions_items_features,
                sessions_items_availabilities,
                choice,
            )

        elif isinstance(choice_index, slice):
            return self.__getitem__(list(range(*choice_index.indices(self.choices.shape[0]))))

        elif isinstance(choice_index, int):
            items_features = self._get_items_features()
            # Get the session indexes
            sessions_indexes = self.choice_dataset.indexes[choice_index]

            sessions_features = self._get_sessions_features(sessions_indexes)
            sessions_items_features = self._get_sessions_items_features(sessions_indexes)

            if self.choice_dataset.sessions_items_availabilities is None:
                sessions_items_availabilities = None
            else:
                if hasattr(self.choice_dataset.sessions_items_availabilities, "iloc"):
                    sessions_items_availabilities = (
                        self.choice_dataset.sessions_items_availabilities.iloc[
                            sessions_indexes
                        ].astype(self.choice_dataset._return_types[3])
                    )
                else:
                    sessions_items_availabilities = (
                        self.choice_dataset.sessions_items_availabilities[sessions_indexes].astype(
                            self.choice_dataset._return_types[3]
                        )
                    )

            choice = self.choice_dataset.choices[choice_index].astype(
                self.choice_dataset._return_types[4]
            )

            return (
                items_features,
                sessions_features,
                sessions_items_features,
                sessions_items_availabilities,
                choice,
            )
        else:
            raise NotImplementedError
