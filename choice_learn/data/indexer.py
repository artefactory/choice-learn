"""Indexer classes for data classes."""
from abc import abstractmethod

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
        storage : choice_modeling.data.store.FeaturesStorage
            Storage object to be indexed.
        """
        self.storage = storage

    def __getitem__(self, sequence_keys):
        """Returns the features appearing at the sequence_index-th position of sequence.

        Parameters
        ----------
        sequence_keys : (int, list, slice)
            keys of values to be retrieved

        Returns:
        --------
        array_like
            features corresponding to the sequence_keys
        """
        if isinstance(sequence_keys, list) or isinstance(sequence_keys, np.ndarray):
            if len(np.array(sequence_keys).shape) > 1:
                return np.stack([self.storage.batch[key] for key in sequence_keys], axis=0)
            return np.array([self.storage.storage[key] for key in sequence_keys])

        if isinstance(sequence_keys, slice):
            raise ValueError("Slicing is not supported for storage")
        return np.array(self.storage.storage[sequence_keys])


class OneHotStorageIndexer(Indexer):
    """Class for Ilocing OneHotStorage."""

    def __init__(self, storage):
        """OneHotStorageIndexer constructor.

        Parameters
        ----------
        storage : choice_modeling.data.store.OneHotStorage
            OneHotStorage object to be indexed.
        """
        self.storage = storage
        self.shape = storage.shape
        self.dtype = storage.dtype

    def __getitem__(self, sequence_keys):
        """Get the 1 indexes corresponding to the sequence_keys and builds the OneHot matrix.

        Parameters
        ----------
        sequence_keys : (int, list, slice)
            keys of values to be retrieved

        Returns:
        --------
        np.ndarray
            OneHot reconstructed vectors corresponding to sequence_keys
        """
        if isinstance(sequence_keys, list):
            # Construction of the OneHot vector from the index of the 1 value
            one_hot = np.zeros((len(sequence_keys), self.shape[1]))
            for i, j in enumerate(sequence_keys):
                one_hot[i, self.storage.storage[j]] = 1
            return one_hot.astype(self.dtype)
        if isinstance(sequence_keys, slice):
            return self[list(range(*sequence_keys.indices(len(self.shape[0]))))]
        # else:
        one_hot = np.zeros(self.shape[1])
        one_hot[self.storage.storage[sequence_keys]] = 1
        return one_hot.astype(self.dtype)


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
            items_features = list(
                items_feature.astype(self.choice_dataset._return_types[0][i])
                for i, items_feature in enumerate(self.choice_dataset.fixed_items_features)
            )
        return items_features

    def _get_contexts_features(self, choices_indexes):
        """Method to access sessions features of the ChoiceDataset.

        Parameters
        ----------
        choices_indexes : list of ints or int
            choices indexes of the contexts features to return

        Returns:
        --------
        tuple of np.ndarray or np.ndarray
            right indexed contexts_features of the ChoiceDataset
        """
        if self.choice_dataset.contexts_features is None:
            contexts_features = None
        else:
            contexts_features = []
            for i, contexts_feature in enumerate(self.choice_dataset.contexts_features):
                if hasattr(contexts_feature, "batch"):
                    contexts_features.append(
                        contexts_feature.batch[choices_indexes].astype(
                            self.choice_dataset._return_types[1][i]
                        )
                    )
                else:
                    contexts_features.append(np.stack(contexts_feature[choices_indexes], axis=0))
        return contexts_features

    def _get_contexts_items_features(self, choices_indexes):
        """Method to access sessions items features of the ChoiceDataset.

        Parameters
        ----------
        choices_indexes : list of ints or int
            indexes of the choices for which we want the contexts items features

        Returns:
        --------
        tuple of np.ndarray or np.ndarray
            right indexes contexts_items_features of the ChoiceDataset
        """
        if self.choice_dataset.contexts_items_features is None:
            return None
        contexts_items_features = []
        for i, contexts_items_feature in enumerate(self.choice_dataset.contexts_items_features):
            if hasattr(contexts_items_feature, "iloc"):
                contexts_items_features.append(contexts_items_feature.iloc[choices_indexes])
            else:
                contexts_items_features.append(
                    np.stack(contexts_items_feature[choices_indexes], axis=0)
                )
        return contexts_items_features

    def __getitem__(self, choices_indexes):
        """Method to access data within the ChoiceDataset from its index.

        One index corresponds to a choice within a session.
        Return order:
            - Fixed item features
            - Contexts features
            - Contexts item features
            - Items availabilities
            - Choices

        Parameters
        ----------
        choices_indexes : int or list of int or slice
            indexes of the choices (that will be mapped to choice & session indexes) to return

        """
        if isinstance(choices_indexes, list):
            # Get the features
            fixed_items_features = self._get_fixed_items_features()
            contexts_features = self._get_contexts_features(choices_indexes)
            contexts_items_features = self._get_contexts_items_features(choices_indexes)

            # Get availabilities
            if self.choice_dataset.contexts_items_availabilities is None:
                contexts_items_availabilities = np.ones(
                    (len(choices_indexes), self.choice_dataset.base_num_items)
                )
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

            if len(self.choice_dataset.fixed_items_features_map) > 0:
                mapped_features = []
                for tuple_index in np.sort(
                    list(self.choice_dataset.fixed_items_features_map.keys())
                ):
                    feat_ind_min = 0
                    unstacked_feat = []
                    for feature_index in np.sort(
                        list(self.choice_dataset.fixed_items_features_map[tuple_index].keys())
                    ):
                        unstacked_feat.append(
                            fixed_items_features[tuple_index][:, feat_ind_min:feature_index]
                        )
                        unstacked_feat.append(
                            self.choice_dataset.fixed_items_features_map[tuple_index][
                                feature_index
                            ].batch[fixed_items_features[tuple_index][:, feature_index]]
                        )
                        feat_ind_min = feature_index + 1
                    mapped_features.append(np.concatenate(unstacked_feat, axis=1))

                fixed_items_features = mapped_features

            if len(self.choice_dataset.contexts_features_map) > 0:
                mapped_features = []
                for tuple_index in np.sort(list(self.choice_dataset.contexts_features_map.keys())):
                    feat_ind_min = 0
                    unstacked_feat = []
                    for feature_index in np.sort(
                        list(self.choice_dataset.contexts_features_map[tuple_index].keys())
                    ):
                        unstacked_feat.append(
                            contexts_features[tuple_index][:, feat_ind_min:feature_index]
                        )
                        unstacked_feat.append(
                            self.choice_dataset.contexts_features_map[tuple_index][
                                feature_index
                            ].batch[contexts_features[tuple_index][:, feature_index]]
                        )
                        feat_ind_min = feature_index + 1
                    mapped_features.append(np.concatenate(unstacked_feat, axis=1))

                contexts_features = mapped_features

            if len(self.choice_dataset.contexts_items_features_map) > 0:
                mapped_features = []
                for tuple_index in np.sort(
                    list(self.choice_dataset.contexts_items_features_map.keys())
                ):
                    feat_ind_min = 0
                    unstacked_feat = []
                    for feature_index in np.sort(
                        list(self.choice_dataset.contexts_items_features_map[tuple_index].keys())
                    ):
                        unstacked_feat.append(
                            contexts_items_features[tuple_index][:, :, feat_ind_min:feature_index]
                        )
                        unstacked_feat.append(
                            self.choice_dataset.contexts_items_features_map[tuple_index][
                                feature_index
                            ].batch[contexts_items_features[tuple_index][:, :, feature_index]]
                        )
                        feat_ind_min = feature_index + 1
                    mapped_features.append(np.concatenate(unstacked_feat, axis=2))

                contexts_items_features = mapped_features

            # Shaping and typing
            if fixed_items_features is not None:
                for i in range(len(fixed_items_features)):
                    fixed_items_features[i] = fixed_items_features[i].astype(
                        self.choice_dataset._return_types[0][i]
                    )
                # items_features were not given as a tuple, so we return do not return it as a tuple
                if not self.choice_dataset._return_items_features_tuple:
                    fixed_items_features = fixed_items_features[0]
                else:
                    fixed_items_features = tuple(fixed_items_features)

            if contexts_features is not None:
                for i in range(len(contexts_features)):
                    contexts_features[i] = contexts_features[i].astype(
                        self.choice_dataset._return_types[1][i]
                    )
                if not self.choice_dataset._return_contexts_features_tuple:
                    contexts_features = contexts_features[0]
                else:
                    contexts_features = tuple(contexts_features)

            if contexts_items_features is not None:
                for i in range(len(contexts_items_features)):
                    contexts_items_features[i] = contexts_items_features[i].astype(
                        self.choice_dataset._return_types[2][i]
                    )
                # sessions_items_features were not given as a tuple, so we return do not return
                # it as a tuple
                if not self.choice_dataset._return_contexts_items_features_tuple:
                    contexts_items_features = contexts_items_features[0]
                else:
                    contexts_items_features = tuple(contexts_items_features)

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
            # Get the features
            fixed_items_features = self._get_fixed_items_features()
            contexts_features = self._get_contexts_features(choices_indexes)
            contexts_items_features = self._get_contexts_items_features(choices_indexes)

            choice = self.choice_dataset.choices[choices_indexes]

            if self.choice_dataset.contexts_items_availabilities is None:
                contexts_items_availabilities = np.ones((self.choice_dataset.base_num_items))
            else:
                contexts_items_availabilities = self.choice_dataset.contexts_items_availabilities[
                    choices_indexes
                ]

            if len(self.choice_dataset.fixed_items_features_map) > 0:
                mapped_features = []
                for tuple_index in np.sort(
                    list(self.choice_dataset.fixed_items_features_map.keys())
                ):
                    feat_ind_min = 0
                    unstacked_feat = []
                    for feature_index in np.sort(
                        list(self.choice_dataset.fixed_items_features_map[tuple_index].keys())
                    ):
                        unstacked_feat.append(
                            fixed_items_features[tuple_index][:, feat_ind_min:feature_index]
                        )
                        unstacked_feat.append(
                            self.choice_dataset.fixed_items_features_map[tuple_index][
                                feature_index
                            ].batch[fixed_items_features[tuple_index][:, feature_index]]
                        )
                        feat_ind_min = feature_index + 1
                    mapped_features.append(np.concatenate(unstacked_feat, axis=1))

                fixed_items_features = mapped_features

            if len(self.choice_dataset.contexts_features_map) > 0:
                mapped_features = []
                for tuple_index in np.sort(list(self.choice_dataset.contexts_features_map.keys())):
                    feat_ind_min = 0
                    unstacked_feat = []
                    for feature_index in np.sort(
                        list(self.choice_dataset.contexts_features_map[tuple_index].keys())
                    ):
                        unstacked_feat.append(
                            contexts_features[tuple_index][feat_ind_min:feature_index]
                        )
                        unstacked_feat.append(
                            self.choice_dataset.contexts_features_map[tuple_index][
                                feature_index
                            ].batch[contexts_features[tuple_index][feature_index]]
                        )
                        feat_ind_min = feature_index + 1
                    mapped_features.append(np.concatenate(unstacked_feat, axis=0))

                contexts_features = mapped_features

            if len(self.choice_dataset.contexts_items_features_map) > 0:
                mapped_features = []
                for tuple_index in np.sort(
                    list(self.choice_dataset.contexts_items_features_map.keys())
                ):
                    feat_ind_min = 0
                    unstacked_feat = []
                    for feature_index in np.sort(
                        list(self.choice_dataset.contexts_items_features_map[tuple_index].keys())
                    ):
                        unstacked_feat.append(
                            contexts_items_features[tuple_index][:, feat_ind_min:feature_index]
                        )
                        unstacked_feat.append(
                            self.choice_dataset.contexts_items_features_map[tuple_index][
                                feature_index
                            ].batch[contexts_items_features[tuple_index][:, feature_index]]
                        )
                        feat_ind_min = feature_index + 1
                    mapped_features.append(np.concatenate(unstacked_feat, axis=1))

                contexts_items_features = mapped_features

            if fixed_items_features is not None:
                for i in range(len(fixed_items_features)):
                    fixed_items_features[i] = fixed_items_features[i].astype(
                        self.choice_dataset._return_types[0][i]
                    )
                # items_features were not given as a tuple, so we return do not return it as a tuple
                if not self.choice_dataset._return_items_features_tuple:
                    fixed_items_features = fixed_items_features[0]
                else:
                    fixed_items_features = tuple(fixed_items_features)

            if contexts_features is not None:
                for i in range(len(contexts_features)):
                    contexts_features[i] = contexts_features[i].astype(
                        self.choice_dataset._return_types[1][i]
                    )
                if not self.choice_dataset._return_contexts_features_tuple:
                    contexts_features = contexts_features[0]
                else:
                    contexts_features = tuple(contexts_features)

            if contexts_items_features is not None:
                for i in range(len(contexts_items_features)):
                    contexts_items_features[i] = contexts_items_features[i].astype(
                        self.choice_dataset._return_types[2][i]
                    )
                # sessions_items_features were not given as a tuple, so we return do not return
                # it as a tuple
                if not self.choice_dataset._return_contexts_items_features_tuple:
                    contexts_items_features = contexts_items_features[0]
                else:
                    contexts_items_features = tuple(contexts_items_features)

            return (
                fixed_items_features,
                contexts_features,
                contexts_items_features,
                contexts_items_availabilities,
                choice,
            )
        print(f"Type{type(choices_indexes)} not handled")
        raise NotImplementedError
