"""Indexer classes for data classes."""

import logging
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
        """To be coded for children classes.

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
        """Return the features appearing at the sequence_index-th position of sequence.

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
        """Return the features appearing at the sequence_index-th position of sequence.

        Parameters
        ----------
        sequence_keys : (int, list, slice)
            keys of values to be retrieved

        Returns
        -------
        array_like
            features corresponding to the sequence_keys
        """
        try:
            if isinstance(sequence_keys, list) or isinstance(sequence_keys, np.ndarray):
                if len(np.array(sequence_keys).shape) > 1:
                    return np.stack([self.storage.batch[key] for key in sequence_keys], axis=0)
                return np.array([self.storage.storage[key] for key in sequence_keys])

            if isinstance(sequence_keys, slice):
                raise ValueError("Slicing is not supported for storage")
            return np.array(self.storage.storage[sequence_keys])
        except KeyError as error:
            print("You are using an ID that is not in the storage:")
            print(error)
            raise


class ArrayStorageIndexer(StorageIndexer):
    """Class for Ilocing/Batching ArrayFeaturesStorage."""

    def __getitem__(self, sequence_keys):
        """Return the features appearing at the sequence_index-th position of sequence.

        Parameters
        ----------
        sequence_keys : (int, list, slice)
            keys of values to be retrieved

        Returns
        -------
        array_like
            features corresponding to the sequence_keys
        """
        try:
            return self.storage.storage[sequence_keys]
        except IndexError as error:
            print("You are using an ID that is not in the storage:")
            print(error)
            raise KeyError


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

        Returns
        -------
        np.ndarray
            OneHot reconstructed vectors corresponding to sequence_keys
        """
        if isinstance(sequence_keys, list) or isinstance(sequence_keys, np.ndarray):
            # Construction of the OneHot vector from the index of the 1 value

            if np.array(sequence_keys).ndim == 1:
                one_hot = []
                for j in sequence_keys:
                    # one_hot.append(self[j])
                    one_hot.append(self.storage.storage[j])
                matrix = np.zeros((len(one_hot), self.shape[1]))
                matrix[np.arange(len(one_hot)), one_hot] = 1
                return matrix.astype(self.dtype)
            one_hot = []
            for j in sequence_keys:
                one_hot.append(self[j])
            return np.stack(one_hot).astype(self.dtype)
        if isinstance(sequence_keys, slice):
            return self[list(range(*sequence_keys.indices(len(self.shape[0]))))]
        # else:
        one_hot = np.zeros(self.shape[1])
        try:
            one_hot[self.storage.storage[sequence_keys]] = 1
        except KeyError as error:
            print("You are using an ID that is not in the storage:")
            print(error)
            raise

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

    def _get_shared_features_by_choice(self, choices_indexes):
        """Access sessions features of the ChoiceDataset.

        Parameters
        ----------
        choices_indexes : list of ints or int
            choices indexes of the shared features to return

        Returns
        -------
        tuple of np.ndarray or np.ndarray
            right indexed contexts_fshared_features_by_choiceeatures of the ChoiceDataset
        """
        if self.choice_dataset.shared_features_by_choice is None:
            shared_features_by_choice = None
        else:
            shared_features_by_choice = []
            for i, shared_feature in enumerate(self.choice_dataset.shared_features_by_choice):
                if hasattr(shared_feature, "batch"):
                    shared_features_by_choice.append(shared_feature.batch[choices_indexes])
                else:
                    # shared_features_by_choice.append(
                    #     np.stack(shared_feature[choices_indexes], axis=0)
                    # )
                    shared_features_by_choice.append(shared_feature[choices_indexes])
        return shared_features_by_choice

    def _get_items_features_by_choice(self, choices_indexes):
        """Access sessions items features of the ChoiceDataset.

        Parameters
        ----------
        choices_indexes : list of ints or int
            indexes of the choices for which to select the items features

        Returns
        -------
        tuple of np.ndarray or np.ndarray
            right indexes items_features_by_choice of the ChoiceDataset
        """
        if self.choice_dataset.items_features_by_choice is None:
            return None
        items_features_by_choice = []
        for i, items_feature in enumerate(self.choice_dataset.items_features_by_choice):
            if hasattr(items_feature, "batch"):
                items_features_by_choice.append(items_feature.batch[choices_indexes])
            else:
                # items_features_by_choice.append(np.stack(items_feature[choices_indexes], axis=0))
                items_features_by_choice.append(items_feature[choices_indexes])
        return items_features_by_choice

    def __getitem__(self, choices_indexes):
        """Access data within the ChoiceDataset from its index.

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

        Returns
        -------
        np.ndarray
            shared_features at choices_indexes
        np.ndarray
            items_features at choices_indexes
        np.ndarray
            available_items_by_choice at choices_indexes
        np.ndarray
            choices at choices_indexes
        """
        if isinstance(choices_indexes, list):
            # Get the features
            shared_features_by_choice = self._get_shared_features_by_choice(choices_indexes)
            items_features_by_choice = self._get_items_features_by_choice(choices_indexes)

            if self.choice_dataset.available_items_by_choice is None:
                available_items_by_choice = np.ones(
                    (len(choices_indexes), self.choice_dataset.base_num_items)
                ).astype("float32")
            else:
                if isinstance(self.choice_dataset.available_items_by_choice, tuple):
                    available_items_by_choice = self.choice_dataset.available_items_by_choice[
                        0
                    ].batch[self.choice_dataset.available_items_by_choice[1][choices_indexes]]
                else:
                    available_items_by_choice = self.choice_dataset.available_items_by_choice[
                        choices_indexes
                    ]
                available_items_by_choice = available_items_by_choice.astype(
                    self.choice_dataset._return_types[2]
                )

            choices = self.choice_dataset.choices[choices_indexes].astype(
                self.choice_dataset._return_types[3]
            )

            ###
            if len(self.choice_dataset.shared_features_by_choice_map) > 0:
                mapped_features = []
                ###
                for tuple_index in range(len(shared_features_by_choice)):
                    if tuple_index in self.choice_dataset.shared_features_by_choice_map.keys():
                        feat_ind_min = 0
                        unstacked_feat = []
                        for feature_index in np.sort(
                            list(
                                self.choice_dataset.shared_features_by_choice_map[
                                    tuple_index
                                ].keys()
                            )
                        ):
                            if feat_ind_min != feature_index:
                                unstacked_feat.append(
                                    shared_features_by_choice[tuple_index][
                                        :, feat_ind_min:feature_index
                                    ]
                                )
                            unstacked_feat.append(
                                self.choice_dataset.shared_features_by_choice_map[tuple_index][
                                    feature_index
                                ].batch[shared_features_by_choice[tuple_index][:, feature_index]]
                            )
                            feat_ind_min = feature_index + 1
                        if feat_ind_min != shared_features_by_choice[tuple_index].shape[1]:
                            unstacked_feat.append(
                                shared_features_by_choice[tuple_index][:, feat_ind_min:]
                            )
                        mapped_features.append(np.hstack(unstacked_feat))
                    else:
                        mapped_features.append(shared_features_by_choice[tuple_index])

                shared_features_by_choice = mapped_features

            if len(self.choice_dataset.items_features_by_choice_map) > 0:
                mapped_features = []
                for tuple_index in range(len(items_features_by_choice)):
                    if tuple_index in self.choice_dataset.items_features_by_choice_map.keys():
                        if items_features_by_choice[tuple_index].ndim == 1:
                            mapped_features.append(
                                self.choice_dataset.items_features_by_choice_map[tuple_index][
                                    0
                                ].batch[items_features_by_choice[tuple_index]]
                            )
                        else:
                            feat_ind_min = 0
                            unstacked_feat = []
                            for feature_index in np.sort(
                                list(
                                    self.choice_dataset.items_features_by_choice_map[
                                        tuple_index
                                    ].keys()
                                )
                            ):
                                if feat_ind_min != feature_index:
                                    unstacked_feat.append(
                                        items_features_by_choice[tuple_index][
                                            :, :, feat_ind_min:feature_index
                                        ]
                                    )
                                unstacked_feat.append(
                                    self.choice_dataset.items_features_by_choice_map[tuple_index][
                                        feature_index
                                    ].batch[
                                        items_features_by_choice[tuple_index][:, :, feature_index]
                                    ]
                                )
                                feat_ind_min = feature_index + 1
                            if feat_ind_min != items_features_by_choice[tuple_index].shape[2]:
                                unstacked_feat.append(
                                    shared_features_by_choice[tuple_index][:, :, feat_ind_min:]
                                )
                            mapped_features.append(np.concatenate(unstacked_feat, axis=2))
                    else:
                        mapped_features.append(items_features_by_choice[tuple_index])

                items_features_by_choice = mapped_features

            if shared_features_by_choice is not None:
                for i in range(len(shared_features_by_choice)):
                    shared_features_by_choice[i] = shared_features_by_choice[i].astype(
                        self.choice_dataset._return_types[0][i]
                    )
                if not self.choice_dataset._return_shared_features_by_choice_tuple:
                    shared_features_by_choice = shared_features_by_choice[0]
                else:
                    shared_features_by_choice = tuple(shared_features_by_choice)

            if items_features_by_choice is not None:
                for i in range(len(items_features_by_choice)):
                    items_features_by_choice[i] = items_features_by_choice[i].astype(
                        self.choice_dataset._return_types[1][i]
                    )
                # items_features_by_choice were not given as a tuple, so we return do not return
                # it as a tuple
                if not self.choice_dataset._return_items_features_by_choice_tuple:
                    items_features_by_choice = items_features_by_choice[0]
                else:
                    items_features_by_choice = tuple(items_features_by_choice)
            return (
                shared_features_by_choice,
                items_features_by_choice,
                available_items_by_choice,
                choices,
            )

        if isinstance(choices_indexes, slice):
            return self.__getitem__(
                list(range(*choices_indexes.indices(self.choice_dataset.choices.shape[0])))
            )

        if isinstance(choices_indexes, int):
            choices_indexes = [choices_indexes]
            (
                shared_features_by_choices,
                items_features_by_choice,
                available_items_by_choice,
                choice,
            ) = self.__getitem__(choices_indexes)
            if shared_features_by_choices is not None:
                if isinstance(shared_features_by_choices, tuple):
                    shared_features_by_choices = tuple(
                        feat[0] for feat in shared_features_by_choices
                    )
                else:
                    shared_features_by_choices = shared_features_by_choices[0]
            if items_features_by_choice is not None:
                if isinstance(items_features_by_choice, tuple):
                    items_features_by_choice = tuple(feat[0] for feat in items_features_by_choice)
                else:
                    items_features_by_choice = items_features_by_choice[0]

            return (
                shared_features_by_choices,
                items_features_by_choice,
                available_items_by_choice[0],
                choice[0],
            )
        logging.error(f"Type{type(choices_indexes)} not handled")
        raise NotImplementedError(f"Type{type(choices_indexes)} not handled")

    def get_full_dataset(self):
        """Return the full dataset.

        This function is here to speed up iteration over dataset when batch_size
        is -1 or length of dataset.

        Returns
        -------
        np.ndarray
            all shared_features
        np.ndarray
            all items_features
        np.ndarray
            all available_items_by_choice
        np.ndarray
            all choices
        """
        if self.choice_dataset.shared_features_by_choice is not None:
            shared_features_by_choice = [
                feat for feat in self.choice_dataset.shared_features_by_choice
            ]
        else:
            shared_features_by_choice = None

        if self.choice_dataset.items_features_by_choice is not None:
            items_features_by_choice = [
                feat for feat in self.choice_dataset.items_features_by_choice
            ]
        else:
            items_features_by_choice = None

        if self.choice_dataset.available_items_by_choice is None:
            available_items_by_choice = np.ones(
                (len(self.choice_dataset), self.choice_dataset.base_num_items)
            ).astype("float32")
        else:
            if isinstance(self.choice_dataset.available_items_by_choice, tuple):
                available_items_by_choice = self.choice_dataset.available_items_by_choice[0].batch[
                    self.choice_dataset.available_items_by_choice[1]
                ]
            else:
                available_items_by_choice = self.choice_dataset.available_items_by_choice
        available_items_by_choice = available_items_by_choice.astype(
            self.choice_dataset._return_types[2]
        )

        choices = self.choice_dataset.choices.astype(self.choice_dataset._return_types[3])

        ###
        if len(self.choice_dataset.shared_features_by_choice_map) > 0:
            mapped_features = []
            ###
            for tuple_index in range(len(shared_features_by_choice)):
                if tuple_index in self.choice_dataset.shared_features_by_choice_map.keys():
                    feat_ind_min = 0
                    unstacked_feat = []
                    for feature_index in np.sort(
                        list(self.choice_dataset.shared_features_by_choice_map[tuple_index].keys())
                    ):
                        unstacked_feat.append(
                            shared_features_by_choice[tuple_index][:, feat_ind_min:feature_index]
                        )
                        unstacked_feat.append(
                            self.choice_dataset.shared_features_by_choice_map[tuple_index][
                                feature_index
                            ].batch[shared_features_by_choice[tuple_index][:, feature_index]]
                        )
                        feat_ind_min = feature_index + 1
                    mapped_features.append(np.concatenate(unstacked_feat, axis=1))
                else:
                    mapped_features.append(shared_features_by_choice[tuple_index])

            shared_features_by_choice = mapped_features

        if len(self.choice_dataset.items_features_by_choice_map) > 0:
            mapped_features = []
            for tuple_index in range(len(items_features_by_choice)):
                if tuple_index in self.choice_dataset.items_features_by_choice_map.keys():
                    feat_ind_min = 0
                    unstacked_feat = []
                    for feature_index in np.sort(
                        list(self.choice_dataset.items_features_by_choice_map[tuple_index].keys())
                    ):
                        unstacked_feat.append(
                            items_features_by_choice[tuple_index][:, :, feat_ind_min:feature_index]
                        )
                        unstacked_feat.append(
                            self.choice_dataset.items_features_by_choice_map[tuple_index][
                                feature_index
                            ].batch[items_features_by_choice[tuple_index][:, :, feature_index]]
                        )
                        feat_ind_min = feature_index + 1
                    mapped_features.append(np.concatenate(unstacked_feat, axis=2))
                else:
                    mapped_features.append(items_features_by_choice[tuple_index])

            items_features_by_choice = mapped_features

        if shared_features_by_choice is not None:
            for i in range(len(shared_features_by_choice)):
                shared_features_by_choice[i] = shared_features_by_choice[i].astype(
                    self.choice_dataset._return_types[0][i]
                )
            if not self.choice_dataset._return_shared_features_by_choice_tuple:
                shared_features_by_choice = shared_features_by_choice[0]
            else:
                shared_features_by_choice = tuple(shared_features_by_choice)

        if items_features_by_choice is not None:
            for i in range(len(items_features_by_choice)):
                items_features_by_choice[i] = items_features_by_choice[i].astype(
                    self.choice_dataset._return_types[1][i]
                )
            # items_features_by_choice were not given as a tuple, so we return do not return
            # it as a tuple
            if not self.choice_dataset._return_items_features_by_choice_tuple:
                items_features_by_choice = items_features_by_choice[0]
            else:
                items_features_by_choice = tuple(items_features_by_choice)

        return (
            shared_features_by_choice,
            items_features_by_choice,
            available_items_by_choice,
            choices,
        )
