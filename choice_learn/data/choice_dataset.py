"""Main classes to handle assortment data."""

import numpy as np
import pandas as pd

from choice_learn.data.indexer import ChoiceDatasetIndexer
from choice_learn.data.store import Store


class ChoiceDataset(object):
    """ChoiceDataset is the main class to handle assortment data minimizing RAM usage.

    The choices are given as a ragged list of choices
    for each session. It is particularly useful if several (a lot) of choices happen
    during the same session. For example if we have the same customer buying several
    items during the same session, all its choices
    can be regrouped under the same session_features. Limits data duplication in such cases.

    The class has same methods/arguments as ChoiceDatset with a slight difference with
    self.choices being a ragged list. The returned features in self.__getitem__ are the same
    as ChoiceDataset. When calling __getitem__(index) we map index to a session index and a
    choice index within the session.
    """

    def __init__(
        self,
        items_features=None,
        sessions_features=None,
        sessions_items_features=None,
        items_features_names=None,
        sessions_features_names=None,
        sessions_items_features_names=None,
        sessions_items_availabilities=None,
        choices=None,  # Should not have None as default value ?
        batch_size=16,
        shuffle=False,
    ):
        """Builds the ChoiceDataset.

        Parameters
        ----------
        items_features : tuple of (array_like, )
            matrix of shape (num_items, num_items_features) containing the features of the items
            e.g. item color
        sessions_features : tuple of (array_like, )
            matrix of shape (num_sessions, num_sess_features) containing the features of the
            sessions e.g. day of week
        sessions_items_features : tuple of (array_like, )
            matrix of shape (num_sessions, num_items, num_ses_items_features) containing the item
            features varying over sessions, e.g. prices
        sessions_items_availabilities : array_like
            binary matrix of shape (num_sessions, num_items) containing the availabitilies of
            products (1. if present 0. otherwise) over sessions
        choices: list of list
            for each sessions we have a list of related choices. Main list has same legnth as
            session_features and sessions_items_features.
        batch_size: int, optional
            size of the batches to return in __iter__ method
        suffle: bool, optional
            whether to shuffle the dataset or not
        """
        # --------- [ Handling features type given as tuples or not ] --------- #
        # If items_features is not given as tuple, transform it internally as a tuple
        # A bit longer because can be None and need to also handle names
        if not isinstance(items_features, tuple) and items_features is not None:
            items_features = (items_features,)
            items_features_names = (items_features_names,)
            self._return_items_features_tuple = False
        # items_features is already a tuple, names are given, checking consistency
        elif items_features is not None and items_features_names is not None:
            if (
                len(items_features) != len(items_features_names)
                and items_features_names is not None
            ):
                raise ValueError("items_features shape and items_features_names shape do not match")
            self._return_items_features_tuple = True
        # In this case names are missing, still transform it as a tuple
        elif items_features is not None:
            self._return_items_features_tuple = True
            items_features_names = (None,) * len(items_features)

        # If sessions_features is not given as tuple, transform it internally as a tuple
        # A bit longer because can be None and need to also handle names
        if not isinstance(sessions_features, tuple) and sessions_features is not None:
            sessions_features = (sessions_features,)
            sessions_features_names = (sessions_features_names,)
            self._return_sessions_features_tuple = False
        # sessions_features is already a tuple, names are given, checking consistency
        elif sessions_features is not None and sessions_features_names is not None:
            if (
                len(sessions_features) != len(sessions_features_names)
                and sessions_features_names is not None
            ):
                raise ValueError(
                    "sessions_features shape and sessions_features_names shape \
                                 do not match"
                )
            self._return_sessions_features_tuple = True
        # In this case names are missing, still transform it as a tuple
        elif sessions_features is not None:
            self._return_sessions_features_tuple = True
            sessions_features_names = (None,) * len(sessions_features)

        # If sessions_items_features is not given as tuple, transform it internally as a tuple
        # A bit longer because can be None and need to also handle names
        if not isinstance(sessions_items_features, tuple) and sessions_items_features is not None:
            sessions_items_features = (sessions_items_features,)
            sessions_items_features_names = (sessions_items_features_names,)
            self._return_sessions_items_features_tuple = False
        # sessions_items_features is already a tuple, names are given, checking consistency
        elif sessions_items_features is not None and sessions_items_features_names is not None:
            if (
                len(sessions_items_features) != len(sessions_items_features_names)
                and sessions_items_features_names is not None
            ):
                raise ValueError(
                    "sessions_items_features shape and \
                                    sessions_items_features_names shape do not match"
                )
            self._return_sessions_items_features_tuple = True
        # In this case names are missing, still transform it as a tuple
        elif sessions_items_features is not None:
            self._return_sessions_items_features_tuple = True
            sessions_items_features_names = (None,) * len(sessions_items_features)

        # --------- [Normalizing features types (DataFrame, List, etc...) -> np.ndarray] --------- #
        #
        # Part of this code is for handling features given as pandas.DataFrame
        # Basically it transforms them to be internally stocked as np.ndarray and keep columns
        # names as features names

        # Handling items_features
        for i, feature in enumerate(items_features):
            if isinstance(feature, pd.DataFrame):
                # Ordering items by id ?
                if "item_id" in feature.columns:
                    feature = feature.set_index("item_id")
                items_features = (
                    items_features[:i]
                    + (feature.loc[np.sort(feature.index)].to_numpy(),)
                    + items_features[i + 1 :]
                )
                items_features_names = (
                    items_features_names[:i]
                    + (feature.columns.tolist(),)
                    + items_features_names[i + 1 :]
                )
            elif isinstance(feature, list):
                items_features = items_features[:i] + (np.array(feature),) + items_features[i + 1 :]

        # Handling sessions_features
        for i, feature in enumerate(sessions_features):
            if isinstance(feature, pd.DataFrame):
                # Ordering sessions by id ?
                if "session_id" in feature.columns:
                    feature = feature.set_index("session_id")
                sessions_features = (
                    sessions_features[:i]
                    + (feature.loc[np.sort(feature.index)].to_numpy(),)
                    + sessions_features[i + 1 :]
                )
                sessions_features_names = (
                    sessions_features_names[:i]
                    + (feature.columns.tolist(),)
                    + sessions_features_names[i + 1 :]
                )
            elif isinstance(feature, list):
                sessions_features = (
                    sessions_features[:i] + (np.array(feature),) + sessions_features[i + 1 :]
                )

        # Handling sessions_items_features
        for i, feature in enumerate(sessions_items_features):
            if isinstance(feature, pd.DataFrame):
                # Ordering sessions and items by id ?
                if "session_id" not in feature.columns:
                    feature["session_id"] = feature.index
                items_index = np.sort(feature.item_id.unique())
                sessions_index = np.sort(feature.session_id.unique())
                names = [f for f in feature.columns if f != "session_id" and f != "item_id"]

                (
                    feature,
                    sessions_items_availabilities,
                ) = self._sessions_items_features_df_to_np(
                    feature, items_index, sessions_index, feature.columns.tolist()
                )

                sessions_items_features = (
                    sessions_items_features[:i] + feature + sessions_items_features[i + 1 :]
                )

                sessions_items_features_names = (
                    sessions_items_features_names[:i]
                    + (names,)
                    + sessions_items_features_names[i + 1 :]
                )
            elif isinstance(feature, list):
                sessions_items_features = (
                    sessions_items_features[:i]
                    + (np.array(feature),)
                    + sessions_items_features[i + 1 :]
                )

        if isinstance(sessions_items_availabilities, list):
            sessions_items_availabilities = np.array(sessions_items_availabilities)

        # Handling choices
        # Choices must then be given as the name of the chosen item
        # Items are sorted by name and attributed an index
        # Cannot be a list of choices yet
        if isinstance(choices, pd.DataFrame):
            # Ordering sessions by id
            if "session_id" in choices.columns:
                choices = choices.set_index("session_id")
            choices = choices.loc[np.sort(choices.index)]
            items = np.sort(np.unique(choices.choice))
            # items is the value (str) of the item
            choices = [np.where(items == c)[0] for c in choices.choice]

        # Setting attributes of ChoiceDataset
        self.items_features = items_features
        self.sessions_features = sessions_features
        self.sessions_items_features = sessions_items_features
        self.sessions_items_availabilities = sessions_items_availabilities

        self.items_features_names = items_features_names
        self.sessions_features_names = sessions_features_names
        self.sessions_items_features_names = sessions_items_features_names

        self.batch_size = batch_size
        self.shuffle = shuffle

        if choices is None:
            # Done to keep a logical order of arguments, and has logic: choices have to be specified
            raise ValueError("Choices must be specified, got None")
        self.ragged_choices = choices
        self.indexes, self.choices = self._build_indexes(choices)
        self.n_choices = len(self.choices)

        # Different consitency checks to ensure everythin is coherent
        self._check_dataset()  # Should handle alone if np.arrays are squeezed
        self._return_types = self._check_types()
        self._check_names()

        # Build .iloc method
        self.indexer = ChoiceDatasetIndexer(self)

    def _build_indexes(self, choices):
        """Builds the indexes dictionnary from the choices.

        Particularly creates a flatten version of the choices and associates an index so that we can
        retrieve from this index the session and the corresponding choice.

        Parameters:
        -----------
        choices: list of list
            raffed version of the choices

        Returns::
        --------
        indexes: dict
            dictionnary of indexes: {index: corresponding_session_index}
        choices: np.ndarray
            flattened (1D) version of the choices
        """
        try:  # 1 choice by session
            if len(np.squeeze(choices).shape) == 1:
                indexes = {i: i for i in range(len(choices))}
                flat_choices = np.squeeze(self.ragged_choices)
            elif len(np.squeeze(choices).shape) == 0:
                indexes = {i: i for i in range(len(choices))}
                flat_choices = np.array([np.squeeze(self.ragged_choices)])
        except ValueError:  # Ragged sequence of choices
            indexes = {}
            flat_choices = []
            total_count = 0
            for sess_nb, sess in enumerate(choices):
                for choice in sess:
                    indexes[total_count] = sess_nb
                    flat_choices.append(choice)
                    total_count += 1
        return indexes, np.array(flat_choices)

    def _check_dataset(self):
        """Verifies that the shapes of the different features are consistent.

        Particularly:
            - Over number of items
            - Over number of sessions
        Verifies that the choices have coherent values
        """
        self._check_num_items_shapes()
        self._check_num_sessions_shapes()
        self._check_choices_coherence()

    def _check_num_items_shapes(self):
        """Verifies that the shapes of the different features are consistent over number of items.

        Particularly:
            - items_features
            - sessions_items_features
            - sessions_items_availabilities
        Sets the argument base_num_items
        """
        if self.items_features is not None:
            base_num_items = self.items_features[0].shape[0]
        elif self.sessions_items_features is not None:
            base_num_items = self.sessions_items_features[0].shape[1]
        elif self.sessions_items_availabilities is not None:
            base_num_items = self.sessions_items_availabilities.shape[1]
        else:
            raise ValueError(
                "No items features, sessions items features or items availabilities are defined"
            )
        self.base_num_items = base_num_items

        if self.items_features is not None:
            for items_feature in self.items_features:
                if items_feature.shape[0] != base_num_items:
                    raise ValueError(f"shapes are (f{items_feature.shape[0]}, {base_num_items})")

        if self.sessions_items_features is not None:
            for sessions_items_feature in self.sessions_items_features:
                if sessions_items_feature.shape[1] != base_num_items:
                    raise ValueError(
                        f"shapes are (f{sessions_items_feature.shape[1]}, {base_num_items})"
                    )
        if self.sessions_items_availabilities is not None:
            if self.sessions_items_availabilities.shape[1] != base_num_items:
                raise ValueError(
                    f"shapes are (f{self.sessions_items_availabilities.shape[1]}, \
                                 {base_num_items})"
                )

    def _check_num_sessions_shapes(self):
        """Verifies that the shapes of the different features are consistent over nb of sessions.

        Particularly:
            - sessions_features
            - sessions_items_features
            - sessions_items_availabilities
        Sets self.base_num_sessions argument.
        """
        base_num_sessions = len(self.ragged_choices)
        self.base_num_sessions = base_num_sessions

        if self.sessions_features is not None:
            for sessions_feature in self.sessions_features:
                if sessions_feature.shape[0] != base_num_sessions:
                    raise ValueError(
                        f"shapes are ({sessions_feature.shape[0]}, {base_num_sessions})"
                    )

        if self.sessions_items_features is not None:
            for sessions_items_feature in self.sessions_items_features:
                if sessions_items_feature.shape[0] != base_num_sessions:
                    raise ValueError(
                        f"shapes are: ({sessions_items_feature.shape[0]}, \
                                     {base_num_sessions})"
                    )
        if self.sessions_items_availabilities is not None:
            if self.sessions_items_availabilities.shape[0] != base_num_sessions:
                raise ValueError(
                    f"shapes are: ({self.sessions_items_availabilities.shape[0]}, \
                        {base_num_sessions})"
                )

    def _check_choices_coherence(self):
        """Verifies that the choices are coherent with the nb of items present in other features.

        Particularly:
            - There is no choice index higher than detected number of items
            - All items are present at least once in the choices
        """
        if np.max(self.choices) > self.base_num_items:
            msg = f"Choices values not coherent with number of items given in features.  \
            In particular, max value of choices is {np.max(self.choices)} while number of  \
            items is {self.base_num_items}"
            raise ValueError(msg)

        unique_choices = set(np.unique(self.choices).flatten())
        missing_choices = set(np.arange(start=0, stop=self.base_num_items, step=1)) - unique_choices
        if len(missing_choices) > 0:
            print(f"Some choices never happen in the dataset: {missing_choices}")

    def _check_types(self):
        """Checks types of elements and store it in order to return right types.

        Particularly:
            - Either int32 or float32 consistently for features.
                float32 is to be preferred unless One-Hot encoding is used.
            - float32 for sessions_items_availabilities
            - int32 for choices
        """
        return_types = []

        item_types = []
        if self.items_features is not None:
            for item_feat in self.items_features:
                if np.issubdtype(item_feat[0].dtype, np.integer):
                    item_types.append(np.int32)
                else:
                    item_types.append(np.float32)
        return_types.append(tuple(item_types))

        session_types = []
        if self.sessions_features is not None:
            for sessions_feat in self.sessions_features:
                if np.issubdtype(sessions_feat[0].dtype, np.integer):
                    session_types.append(np.int32)
                else:
                    session_types.append(np.float32)
        return_types.append(tuple(session_types))

        session_item_types = []
        if self.sessions_items_features is not None:
            for session_item_feat in self.sessions_items_features:
                if np.issubdtype(session_item_feat[0].dtype, np.integer):
                    session_item_types.append(np.int32)
                else:
                    session_item_types.append(np.float32)
        return_types.append(tuple(session_item_types))
        return_types.append(np.float32)
        return_types.append(np.int32)

        return return_types

    def _check_names(self):
        """Verifies that the names given to features are consistent with the features themselves."""
        if self.items_features_names is not None:
            for name, features in zip(self.items_features_names, self.items_features):
                if name is not None:
                    if len(name) != features.shape[1]:
                        raise ValueError(
                            f"Specififed items_features_names has \
                    length {len(name)} while items_features has {features.shape[1]} elements"
                        )

        if self.sessions_features_names is not None:
            for name, features in zip(self.sessions_features_names, self.sessions_features):
                if name is not None:
                    if len(name) != features.shape[1]:
                        raise ValueError(
                            f"Specified sessions_features_names has \
                    length {len(name)} while sessions_features has {features.shape[1]} elements"
                        )

        if self.sessions_items_features_names is not None:
            for (
                name,
                features,
            ) in zip(self.sessions_items_features_names, self.sessions_items_features):
                if name is not None:
                    if len(name) != features.shape[2]:
                        raise ValueError(
                            f"Specified \
                        sessions_items_features_names has length {len(name)} while \
                        sessions_items_features has {features.shape[1]} elements"
                        )

    def __len__(self):
        """Returns length of the dataset e.g. total number of sessions.

        Returns:
        -------
        int
            total number of sessions
        """
        return self.base_num_sessions

    def get_num_items(self):
        """Method to access the total number of different items.

        Returns:
        -------
        int
            total number of different items
        """
        return self.base_num_items

    def get_num_sessions(self):
        """Method to access the total number of different sessions.

        Redundant with __len__ method.

        Returns:
        -------
        int
            total number of different sessions
        """
        return len(self)

    def get_num_choices(self):
        """Method to access the total number of different sessions.

        Returns:
        -------
        int
            total number of different sessions
        """
        return self.n_choices

    @classmethod
    def _sessions_items_features_df_to_np(
        cls,
        df,
        items_index,
        sessions_index,
        features,
        items_id_column="item_id",
        sessions_id_column="session_id",
    ):
        """Builds sessions_items_features and sessions_items_availabilities from dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing all the features for each item and sessions
        items_index : list
            List of items
        sessions_index : list
            List of sessions
        features : list
            List of columns of df that represents the items_features (for sessions_items_features)

        Returns:
        -------
        np.ndarray of shape (n_sessions, n_items, n_features)
            Corresponding sessions_items_features
        np.ndarray of shape (n_sessions, n_items)
            Corresponding availabilities
        """
        try:
            features.remove("session_id")
        except ValueError:
            pass
        try:
            features.remove("item_id")
        except ValueError:
            pass

        sessions_items_features = []
        sessions_items_availabilities = []
        for sess in sessions_index:
            sess_df = df.loc[df[sessions_id_column] == sess]

            if len(sess_df) == len(items_index):
                sess_df = sess_df.T
                sess_df.columns = sess_df.loc[items_id_column]
                if features is not None:
                    sessions_items_features.append(sess_df[items_index].loc[features].T.values)
                sessions_items_availabilities.append(np.ones(len(items_index)))
            else:
                sess_feats = []
                sess_av = []
                for item in items_index:
                    item_df = sess_df.loc[sess_df[items_id_column] == item]
                    if len(item_df) > 0:
                        if features is not None:
                            sess_feats.append(item_df[features].to_numpy()[0])
                        sess_av.append(1)
                    else:
                        if features is not None:
                            sess_feats.append(np.zeros(len(features)))
                        sess_av.append(0)
                sessions_items_features.append(sess_feats)
                sessions_items_availabilities.append(sess_av)

        if features is not None:
            sessions_items_features = (np.array(sessions_items_features),)
        else:
            sessions_items_features = None
        return sessions_items_features, np.array(sessions_items_availabilities)

    @classmethod
    def from_single_df(
        cls,
        df,
        items_features_columns,
        sessions_features_columns,
        sessions_items_features_columns,
        items_id_column="item_id",
        sessions_id_column="session_id",
        choices_column="choice",
        choice_mode="items_name",
    ):
        """Builds numpy arrays for ChoiceDataset from a single dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            dataframe in Long format
        items_features_columns : list
            Columns of the dataframe that are item features
        sessions_features_columns : list
            Columns of the dataframe that are session features
        sessions_items_features_columns : list
            Columns of the dataframe that are session-item features
        items_id_column: str, optional
            Name of the column containing the item ids, default is "items_id"
        sessions_id_column: str, optional
            Name of the column containing the sessions ids, default is "sessions_id"
        choices_column: str, optional
            Name of the column containing the choices, default is "choice"

        Returns:
        -------
        ChoiceDataset
            corresponding ChoiceDataset
        """
        # Ordering items and sessions by id
        items = np.sort(df[items_id_column].unique())
        sessions = np.sort(df[sessions_id_column].unique())

        if items_features_columns is not None:
            items_features = df[items_features_columns + [items_id_column]].drop_duplicates()
            items_features = items_features.set_index(items_id_column)
            items_features = (items_features.loc[items].to_numpy(),)

            items_features_columns = (items_features_columns,)
        else:
            items_features = None

        if sessions_features_columns is not None:
            sessions_features = df[
                sessions_features_columns + [sessions_id_column]
            ].drop_duplicates()
            sessions_features = sessions_features.set_index(sessions_id_column)
            sessions_features = (sessions_features.loc[sessions].to_numpy(),)

            sessions_features_columns = (sessions_features_columns,)
        else:
            sessions_features = None

        (
            sessions_items_features,
            sessions_items_availabilities,
        ) = cls._sessions_items_features_df_to_np(
            df,
            items_index=items,
            sessions_index=sessions,
            features=sessions_items_features_columns,
            items_id_column=items_id_column,
            sessions_id_column=sessions_id_column,
        )
        sessions_items_features_columns = (
            (sessions_items_features_columns,)
            if sessions_items_features_columns is not None
            else None
        )

        if choice_mode == "item_id":
            choices = df[[choices_column, sessions_id_column]].drop_duplicates(sessions_id_column)
            choices = choices.set_index(sessions_id_column)
            choices = choices.loc[sessions].to_numpy()
            # items is the value (str) of the item
            choices = [np.where(items == c)[0] for c in choices]
        elif choice_mode == "one_zero":
            choices = df[[items_id_column, choices_column, sessions_id_column]]
            choices = choices.loc[choices[choices_column] == 1]
            choices = choices = choices.set_index(sessions_id_column)
            choices = (
                choices.loc[sessions][items_id_column]
                .map({k: v for v, k in enumerate(items)})
                .to_numpy()
            )
        else:
            raise ValueError(
                f"choice_mode {choice_mode} not recognized. Must be in ['item_id', 'one_zero']"
            )
        return ChoiceDataset(
            items_features=items_features,
            sessions_features=sessions_features,
            sessions_items_features=sessions_items_features,
            sessions_items_availabilities=sessions_items_availabilities,
            choices=choices,
            items_features_names=items_features_columns,
            sessions_features_names=sessions_features_columns,
            sessions_items_features_names=sessions_items_features_columns,
        )

    def save(self):
        """Method to save the dataset."""
        raise NotImplementedError

    def summary(self):
        """Method to display a summary of the dataset."""
        print("Summary of the dataset:")
        print("Number of items:", self.get_num_items())
        print("Number of sessions:", self.get_num_sessions())
        print(
            "Number of choices:",
            self.get_num_choices(),
            "Averaging",
            self.get_num_choices() / self.get_num_sessions(),
            "choices per session",
        )
        if self.items_features is not None:
            print(f"Items features: {self.items_features_names}")
        if self.items_features is not None:
            print(f"{sum([f.shape[1] for f in self.items_features])} items features")
        else:
            print("No items features registered")

        if self.sessions_features is not None:
            print(f"Sessions features: {self.sessions_features_names}")
        if self.sessions_features is not None:
            print(f"{sum([f.shape[1] for f in self.sessions_features])} session features")
        else:
            print("No sessions features registered")

        if self.sessions_items_features is not None:
            print(f"Session Items features: {self.sessions_items_features_names}")
        if self.sessions_items_features is not None:
            print(
                f"{sum([f.shape[2] for f in self.sessions_items_features])} sessions \
                  items features"
            )
        else:
            print("No sessions items features registered")

    def get_choice_batch(self, choice_index):
        """Method to access data within the ListChoiceDataset from its index.

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
            if self.items_features is None:
                items_features = None
            else:
                items_features = tuple(
                    items_feature.astype(self._return_types[0][i])
                    for i, items_feature in enumerate(self.items_features)
                )
                # items_features were not given as a tuple, so we return do not return it as a tuple
                if not self._return_items_features_tuple:
                    items_features = items_features[0]

            # Get the session indexes
            sessions_indexes = [self.indexes[i] for i in choice_index]

            if self.sessions_features is None:
                sessions_features = None
            else:
                sessions_features = tuple(
                    np.stack(sessions_feature[sessions_indexes], axis=0).astype(
                        self._return_types[1][i]
                    )
                    if not isinstance(sessions_feature, Store)
                    else sessions_feature.iloc[sessions_indexes]
                    for i, sessions_feature in enumerate(self.sessions_features)
                )
                # sessions_features were not given as a tuple, so we return do not return it
                # as a tuple
                if not self._return_sessions_features_tuple:
                    sessions_features = sessions_features[0]

            if self.sessions_items_features is None:
                sessions_items_features = None
            else:
                sessions_items_features = tuple(
                    np.stack(sessions_items_feature[sessions_indexes], axis=0).astype(
                        self._return_types[2][i]
                    )
                    if not isinstance(sessions_items_feature, Store)
                    else sessions_items_feature.iloc[sessions_indexes]
                    for i, sessions_items_feature in enumerate(self.sessions_items_features)
                )
                # sessions_items_features were not given as a tuple, so we return do not return
                # it as a tuple
                if not self._return_sessions_items_features_tuple:
                    sessions_items_features = sessions_items_features[0]

            if self.sessions_items_availabilities is None:
                sessions_items_availabilities = None
            else:
                sessions_items_availabilities = self.sessions_items_availabilities[
                    sessions_indexes
                ].astype(self._return_types[3])

            choice = self.choices[choice_index].astype(self._return_types[4])

            return (
                items_features,
                sessions_features,
                sessions_items_features,
                sessions_items_availabilities,
                choice,
            )

        if isinstance(choice_index, slice):
            return self.get_choice_batch(list(range(*choice_index.indices(self.choices.shape[0]))))

        session_index = self.indexes[choice_index]
        choice = self.choices[choice_index]

        if self.items_features is None:
            items_features = None
        else:
            items_features = tuple(items_feature for items_feature in self.items_features)

        if self.sessions_features is None:
            sessions_features = None
        else:
            sessions_features = tuple(
                sessions_feature[session_index] for sessions_feature in self.sessions_features
            )

        if self.sessions_items_features is None:
            sessions_items_features = None
        else:
            sessions_items_features = tuple(
                sessions_items_feature[session_index]
                for sessions_items_feature in self.sessions_items_features
            )

        if self.sessions_items_availabilities is None:
            sessions_items_availabilities = None
        else:
            sessions_items_availabilities = self.sessions_items_availabilities[session_index]

        return (
            items_features,
            sessions_features,
            sessions_items_features,
            sessions_items_availabilities,
            choice,
        )

    def __getitem__(self, session_indexes):
        """Method to create a sub-ChoiceDataset with only a subset of sessions, from their indexes.

        Parameters
        ----------
        indexes : np.ndarray
            indexes of the sessions to keep, shape should be (num_sessions,)

        Returns:
        -------
        ChoiceDataset
            ChoiceDataset with only the sessions indexed by indexes
        """
        if isinstance(session_indexes, int):
            session_indexes = [session_indexes]
        elif isinstance(session_indexes, slice):
            return self.__getitem__(list(range(*session_indexes.indices(len(self.ragged_choices)))))

        return ChoiceDataset(
            items_features=self.items_features,
            sessions_features=tuple(
                self.sessions_features[i][session_indexes]
                for i in range(len(self.sessions_features))
            ),
            sessions_items_features=tuple(
                self.sessions_items_features[i][session_indexes]
                for i in range(len(self.sessions_items_features))
            ),
            sessions_items_availabilities=self.sessions_items_availabilities[session_indexes],
            choices=[self.ragged_choices[i] for i in session_indexes],
            batch_size=self.batch_size,
            items_features_names=self.items_features_names,
            sessions_features_names=self.sessions_features_names,
            sessions_items_features_names=self.sessions_items_features_names,
        )

    def old_batch(self, batch_size=None, shuffle=None, sample_weight=None):
        """Iterates over dataset return batches of length self.batch_size.

        Parameters
        ----------
        batch_size : int
            batch size to set
        shuffle: bool
            Whether or not to shuffle the dataset
        sample_weight : Iterable
            list of weights to be returned with the right indexing during the shuffling
        """
        if batch_size is None:
            batch_size = self.batch_size
        if shuffle is None:
            shuffle = self.shuffle
        if batch_size == -1:
            batch_size = self.get_num_choices()

        # Get indexes for each choice
        num_choices = self.get_num_choices()
        indexes = np.arange(num_choices)
        # Shuffle indexes
        if shuffle and not batch_size == -1:
            indexes = np.random.permutation(indexes)

        yielded_size = 0
        while yielded_size < num_choices:
            # Return sample_weight if not None, for index matching
            if sample_weight is not None:
                yield (
                    self.get_choice_batch(
                        indexes[yielded_size : yielded_size + batch_size].tolist()
                    ),
                    sample_weight[indexes[yielded_size : yielded_size + batch_size].tolist()],
                )
            else:
                yield self.get_choice_batch(
                    indexes[yielded_size : yielded_size + batch_size].tolist()
                )
            yielded_size += batch_size

            # Special exit strategy for batch_size = -1
            if batch_size == -1:
                yielded_size += 2 * num_choices

    @property
    def batch(self):
        """Indexer."""
        return self.indexer

    def iter_batch(self, batch_size=None, shuffle=None, sample_weight=None):
        """Iterates over dataset return batches of length self.batch_size.

        Newer version.

        Parameters
        ----------
        batch_size : int
            batch size to set
        shuffle: bool
            Whether or not to shuffle the dataset
        sample_weight : Iterable
            list of weights to be returned with the right indexing during the shuffling
        """
        if batch_size is None:
            batch_size = self.batch_size
        if shuffle is None:
            shuffle = self.shuffle
        if batch_size == -1:
            batch_size = self.get_num_choices()

        # Get indexes for each choice
        num_choices = self.get_num_choices()
        indexes = np.arange(num_choices)
        # Shuffle indexes
        if shuffle and not batch_size == -1:
            indexes = np.random.permutation(indexes)

        yielded_size = 0
        while yielded_size < num_choices:
            # Return sample_weight if not None, for index matching
            if sample_weight is not None:
                yield (
                    self.batch[indexes[yielded_size : yielded_size + batch_size].tolist()],
                    sample_weight[indexes[yielded_size : yielded_size + batch_size].tolist()],
                )
            else:
                yield self.batch[indexes[yielded_size : yielded_size + batch_size].tolist()]
            yielded_size += batch_size

            # Special exit strategy for batch_size = -1
            if batch_size == -1:
                yielded_size += 2 * num_choices

    def filter(self, bool_list):
        """Filter over sessions indexes following bool.

        Parameters
        ----------
        bool_list : list of boolean
            list of booleans of length self.get_num_sessions() to filter sessions.
            True to keep, False to discard.
        """
        indexes = list(range(len(bool_list)))
        indexes = [i for i, keep in zip(indexes, bool_list) if keep]
        return self[indexes]
