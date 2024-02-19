"""Main classes to handle assortment data."""

import numpy as np
import pandas as pd

from choice_learn.data.indexer import ChoiceDatasetIndexer
from choice_learn.data.storage import Storage


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
        choices,  # Should not have None as default value ?
        fixed_items_features=None,
        contexts_features=None,  # as many context as choices.  values or ids (look at key)
        contexts_items_features=None,
        contexts_items_availabilities=None,
        features_by_ids=[],  # list of (name, FeaturesStorage)
        fixed_items_features_names=None,
        contexts_features_names=None,
        contexts_items_features_names=None,
    ):
        """Builds the ChoiceDataset.

        Parameters
        ----------
        choices: list or np.ndarray
            list of chosen items indexes
        fixed_items_features : tuple of (array_like, )
            matrix of shape (num_items, num_items_features) containing the features of the items
            that never change, e.g. item color, default is None
        contexts_features : tuple of (array_like, )
            matrix of shape (num_choices, num_contexts_features) containing the features of the
            different contexts that are common to all items (e.g. store features,
            customer features, etc...)
        contexts_items_features : tuple of (array_like, ), default is None
            matrix of shape (num_choices, num_items, num_contexts_items_features)
            containing the features
            of the items that change over time (e.g. price, promotion, etc...), default is None
        contexts_items_availabilities : array_like
            matrix of shape (num_choices, num_items) containing the availabilities of the items
            over the different choices, default is None
        features_by_ids : list of (name, FeaturesStorage)
            List of Storage objects. Their name must correspond to a feature name
            among fixed_items, contexts,
            contexts_items and their ids must match to those features values. Default is []
        fixed_items_features_names : tuple of (array_like, )
            list of names of the fixed_items_features, default is None
        contexts_features_names : tuple of (array_like, )
            list of names of the contexts_features, default is None
        contexts_items_features_names : tuple of (array_like, )
            list of names of the contexts_items_features, default is None
        """
        if choices is None:
            # Done to keep a logical order of arguments, and has logic: choices have to be specified
            raise ValueError("Choices must be specified, got None")

        # --------- [ Handling features type given as tuples or not ] --------- #
        # If items_features is not given as tuple, transform it internally as a tuple
        # A bit longer because can be None and need to also handle names

        if fixed_items_features is not None:
            if not isinstance(fixed_items_features, tuple):
                if fixed_items_features_names is not None:
                    if len(fixed_items_features) == len(fixed_items_features_names):
                        raise ValueError(
                            """Number of features given does not match number
                                         of features names given."""
                        )
                self._return_items_features_tuple = False

                fixed_items_features = (fixed_items_features,)
                fixed_items_features_names = (fixed_items_features_names,)
            else:
                self._return_items_features_tuple = True

                # items_features is already a tuple, names are given, checking consistency
                if fixed_items_features_names is not None:
                    for f, name in zip(fixed_items_features, fixed_items_features_names):
                        if len(f[0]) != len(name):
                            raise ValueError(
                                "items_features shape and items_features_names shape do not match"
                            )
                # In this case names are missing, still transform it as a tuple
                else:
                    fixed_items_features_names = (None,) * len(fixed_items_features)
        else:
            self._return_items_features_tuple = True

        # If choices_features is not given as tuple, transform it internally as a tuple
        # A bit longer because can be None and need to also handle names
        if contexts_features is not None:
            if not isinstance(contexts_features, tuple):
                self._return_contexts_features_tuple = False
                if contexts_features_names is not None:
                    if len(contexts_features[0]) != len(contexts_features_names):
                        raise ValueError(
                            f"""Number of features given does not match
                                         number of features names given:
                                           {len(contexts_features[0])} and
                                            {len(contexts_features_names)}"""
                        )

                contexts_features_names = (contexts_features_names,)
                contexts_features = (contexts_features,)

            # choices_features is already a tuple, names are given, checking consistency
            else:
                self._return_contexts_features_tuple = True
                if contexts_features_names is not None:
                    for f, name in zip(contexts_features, contexts_features_names):
                        if len(f[0]) != len(name):
                            raise ValueError(
                                """contexts_features shape and contexts_features_names
                                shape do not match"""
                            )

                # In this case names are missing, still transform it as a tuple
                else:
                    contexts_features_names = (None,) * len(contexts_features)
        else:
            self._return_contexts_features_tuple = True

        if not isinstance(contexts_items_features, tuple) and contexts_items_features is not None:
            self._return_contexts_items_features_tuple = False
            if contexts_items_features_names is not None:
                if len(contexts_items_features[0][0]) != len(contexts_items_features_names):
                    raise ValueError(
                        f"""Number of features given does not match
                                     number of features names given for contexts_items:
                                     {len(contexts_items_features[0][0])} and
                                     {len(contexts_items_features_names)}"""
                    )
            contexts_items_features = (contexts_items_features,)
            contexts_items_features_names = (contexts_items_features_names,)

        # sessions_items_features is already a tuple, names are given, checking consistency
        elif contexts_items_features is not None and contexts_items_features_names is not None:
            for f, name in zip(contexts_items_features, contexts_items_features_names):
                if len(f[0][0]) != len(name):
                    raise ValueError(
                        """contexts_items_features shape and
                        contexts_items_features_names shape do not match"""
                    )
            self._return_contexts_items_features_tuple = True
        # In this case names are missing, still transform it as a tuple
        elif contexts_items_features is not None:
            self._return_contexts_items_features_tuple = True
            contexts_items_features_names = (None,) * len(contexts_items_features)

        else:
            self._return_contexts_items_features_tuple = True

        # --------- [Normalizing features types (DataFrame, List, etc...) -> np.ndarray] --------- #
        #
        # Part of this code is for handling features given as pandas.DataFrame
        # Basically it transforms them to be internally stocked as np.ndarray and keep columns
        # names as features names

        # Handling items_features
        if fixed_items_features is not None:
            for i, feature in enumerate(fixed_items_features):
                if isinstance(feature, pd.DataFrame):
                    # Ordering items by id ?
                    if "item_id" in feature.columns:
                        feature = feature.set_index("item_id")
                    fixed_items_features = (
                        fixed_items_features[:i]
                        + (feature.loc[np.sort(feature.index)].to_numpy(),)
                        + fixed_items_features[i + 1 :]
                    )
                    fixed_items_features_names = (
                        fixed_items_features_names[:i]
                        + (feature.columns.tolist(),)
                        + fixed_items_features_names[i + 1 :]
                    )
                elif isinstance(feature, list):
                    fixed_items_features = (
                        fixed_items_features[:i]
                        + (np.array(feature),)
                        + fixed_items_features[i + 1 :]
                    )

        # Handling context features
        if contexts_features is not None:
            for i, feature in enumerate(contexts_features):
                if isinstance(feature, pd.DataFrame):
                    # Ordering choices by id ?
                    if "context_id" in feature.columns:
                        feature = feature.set_index("context_id")
                    contexts_features = (
                        contexts_features[:i]
                        + (feature.loc[np.sort(feature.index)].to_numpy(),)
                        + contexts_features[i + 1 :]
                    )
                    contexts_features_names = (
                        contexts_features_names[:i]
                        + (feature.columns,)
                        + contexts_features_names[i + 1 :]
                    )
                elif isinstance(feature, list):
                    contexts_features = (
                        contexts_features[:i] + (np.array(feature),) + contexts_features[i + 1 :]
                    )
        # Handling contexts_items_features
        if contexts_items_features is not None:
            for i, feature in enumerate(contexts_items_features):
                if isinstance(feature, pd.DataFrame):
                    # Ordering choices by id ?
                    if "context_id" in feature.columns:
                        if "item_id" in feature.columns:
                            all_items = np.sort(feature.item_id.unique())
                            feature_array = []
                            temp_availabilities = []
                            for sess in np.sort(feature.context_id.unique()):
                                sess_df = feature.loc[feature.context_id == sess]
                                sess_df = sess_df[
                                    sess_df.columns.difference(["context_id"])
                                ].set_index("item_id")
                                sess_feature = []
                                sessions_availabilities = []
                                for item in all_items:
                                    if item in sess_df.index:
                                        sess_feature.append(sess_df.loc[item].to_numpy())
                                        sessions_availabilities.append(1)
                                    else:
                                        sess_feature.append(np.zeros(len(sess_df.columns)))
                                        sessions_availabilities.append(0)
                                feature_array.append(sess_feature)
                                temp_availabilities.append(sessions_availabilities)
                            contexts_items_features = (
                                contexts_items_features[:i]
                                + (np.stack(feature_array, axis=0),)
                                + contexts_items_features[i + 1 :]
                            )
                            contexts_items_features_names = (
                                contexts_items_features_names[:i]
                                + (sess_df.columns,)
                                + contexts_items_features_names[i + 1 :]
                            )
                            if (
                                contexts_items_availabilities is None
                                and len(np.unique(temp_availabilities)) > 1
                            ):
                                contexts_items_availabilities = np.array(temp_availabilities)
                        else:
                            feature = feature.set_index("context_id")
                            contexts_items_features = (
                                contexts_items_features[:i]
                                + (feature.loc[np.sort(feature.index)].to_numpy(),)
                                + contexts_items_features[i + 1 :]
                            )
                            contexts_items_features_names = (
                                contexts_items_features_names[:i]
                                + (feature.columns,)
                                + contexts_items_features_names[i + 1 :]
                            )
                    else:
                        raise ValueError("context_id column not found in contexts_items_features")
                elif isinstance(feature, list):
                    contexts_items_features = (
                        contexts_items_features[:i]
                        + (np.array(feature),)
                        + contexts_items_features[i + 1 :]
                    )
        if contexts_items_availabilities is not None:
            if isinstance(contexts_items_availabilities, list):
                contexts_items_availabilities = np.array(
                    contexts_items_availabilities, dtype=object
                )
            elif isinstance(contexts_items_availabilities, pd.DataFrame):
                if "context_id" in contexts_items_availabilities.columns:
                    if "item_id" in contexts_items_availabilities.columns:
                        av_array = []
                        for sess in np.sort(contexts_items_availabilities.context_id):
                            sess_df = contexts_items_availabilities.loc[
                                contexts_items_availabilities.context_id == sess
                            ]
                            sess_df = sess_df.set_index("item_id")
                            av_array.append(sess_df.loc[np.sort(sess_df.index)].to_numpy())
                        contexts_items_availabilities = np.array(av_array)
                    else:
                        feature = feature.set_index("context_id")
                        contexts_items_availabilities = contexts_items_availabilities.loc[
                            np.sort(feature.index)
                        ].to_numpy()

        # Handling choices
        # Choices must then be given as the name of the chosen item
        # Items are sorted by name and attributed an index
        if isinstance(choices, pd.DataFrame):
            # Ordering choices by id
            if "context_id" in choices.columns:
                choices = choices.set_index("context_id")
            choices = choices.loc[np.sort(choices.index)]
            items = np.sort(np.unique(choices.choice))
            # items is the value (str) of the item
            choices = [np.where(items == c)[0] for c in choices.choice]
            choices = np.squeeze(choices)
        elif isinstance(choices, list):
            choices = np.array(choices)

        # Setting attributes of ChoiceDataset
        self.fixed_items_features = fixed_items_features
        self.contexts_features = contexts_features
        self.contexts_items_features = contexts_items_features
        self.contexts_items_availabilities = contexts_items_availabilities
        self.choices = choices

        for fid in features_by_ids:
            if not isinstance(fid, Storage):
                raise ValueError("FeaturesByID must be Storage object")
        self.features_by_ids = features_by_ids

        self.fixed_items_features_names = fixed_items_features_names
        self.contexts_features_names = contexts_features_names
        self.contexts_items_features_names = contexts_items_features_names

        # What about typing ? should builf after check to change it ?
        (
            self.fixed_items_features_map,
            self.contexts_features_map,
            self.contexts_items_features_map,
        ) = self._build_features_by_ids()

        self.n_choices = len(self.choices)

        # Different consitency checks to ensure everything is coherent
        self._check_dataset()  # Should handle alone if np.arrays are squeezed
        self._return_types = self._check_types()
        self._check_names()

        # Build .iloc method
        self.indexer = ChoiceDatasetIndexer(self)

    def _build_features_by_ids(self):
        """Builds mapping function.

        Those mapping functions are so that at indexing,
        the features are rebuilt with the features by id.

        Returns:
        --------
        tuple
            indexes and features_by_id of fixed_items_features
        tuple
            indexes and features_by_id of contexts_features
        tuple
            indexes and features_by_id of contexts_items_features
        """
        if len(self.features_by_ids) == 0:
            print("No features_by_ids given.")
            return {}, {}, {}

        if (
            self.fixed_items_features_names is None
            and self.contexts_features_names is None
            and self.contexts_items_features_names is None
        ):
            raise ValueError(
                "No features_names given, match with fiven features_by_ids impossible."
            )

        fixed_items_features_map = {}
        contexts_features_map = {}
        contexts_items_features_map = {}

        if self.fixed_items_features_names is not None:
            for i, feature in enumerate(self.fixed_items_features_names):
                if feature is not None:
                    for j, column_name in enumerate(feature):
                        for feature_by_id in self.features_by_ids:
                            if column_name == feature_by_id.name:
                                index_dict = fixed_items_features_map.get(i, {})
                                index_dict[j] = feature_by_id
                                fixed_items_features_map[i] = index_dict

        if self.contexts_features_names is not None:
            for i, feature in enumerate(self.contexts_features_names):
                if feature is not None:
                    for j, column_name in enumerate(feature):
                        for feature_by_id in self.features_by_ids:
                            if column_name == feature_by_id.name:
                                index_dict = contexts_features_map.get(i, {})
                                index_dict[j] = feature_by_id
                                contexts_features_map[i] = index_dict

        if self.contexts_items_features_names is not None:
            for i, feature in enumerate(self.contexts_items_features_names):
                if feature is not None:
                    for k, column_name in enumerate(feature):
                        for feature_by_id in self.features_by_ids:
                            if column_name == feature_by_id.name:
                                index_dict = contexts_items_features_map.get(i, {})
                                index_dict[k] = feature_by_id
                                contexts_items_features_map[i] = index_dict
                                # contexts_items_features_map.append(((i, k), feature_by_id))

        if len(fixed_items_features_map) + len(contexts_features_map) + sum(
            [len(c.keys()) for c in contexts_items_features_map.values()]
        ) != len(self.features_by_ids):
            raise ValueError("Some features_by_ids were not matched with features_names.")

        return fixed_items_features_map, contexts_features_map, contexts_items_features_map

    def _check_dataset(self):
        """Verifies that the shapes of the different features are consistent.

        Particularly:
            - Over number of items
            - Over number of choices
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
        if self.fixed_items_features is not None:
            base_num_items = self.fixed_items_features[0].shape[0]
        elif self.contexts_items_features is not None:
            base_num_items = self.contexts_items_features[0].shape[1]
        elif self.contexts_items_availabilities is not None:
            base_num_items = self.contexts_items_availabilities.shape[1]
        else:
            raise ValueError(
                "No items features, sessions items features or items availabilities are defined"
            )
        self.base_num_items = base_num_items

        if self.fixed_items_features is not None:
            for items_feature in self.fixed_items_features:
                if items_feature.shape[0] != base_num_items:
                    raise ValueError(f"shapes are (f{items_feature.shape[0]}, {base_num_items})")

        if self.contexts_items_features is not None:
            for sessions_items_feature in self.contexts_items_features:
                if sessions_items_feature.shape[1] != base_num_items:
                    raise ValueError(
                        f"shapes are (f{sessions_items_feature.shape[1]}, {base_num_items})"
                    )
        if self.contexts_items_availabilities is not None:
            if self.contexts_items_availabilities.shape[1] != base_num_items:
                raise ValueError(
                    f"shapes are (f{self.contexts_items_availabilities.shape[1]}, \
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
        base_num_sessions = len(self.choices)
        self.base_num_sessions = base_num_sessions

        if self.contexts_features is not None:
            for sessions_feature in self.contexts_features:
                if sessions_feature.shape[0] != base_num_sessions:
                    raise ValueError(
                        f"shapes are ({sessions_feature.shape[0]}, {base_num_sessions})"
                    )

        if self.contexts_items_features is not None:
            for sessions_items_feature in self.contexts_items_features:
                if sessions_items_feature.shape[0] != base_num_sessions:
                    raise ValueError(
                        f"shapes are: ({sessions_items_feature.shape[0]}, \
                                     {base_num_sessions})"
                    )
        if self.contexts_items_availabilities is not None:
            if self.contexts_items_availabilities.shape[0] != base_num_sessions:
                raise ValueError(
                    f"shapes are: ({self.contexts_items_availabilities.shape[0]}, \
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
        if self.fixed_items_features is not None:
            for item_feat in self.fixed_items_features:
                if np.issubdtype(item_feat[0].dtype, np.integer):
                    item_types.append(np.int32)
                else:
                    item_types.append(np.float32)

        for indexes, f_dict in self.fixed_items_features_map.items():
            sample_dtype = next(iter(f_dict.values())).get_storage_type()
            item_types[indexes] = sample_dtype
        return_types.append(tuple(item_types))

        session_types = []
        if self.contexts_features is not None:
            for sessions_feat in self.contexts_features:
                if np.issubdtype(sessions_feat[0].dtype, np.integer):
                    session_types.append(np.int32)
                else:
                    session_types.append(np.float32)
        for indexes, f_dict in self.contexts_features_map.items():
            sample_dtype = next(iter(f_dict.values())).get_storage_type()
            session_types[indexes] = sample_dtype
        return_types.append(tuple(session_types))

        session_item_types = []
        if self.contexts_items_features is not None:
            for session_item_feat in self.contexts_items_features:
                if np.issubdtype(session_item_feat[0].dtype, np.integer):
                    session_item_types.append(np.int32)
                else:
                    session_item_types.append(np.float32)
        for indexes, f_dict in self.contexts_items_features_map.items():
            sample_dtype = next(iter(f_dict.values())).get_storage_type()
            session_item_types[indexes] = sample_dtype
        return_types.append(tuple(session_item_types))
        return_types.append(np.float32)
        return_types.append(np.int32)

        return return_types

    def _check_names(self):
        """Verifies that the names given to features are consistent with the features themselves."""
        if self.fixed_items_features_names is not None:
            for name, features in zip(self.fixed_items_features_names, self.fixed_items_features):
                if name is not None:
                    if len(name) != features.shape[1]:
                        raise ValueError(
                            f"Specififed items_features_names has \
                    length {len(name)} while items_features has {features.shape[1]} elements"
                        )

        if self.contexts_features_names is not None:
            for name, features in zip(self.contexts_features_names, self.contexts_features):
                if name is not None:
                    if len(name) != features.shape[1]:
                        raise ValueError(
                            f"Specified sessions_features_names has \
                    length {len(name)} while sessions_features has {features.shape[1]} elements"
                        )

        if self.contexts_items_features_names is not None:
            for (
                name,
                features,
            ) in zip(self.contexts_items_features_names, self.contexts_items_features):
                if name is not None:
                    if len(name) != features.shape[2]:
                        raise ValueError(
                            f"Specified \
                        sessions_items_features_names has length {len(name)} while \
                        sessions_items_features has {features.shape[2]} elements"
                        )

    def __len__(self):
        """Returns length of the dataset e.g. total number of choices.

        Returns:
        -------
        int
            total number of choices
        """
        return len(self.choices)

    def get_n_items(self):
        """Method to access the total number of different items.

        Returns:
        -------
        int
            total number of different items
        """
        return self.base_num_items

    def get_n_choices(self):
        """Method to access the total number of different choices.

        Redundant with __len__ method.

        Returns:
        -------
        int
            total number of different choices
        """
        return len(self)

    @classmethod
    def _contexts_items_features_df_to_np(
        cls,
        df,
        items_index,
        contexts_index,
        features,
        items_id_column="item_id",
        contexts_id_column="contexts_id",
    ):
        """Builds contexts_items_features and contexts_items_availabilities from dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing all the features for each item and sessions
        items_index : list
            List of items
        contexts_index : list
            List of sessions
        features : list
            List of columns of df that represents the items_features (for sessions_items_features)
        items_id_column: str, optional
            Name of the column containing the item ids, default is "items_id"
        contexts_id_column: str, optional
            Name of the column containing the sessions ids, default is "contexts_id"

        Returns:
        -------
        np.ndarray of shape (n_choices, n_items, n_features)
            Corresponding contexts_items_features
        np.ndarray of shape (n_choices, n_items)
            Corresponding availabilities
        """
        try:
            features.remove("context_id")
        except ValueError:
            pass
        try:
            features.remove("item_id")
        except ValueError:
            pass

        contexts_items_features = []
        contexts_items_availabilities = []
        for sess in contexts_index:
            sess_df = df.loc[df[contexts_id_column] == sess]

            if len(sess_df) == len(items_index):
                sess_df = sess_df.T
                sess_df.columns = sess_df.loc[items_id_column]
                if features is not None:
                    contexts_items_features.append(sess_df[items_index].loc[features].T.values)
                contexts_items_availabilities.append(np.ones(len(items_index)).astype("float32"))
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
                contexts_items_features.append(sess_feats)
                contexts_items_availabilities.append(sess_av)

        if features is not None:
            sessions_items_features = (np.array(contexts_items_features),)
        else:
            sessions_items_features = None
        return sessions_items_features, np.array(contexts_items_availabilities)

    @classmethod
    def from_single_wide_df(
        cls,
        df,
        items_id,
        fixed_items_suffixes=None,
        contexts_features_columns=None,
        contexts_items_features_suffixes=None,
        contexts_items_availabilities_suffix=None,
        choices_column="choice",
        choice_mode="items_id",
    ):
        """Builds numpy arrays for ChoiceDataset from a single dataframe in wide format.

        Parameters
        ----------
        df : pandas.DataFrame
            dataframe in Wide format
        items_id : list
            List of items ids
        fixed_items_suffixes : list
            Suffixes of the columns of the dataframe that are item features, default is None
        contexts_features_suffixes : list
            Suffixes of the columns of the dataframe that are contexts features, default is None
        contexts_items_suffixes : list
            Suffixes of the columns of the dataframe that are context-item features, default is None
        contexts_items_availabilities_suffix: list
            Suffixes of the columns of the dataframe that are context-item availabilities,
        choice_column: str, optional
            Name of the column containing the choices, default is "choice"
        choice_mode: str, optional
            How choice is indicated in df, either "items_name" or "items_index",
            default is "items_id"

        Returns:
        -------
        ChoiceDataset
            corresponding ChoiceDataset
        """
        if fixed_items_suffixes is not None:
            fixed_items_features = {"item_id": []}
            for item in items_id:
                fixed_items_features["item_id"].append(item)
                for feature in fixed_items_suffixes:
                    feature_value = df[f"{feature}_{item}"].unique()
                    if len(feature_value) > 1:
                        raise ValueError(
                            f"More than one value for feature {feature} for item {item}"
                        )
                    fixed_items_features[feature] = (
                        fixed_items_features.get(feature, []),
                        +[feature_value],
                    )
            fixed_items_features = pd.DataFrame(fixed_items_features)
        else:
            fixed_items_features = None

        if contexts_features_columns is not None:
            contexts_features = df[contexts_features_columns]
        else:
            contexts_features = None

        if contexts_items_features_suffixes is not None:
            contexts_items_features = []
            for item in items_id:
                columns = [f"{item}_{feature}" for feature in contexts_items_features_suffixes]
                for col in columns:
                    if col not in df.columns:
                        print(
                            f"Column {col} was not in DataFrame,\
                            dummy creation of the feature with zeros."
                        )
                        df[col] = 0
                contexts_items_features.append(df[columns].to_numpy())
            contexts_items_features = np.stack(contexts_items_features, axis=1)
        else:
            contexts_items_features = None

        if contexts_items_availabilities_suffix is not None:
            if isinstance(contexts_items_availabilities_suffix, list):
                if not len(contexts_items_availabilities_suffix) == len(items_id):
                    raise ValueError(
                        "You have given a list of columns for availabilities."
                        "We consider that it is one for each item but lenght do not match"
                    )
                print("You have given a list of columns for availabilities.")
                print("We consider that it is one for each item")
                contexts_items_availabilities = df[contexts_items_availabilities_suffix].to_numpy()
            else:
                columns = [f"{item}_{contexts_items_availabilities_suffix}" for item in items_id]
                contexts_items_availabilities = df[columns].to_numpy()
        else:
            contexts_items_availabilities = None

        choices = df[choices_column]
        if choice_mode == "items_id":
            choices = np.squeeze([np.where(items_id == c)[0] for c in choices])

        return ChoiceDataset(
            fixed_items_features=fixed_items_features,
            contexts_features=contexts_features,
            contexts_items_features=contexts_items_features,
            contexts_items_features_names=contexts_items_features_suffixes,
            contexts_items_availabilities=contexts_items_availabilities,
            choices=choices,
        )

    @classmethod
    def from_single_long_df(
        cls,
        df,
        fixed_items_features_columns=None,
        contexts_features_columns=None,
        contexts_items_features_columns=None,
        items_id_column="item_id",
        contexts_id_column="context_id",
        choices_column="choice",
        choice_mode="items_id",
    ):
        """Builds numpy arrays for ChoiceDataset from a single dataframe in long format.

        Parameters
        ----------
        df : pandas.DataFrame
            dataframe in Long format
        fixed_items_features_columns : list
            Columns of the dataframe that are item features, default is None
        contexts_features_columns : list
            Columns of the dataframe that are contexts features, default is None
        contexts_items_features_columns : list
            Columns of the dataframe that are context-item features, default is None
        items_id_column: str, optional
            Name of the column containing the item ids, default is "items_id"
        contexts_id_column: str, optional
            Name of the column containing the sessions ids, default is "contexts_id"
        choices_column: str, optional
            Name of the column containing the choices, default is "choice"
        choice_mode: str, optional
            How choice is indicated in df, either "items_name" or "one_zero",
            default is "items_id"

        Returns:
        -------
        ChoiceDataset
            corresponding ChoiceDataset
        """
        # Ordering items and sessions by id
        items = np.sort(df[items_id_column].unique())
        sessions = np.sort(df[contexts_id_column].unique())

        if fixed_items_features_columns is not None:
            items_features = df[fixed_items_features_columns + [items_id_column]].drop_duplicates()
            items_features = items_features.set_index(items_id_column)
            items_features = (items_features.loc[items].to_numpy(),)

            items_features_columns = (fixed_items_features_columns,)
        else:
            items_features = None
            items_features_columns = None

        if contexts_features_columns is not None:
            contexts_features = df[
                contexts_features_columns + [contexts_id_column]
            ].drop_duplicates()
            contexts_features = contexts_features.set_index(contexts_id_column)
            contexts_features = (contexts_features.loc[sessions].to_numpy(),)

            contexts_features_columns = (contexts_features_columns,)
        else:
            contexts_features = None
            contexts_features_columns = None

        (
            contexts_items_features,
            contexts_items_availabilities,
        ) = cls._contexts_items_features_df_to_np(
            df,
            items_index=items,
            contexts_index=sessions,
            features=contexts_items_features_columns,
            items_id_column=items_id_column,
            contexts_id_column=contexts_id_column,
        )
        contexts_items_features_columns = (
            (contexts_items_features_columns,)
            if contexts_items_features_columns is not None
            else None
        )

        if choice_mode == "items_id":
            choices = df[[choices_column, contexts_id_column]].drop_duplicates(contexts_id_column)
            choices = choices.set_index(contexts_id_column)
            choices = choices.loc[sessions].to_numpy()
            # items is the value (str) of the item
            choices = np.squeeze([np.where(items == c)[0] for c in choices])
        elif choice_mode == "one_zero":
            choices = df[[items_id_column, choices_column, contexts_id_column]]
            choices = choices.loc[choices[choices_column] == 1]
            choices = choices.set_index(contexts_id_column)
            choices = (
                choices.loc[sessions][items_id_column]
                .map({k: v for v, k in enumerate(items)})
                .to_numpy()
            )
        else:
            raise ValueError(
                f"choice_mode {choice_mode} not recognized. Must be in ['items_id', 'one_zero']"
            )
        return ChoiceDataset(
            fixed_items_features=items_features,
            contexts_features=contexts_features,
            contexts_items_features=contexts_items_features,
            contexts_items_availabilities=contexts_items_availabilities,
            choices=choices,
            fixed_items_features_names=items_features_columns,
            contexts_features_names=contexts_features_columns,
            contexts_items_features_names=contexts_items_features_columns,
        )

    def save(self):
        """Method to save the dataset."""
        raise NotImplementedError

    def summary(self):
        """Method to display a summary of the dataset."""
        print("%=====================================================================%")
        print("%%% Summary of the dataset:")
        print("%=====================================================================%")
        print("Number of items:", self.get_n_items())
        print(
            "Number of choices:",
            len(self),
        )
        print("%=====================================================================%")
        if self.fixed_items_features is not None:
            print(" Fixed Items Features:")
            print(f" {sum([f.shape[1] for f in self.fixed_items_features])} items features")
            if self.fixed_items_features_names is not None:
                print(f" with names: {self.fixed_items_features_names}")
        else:
            print(" No items features registered")
        print("\n")

        if self.contexts_features is not None:
            print(" Contexts features:")
            print(f" {sum([f.shape[1] for f in self.contexts_features])} context features")
            if self.contexts_features_names is not None:
                print(f" with names: {self.contexts_features_names}")
        else:
            print(" No sessions features registered")
        print("\n")

        if self.contexts_items_features is not None:
            print(" Contexts Items features:")
            print(
                f""" {sum([f.shape[2] for f in self.contexts_items_features])} context
                 items features"""
            )
            if self.contexts_items_features_names is not None:
                print(f" with names: {self.contexts_items_features_names}")
        else:
            print(" No sessions items features registered")
        print("%=====================================================================%")
        return ""

    def get_choices_batch(self, choices_indexes, features=None):
        """Method to access data within the ListChoiceDataset from its index.

        One index corresponds to a choice within a session.

        Return order:
            - df_chosen_item, df of length batch_size
            - dfs_available_items, list of lentch batch_size of dfs of length n_available_items

        Parameters
        ----------
        choices_indexes : int or list of int or slice
            indexes of the choices (that will be mapped to choice & session indexes) to return
        features : list of str, optional
            list of features to return. None returns all of them, default is None.

        Returns:
        --------
        tuple of (array_like, )
            tuple of arrays containing the features of the different items
        tuple of (array_like, )
            tuple of arrays containing the features of the different contexts
        tuple of (array_like, )
            tuple of arrays containing the features of the different contexts_items
        array_like
            array containing the availabilities of the different items
        array_like
            array containing the choices (indexes of chosen items)
        """
        _ = features
        if isinstance(choices_indexes, list):
            if self.fixed_items_features is None:
                fixed_items_features = None
            else:
                fixed_items_features = list(
                    items_feature
                    # .astype(self._return_types[0][i])
                    for i, items_feature in enumerate(self.fixed_items_features)
                )

            if self.contexts_features is None:
                contexts_features = None
            else:
                contexts_features = list(
                    contexts_features[choices_indexes]
                    # .astype(self._return_types[1][i])
                    for i, contexts_features in enumerate(self.contexts_features)
                )
                # sessions_features were not given as a tuple, so we return do not return it
                # as a tuple

            if self.contexts_items_features is None:
                contexts_items_features = None
            else:
                contexts_items_features = list(
                    contexts_items_feature[choices_indexes]
                    # .astype(self._return_types[2][i])
                    for i, contexts_items_feature in enumerate(self.contexts_items_features)
                )

            if self.contexts_items_availabilities is None:
                contexts_items_availabilities = np.ones(
                    (len(choices_indexes), self.base_num_items)
                ).astype("float32")
            else:
                contexts_items_availabilities = self.contexts_items_availabilities[choices_indexes]
                # .astype(self._return_types[3])

            choices = self.choices[choices_indexes].astype(self._return_types[4])

            if len(self.fixed_items_features_map) > 0:
                mapped_features = []
                for tuple_index in np.sort(list(self.fixed_items_features_map.keys())):
                    feat_ind_min = 0
                    unstacked_feat = []
                    for feature_index in np.sort(
                        list(self.fixed_items_features_map[tuple_index].keys())
                    ):
                        unstacked_feat.append(
                            fixed_items_features[tuple_index][:, feat_ind_min:feature_index]
                        )
                        unstacked_feat.append(
                            self.fixed_items_features_map[tuple_index][feature_index].batch[
                                fixed_items_features[tuple_index][:, feature_index]
                            ]
                        )
                        feat_ind_min = feature_index + 1
                    mapped_features.append(np.concatenate(unstacked_feat, axis=1))

                fixed_items_features = mapped_features

            if len(self.contexts_features_map) > 0:
                mapped_features = []
                for tuple_index in np.sort(list(self.contexts_features_map.keys())):
                    feat_ind_min = 0
                    unstacked_feat = []
                    for feature_index in np.sort(
                        list(self.contexts_features_map[tuple_index].keys())
                    ):
                        unstacked_feat.append(
                            contexts_features[tuple_index][:, feat_ind_min:feature_index]
                        )
                        unstacked_feat.append(
                            self.contexts_features_map[tuple_index][feature_index].batch[
                                contexts_features[tuple_index][:, feature_index]
                            ]
                        )
                        feat_ind_min = feature_index + 1
                    mapped_features.append(np.concatenate(unstacked_feat, axis=1))

                contexts_features = mapped_features

            if len(self.contexts_items_features_map) > 0:
                mapped_features = []
                for tuple_index in np.sort(list(self.contexts_items_features_map.keys())):
                    feat_ind_min = 0
                    unstacked_feat = []
                    for feature_index in np.sort(
                        list(self.contexts_items_features_map[tuple_index].keys())
                    ):
                        unstacked_feat.append(
                            contexts_items_features[tuple_index][:, :, feat_ind_min:feature_index]
                        )
                        unstacked_feat.append(
                            self.contexts_items_features_map[tuple_index][feature_index].batch[
                                contexts_items_features[tuple_index][:, :, feature_index]
                            ]
                        )
                        feat_ind_min = feature_index + 1
                    mapped_features.append(np.concatenate(unstacked_feat, axis=2))

                contexts_items_features = mapped_features

            if fixed_items_features is not None:
                for i in range(len(fixed_items_features)):
                    fixed_items_features[i] = fixed_items_features[i].astype(
                        self._return_types[0][i]
                    )
                # items_features were not given as a tuple, so we return do not return it as a tuple
                if not self._return_items_features_tuple:
                    fixed_items_features = fixed_items_features[0]
                else:
                    fixed_items_features = tuple(fixed_items_features)

            if contexts_features is not None:
                for i in range(len(contexts_features)):
                    contexts_features[i] = contexts_features[i].astype(self._return_types[1][i])
                if not self._return_contexts_features_tuple:
                    contexts_features = contexts_features[0]
                else:
                    contexts_features = tuple(contexts_features)

            if contexts_items_features is not None:
                for i in range(len(contexts_items_features)):
                    contexts_items_features[i] = contexts_items_features[i].astype(
                        self._return_types[2][i]
                    )
                # sessions_items_features were not given as a tuple, so we return do not return
                # it as a tuple
                if not self._return_contexts_items_features_tuple:
                    contexts_items_features = contexts_items_features[0]
                else:
                    contexts_items_features = tuple(contexts_items_features)

            return (
                fixed_items_features,
                contexts_features,
                contexts_items_features,
                contexts_items_availabilities,
                choices,
            )

        if isinstance(choices_indexes, slice):
            return self.get_choice_batch(
                list(range(*choices_indexes.indices(self.choices.shape[0])))
            )

        ### New
        choice = self.choices[choices_indexes]

        # fif = self.fif ?
        if self.fixed_items_features is None:
            fixed_items_features = None
        else:
            fixed_items_features = list(
                items_feature for items_feature in self.fixed_items_features
            )

        if self.contexts_features is None:
            contexts_features = None
        else:
            contexts_features = list(
                contexts_feature[choices_indexes] for contexts_feature in self.contexts_features
            )

        if self.contexts_items_features is None:
            contexts_items_features = None
        else:
            contexts_items_features = list(
                contexts_items_feature[choices_indexes]
                for contexts_items_feature in self.contexts_items_features
            )

        if self.contexts_items_availabilities is None:
            contexts_items_availabilities = np.ones((self.base_num_items)).astype("float32")
        else:
            contexts_items_availabilities = self.contexts_items_availabilities[choices_indexes]

        if len(self.fixed_items_features_map) > 0:
            mapped_features = []
            for tuple_index in np.sort(list(self.fixed_items_features_map.keys())):
                feat_ind_min = 0
                unstacked_feat = []
                for feature_index in np.sort(
                    list(self.fixed_items_features_map[tuple_index].keys())
                ):
                    unstacked_feat.append(
                        fixed_items_features[tuple_index][:, feat_ind_min:feature_index]
                    )
                    unstacked_feat.append(
                        self.fixed_items_features_map[tuple_index][feature_index].batch[
                            fixed_items_features[tuple_index][:, feature_index]
                        ]
                    )
                    feat_ind_min = feature_index + 1
                mapped_features.append(np.concatenate(unstacked_feat, axis=1))

            fixed_items_features = mapped_features

        if len(self.contexts_features_map) > 0:
            mapped_features = []
            for tuple_index in np.sort(list(self.contexts_features_map.keys())):
                feat_ind_min = 0
                unstacked_feat = []
                for feature_index in np.sort(list(self.contexts_features_map[tuple_index].keys())):
                    unstacked_feat.append(
                        contexts_features[tuple_index][feat_ind_min:feature_index]
                    )
                    unstacked_feat.append(
                        self.contexts_features_map[tuple_index][feature_index].batch[
                            contexts_features[tuple_index][feature_index]
                        ]
                    )
                    feat_ind_min = feature_index + 1
                mapped_features.append(np.concatenate(unstacked_feat, axis=0))

            contexts_features = mapped_features

        if len(self.contexts_items_features_map) > 0:
            mapped_features = []
            for tuple_index in np.sort(list(self.contexts_items_features_map.keys())):
                feat_ind_min = 0
                unstacked_feat = []
                for feature_index in np.sort(
                    list(self.contexts_items_features_map[tuple_index].keys())
                ):
                    unstacked_feat.append(
                        contexts_items_features[tuple_index][:, feat_ind_min:feature_index]
                    )
                    unstacked_feat.append(
                        self.contexts_items_features_map[tuple_index][feature_index].batch[
                            contexts_items_features[tuple_index][:, feature_index]
                        ]
                    )
                    feat_ind_min = feature_index + 1
                mapped_features.append(np.concatenate(unstacked_feat, axis=1))

            contexts_items_features = mapped_features

        if fixed_items_features is not None:
            for i in range(len(fixed_items_features)):
                fixed_items_features[i] = fixed_items_features[i].astype(self._return_types[0][i])
            # items_features were not given as a tuple, so we return do not return it as a tuple
            if not self._return_items_features_tuple:
                fixed_items_features = fixed_items_features[0]
            else:
                fixed_items_features = tuple(fixed_items_features)

        if contexts_features is not None:
            for i in range(len(contexts_features)):
                contexts_features[i] = contexts_features[i].astype(self._return_types[1][i])
            if not self._return_contexts_features_tuple:
                contexts_features = contexts_features[0]
            else:
                contexts_features = tuple(contexts_features)

        if contexts_items_features is not None:
            for i in range(len(contexts_items_features)):
                contexts_items_features[i] = contexts_items_features[i].astype(
                    self._return_types[2][i]
                )
            # sessions_items_features were not given as a tuple, so we return do not return
            # it as a tuple
            if not self._return_contexts_items_features_tuple:
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

    def __getitem__(self, choices_indexes):
        """Method to create a sub-ChoiceDataset with only a subset of choices, from their indexes.

        Parameters
        ----------
        choices_indexes : np.ndarray
            indexes of the contexts / choices to keep, shape should be (num_choices,)

        Returns:
        -------
        ChoiceDataset
            ChoiceDataset with only the sessions indexed by indexes
        """
        if isinstance(choices_indexes, int):
            choices_indexes = [choices_indexes]
        elif isinstance(choices_indexes, slice):
            return self.__getitem__(list(range(*choices_indexes.indices(len(self.choices)))))

        try:
            if self.fixed_items_features[0] is None:
                fixed_items_features = None
            else:
                fixed_items_features = self.fixed_items_features
        except TypeError:
            fixed_items_features = self.fixed_items_features

        try:
            if self.contexts_features[0] is None:
                contexts_features = None
            else:
                contexts_features = tuple(
                    self.contexts_features[i][choices_indexes]
                    for i in range(len(self.contexts_features))
                )
        except TypeError:
            contexts_features = None

        try:
            if self.contexts_items_features[0] is None:
                contexts_items_features = None
            else:
                contexts_items_features = tuple(
                    self.contexts_items_features[i][choices_indexes]
                    for i in range(len(self.contexts_items_features))
                )
        except TypeError:
            contexts_items_features = None

        try:
            if self.fixed_items_features_names[0] is None:
                fixed_items_features_names = None
            else:
                fixed_items_features_names = self.fixed_items_features_names
        except TypeError:
            fixed_items_features_names = None
        try:
            if self.contexts_features_names[0] is None:
                contexts_features_names = None
            else:
                contexts_features_names = self.contexts_features_names
        except TypeError:
            contexts_features_names = None
        try:
            if self.contexts_items_features_names[0] is None:
                contexts_items_features_names = None
            else:
                contexts_items_features_names = self.contexts_items_features_names
        except TypeError:
            contexts_items_features_names = None

        try:
            contexts_items_availabilities = self.contexts_items_availabilities[choices_indexes]
        except TypeError:
            contexts_items_availabilities = None
        return ChoiceDataset(
            fixed_items_features=fixed_items_features,
            contexts_features=contexts_features,
            contexts_items_features=contexts_items_features,
            contexts_items_availabilities=contexts_items_availabilities,
            choices=[self.choices[i] for i in choices_indexes],
            fixed_items_features_names=fixed_items_features_names,
            contexts_features_names=contexts_features_names,
            contexts_items_features_names=contexts_items_features_names,
            features_by_ids=self.features_by_ids,
        )

    @property
    def batch(self):
        """Indexer. Corresponds to get_choice_batch, but with [] logic."""
        return self.indexer

    def iter_batch(self, batch_size, shuffle=False, sample_weight=None):
        """Iterates over dataset return batches of length batch_size.

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
        if batch_size == -1:
            batch_size = len(self)
        # Get indexes for each choice
        num_choices = len(self)
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
            list of booleans of length self.get_n_contexts() to filter contexts.
            True to keep, False to discard.
        """
        indexes = [i for i, keep in enumerate(bool_list) if keep]
        return self[indexes]

    def get_n_fixed_items_features(self):
        """Method to access the number of fixed items features.

        Returns:
        -------
        int
            number of fixed items features
        """
        if self.fixed_items_features is not None:
            n_features = 0
            for fixed_features in self.fixed_items_features:
                n_features += fixed_features.shape[1]
            return n_features
        return 0

    def get_n_contexts_features(self):
        """Method to access the number of contexts features.

        Returns:
        -------
        int
            number of fixed items features
        """
        if self.contexts_features is not None:
            n_features = 0
            for context_features in self.contexts_features:
                n_features += context_features.shape[1]
            return n_features
        return 0

    def get_n_contexts_items_features(self):
        """Method to access the number of context items features.

        Returns:
        -------
        int
            number of fixed items features
        """
        if self.contexts_items_features is not None:
            n_features = 0
            for contexts_items_features in self.contexts_items_features:
                n_features += contexts_items_features.shape[2]
            return n_features
        return 0
