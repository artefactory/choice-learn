"""Main classes to handle assortment data."""

import logging

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
        shared_features_by_choice=None,  # as many as choices.  values or ids (look at key)
        items_features_by_choice=None,
        available_items_by_choice=None,
        features_by_ids=[],  # list of (name, FeaturesStorage)
        shared_features_by_choice_names=None,
        items_features_by_choice_names=None,
    ):
        """Build the ChoiceDataset.

        Parameters
        ----------
        choices: list or np.ndarray
            list of chosen items indexes
        shared_features_by_choice : tuple of (array_like, )
            matrix of shape (num_choices, num_shared_features) containing the features of the
            different choices that are common to all items (e.g. store features,
            customer features, etc...)
        items_features_by_choice : tuple of (array_like, ), default is None
            matrix of shape (num_choices, num_items, num_items_features)
            containing the features
            of the items that change over time (e.g. price, promotion, etc...), default is None
        available_items_by_choice : array_like
            matrix of shape (num_choices, num_items) containing the availabilities of the items
            over the different choices, default is None
        features_by_ids : list of (name, FeaturesStorage)
            List of Storage objects. Their name must correspond to a feature name
            among shared_features_by_choice or items_features_by_choice
            and their ids must match to those features values. Default is []
        shared_features_by_choice_names : tuple of (array_like, )
            list of names of the shared_features_by_choice, default is None
            Shapes must match with shared_features_by_choice
        items_features_by_choice_names : tuple of (array_like, )
            list of names of the items_features_by_choice, default is None
            Shapes must match with items_features_by_choice
        """
        if choices is None:
            # Done to keep a logical order of arguments, and has logic: choices have to be specified
            raise ValueError("Choices must be specified, got None")

        # --------- [ Handling features type given as tuples or not ] --------- #

        # If shared_features_by_choice is not given as tuple, transform it internally as a tuple
        # A bit longer because can be None and need to also handle names
        if shared_features_by_choice is not None:
            if not isinstance(shared_features_by_choice, tuple):
                self._return_shared_features_by_choice_tuple = False
                if shared_features_by_choice_names is not None:
                    if len(shared_features_by_choice[0]) != len(shared_features_by_choice_names):
                        raise ValueError(
                            f"""Number of features given does not match
                                         number of features names given:
                                           {len(shared_features_by_choice[0])} and
                                            {len(shared_features_by_choice_names)}"""
                        )
                else:
                    logging.warning(
                        """Shared Features Names were not provided, will not be able to
                                    fit models needing them such as Conditional Logit."""
                    )

                shared_features_by_choice_names = (shared_features_by_choice_names,)
                shared_features_by_choice = (shared_features_by_choice,)

            # choices_features is already a tuple, names are given, checking consistency
            else:
                self._return_shared_features_by_choice_tuple = True
                if shared_features_by_choice_names is not None:
                    for sub_k, (sub_features, sub_names) in enumerate(
                        zip(shared_features_by_choice, shared_features_by_choice_names)
                    ):
                        if len(sub_features[0]) != len(sub_names):
                            raise ValueError(
                                f"""{sub_k}-th given shared_features_by_choice and
                                shared_features_by_choice_names shapes do not match:
                                {len(sub_features[0])} and {len(sub_names)}."""
                            )
                # In this case names are missing, still transform it as a tuple
                else:
                    logging.warning(
                        """Shared Features Names were not provided, will not be able to
                                    fit models needing them such as Conditional Logit."""
                    )
                    shared_features_by_choice_names = (None,) * len(shared_features_by_choice)
        else:
            self._return_shared_features_by_choice_tuple = False

        # If items_features_by_choice is not given as tuple, transform it internally as a tuple
        # A bit longer because can be None and need to also handle names

        if not isinstance(items_features_by_choice, tuple) and items_features_by_choice is not None:
            self._return_items_features_by_choice_tuple = False
            if items_features_by_choice_names is not None:
                if len(items_features_by_choice[0][0]) != len(items_features_by_choice_names):
                    raise ValueError(
                        f"""Number of items_features_by_choice given does not match
                                     number of items_features_by_choice_names given:
                                     {len(items_features_by_choice[0][0])} and
                                     {len(items_features_by_choice_names)}"""
                    )
            else:
                logging.warning(
                    """Items Features Names were not provided, will not be able to
                                fit models needing them such as Conditional Logit."""
                )
            items_features_by_choice = (items_features_by_choice,)
            items_features_by_choice_names = (items_features_by_choice_names,)

        # items_features_by_choice is already a tuple, names are given, checking consistency
        elif items_features_by_choice is not None and items_features_by_choice_names is not None:
            for sub_k, (sub_features, sub_names) in enumerate(
                zip(items_features_by_choice, items_features_by_choice_names)
            ):
                # Split if feature is full FeaturesStorage
                if np.array(sub_features).ndim == 1:
                    # check features_by_ids
                    logging.info(
                        "feature of dimension 1 detected -  a FeatureByIDs MUST be provided"
                    )
                    for fbid in features_by_ids:
                        if fbid.name == sub_names[0]:
                            logging.info("FeatureByIDs found")
                            break
                    else:
                        raise ValueError(
                            """FeatureByIDs must be provided when items_features\
                                of shape (n_choices, 1) is given."""
                        )

                elif len(sub_features[0][0]) != len(sub_names):
                    raise ValueError(
                        f"""{sub_k}-th given items_features_by_choice with names
                        {sub_names} and
                        items_features_by_choice_names shapes do not match:
                        {len(sub_features[0][0])} and {len(sub_names)}."""
                    )
            self._return_items_features_by_choice_tuple = True

        # In this case names are missing, still transform it as a tuple
        elif items_features_by_choice is not None:
            logging.warning(
                """Items Features Names were not provided, will not be able to
                            fit models needing them such as Conditional Logit."""
            )
            self._return_items_features_by_choice_tuple = True
            items_features_by_choice_names = (None,) * len(items_features_by_choice)

        else:
            self._return_items_features_by_choice_tuple = False

        # --------- [Normalizing features types (DataFrame, List, etc...) -> np.ndarray] --------- #
        #
        # Part of this code is for handling features given as pandas.DataFrame
        # Basically it transforms them to be internally stocked as np.ndarray and keep columns
        # names as features names

        # Handling shared features
        if shared_features_by_choice is not None:
            for i, feature in enumerate(shared_features_by_choice):
                if isinstance(feature, pd.DataFrame):
                    # Ordering choices by id ?
                    if "choice_id" in feature.columns:
                        feature = feature.set_index("choice_id")
                    shared_features_by_choice = (
                        shared_features_by_choice[:i]
                        + (feature.loc[np.sort(feature.index)].to_numpy(),)
                        + shared_features_by_choice[i + 1 :]
                    )
                    if shared_features_by_choice_names[i] is not None:
                        logging.warning(
                            f"""shared_features_by_choice_names {shared_features_by_choice_names[i]}
                            were given. They will be overwritten with DF columns names:
                            {feature.columns}"""
                        )
                    shared_features_by_choice_names = (
                        shared_features_by_choice_names[:i]
                        + (feature.columns,)
                        + shared_features_by_choice_names[i + 1 :]
                    )
                elif isinstance(feature, list):
                    shared_features_by_choice = (
                        shared_features_by_choice[:i]
                        + (np.array(feature),)
                        + shared_features_by_choice[i + 1 :]
                    )
        # Handling items_features_by_choice
        if items_features_by_choice is not None:
            for i, feature in enumerate(items_features_by_choice):
                if isinstance(feature, pd.DataFrame):
                    # Ordering choices by id ?
                    # TODO: here choice_id was context_id > make sure this change does not affect
                    # some code somewhere
                    if "choice_id" in feature.columns:
                        if "item_id" in feature.columns:
                            all_items = np.sort(feature.item_id.unique())
                            feature_array = []
                            temp_availabilities = []
                            for sess in np.sort(feature.choice_id.unique()):
                                sess_df = feature.loc[feature.choice_id == sess]
                                sess_df = sess_df[
                                    sess_df.columns.difference(["choice_id"])
                                ].set_index("item_id")
                                sess_feature = []
                                choice_availabilities = []
                                for item in all_items:
                                    if item in sess_df.index:
                                        sess_feature.append(sess_df.loc[item].to_numpy())
                                        choice_availabilities.append(1)
                                    else:
                                        sess_feature.append(np.zeros(len(sess_df.columns)))
                                        choice_availabilities.append(0)
                                feature_array.append(sess_feature)
                                temp_availabilities.append(choice_availabilities)

                            items_features_by_choice = (
                                items_features_by_choice[:i]
                                + (np.stack(feature_array, axis=0),)
                                + items_features_by_choice[i + 1 :]
                            )

                            if items_features_by_choice_names[i] is not None:
                                logging.warning(
                                    f"""items_features_by_choice_names
                                    {items_features_by_choice_names[i]} were given. They will be
                                    overwritten with DF columns names: {feature.columns}"""
                                )
                            items_features_by_choice_names = (
                                items_features_by_choice_names[:i]
                                + (sess_df.columns,)
                                + items_features_by_choice_names[i + 1 :]
                            )
                            if (
                                available_items_by_choice is None
                                and len(np.unique(temp_availabilities)) > 1
                            ):
                                logging.info(
                                    f"""available_items_by_choice were not given and computed from
                                    {i}-th items_features_by_choice."""
                                )
                                available_items_by_choice = np.array(temp_availabilities)
                        else:
                            feature = feature.set_index("choice_id")
                            items_features_by_choice = (
                                items_features_by_choice[:i]
                                + (feature.loc[np.sort(feature.index)].to_numpy(),)
                                + items_features_by_choice[i + 1 :]
                            )
                            if items_features_by_choice_names[i] is not None:
                                logging.warning(
                                    f"""items_features_by_choice_names
                                    {items_features_by_choice_names[i]} were given. They will be
                                    overwritten with DF columns names: {feature.columns}"""
                                )
                            items_features_by_choice_names = (
                                items_features_by_choice_names[:i]
                                + (feature.columns,)
                                + items_features_by_choice_names[i + 1 :]
                            )
                    else:
                        raise ValueError(
                            f"""A 'choice_id' column must be integrated in {i}-th
                            items_features_by_choice DF, in order to identify each choice."""
                        )
                elif isinstance(feature, list):
                    items_features_by_choice = (
                        items_features_by_choice[:i]
                        + (np.array(feature),)
                        + items_features_by_choice[i + 1 :]
                    )
        # Handling available_items_by_choice
        if available_items_by_choice is not None:
            if isinstance(available_items_by_choice, list):
                available_items_by_choice = np.array(
                    available_items_by_choice,
                    dtype=object,  # Are you sure ?
                )
            elif isinstance(available_items_by_choice, pd.DataFrame):
                if "choice_id" in available_items_by_choice.columns:
                    if "item_id" in available_items_by_choice.columns:
                        av_array = []
                        for sess in np.sort(available_items_by_choice.choice_id):
                            sess_df = available_items_by_choice.loc[
                                available_items_by_choice.choice_id == sess
                            ]
                            sess_df = sess_df.set_index("item_id")
                            av_array.append(sess_df.loc[np.sort(sess_df.index)].to_numpy())
                        available_items_by_choice = np.array(av_array)
                    else:
                        feature = feature.set_index("choice_id")
                        available_items_by_choice = available_items_by_choice.loc[
                            np.sort(feature.index)
                        ].to_numpy()
                else:
                    logging.info(
                        "No 'choice_id' column found in available_items_by_choice DF, using index"
                    )
                    available_items_by_choice = available_items_by_choice.to_numpy()

        # Handling choices
        # Choices must then be given as the name of the chosen item
        # Items are sorted by name and attributed an index
        # TODO: Keep items_id as an attribute ?
        if isinstance(choices, pd.DataFrame):
            # Ordering choices by id
            if "choice_id" in choices.columns:
                choices = choices.set_index("choice_id")
            choices = choices.loc[np.sort(choices.index)]
            items = np.sort(np.unique(choices.choice))
            # items is the value (str) of the item
            choices = [np.where(items == c)[0] for c in choices.choice]
            choices = np.squeeze(choices)
        elif isinstance(choices, pd.Series):
            choices = choices.to_numpy()
        elif isinstance(choices, list):
            choices = np.array(choices)

        # Setting attributes of ChoiceDataset
        self.shared_features_by_choice = shared_features_by_choice
        self.items_features_by_choice = items_features_by_choice
        self.available_items_by_choice = available_items_by_choice
        self.choices = choices

        for fid in features_by_ids:
            if not isinstance(fid, Storage):
                raise ValueError("FeaturesByID must be Storage object")
        self.features_by_ids = features_by_ids

        self.shared_features_by_choice_names = shared_features_by_choice_names
        self.items_features_by_choice_names = items_features_by_choice_names

        # What about typing ? should builf after check to change it ?
        (
            self.shared_features_by_choice_map,
            self.items_features_by_choice_map,
        ) = self._build_features_by_ids()
        self.check_features_by_ids()

        # self.n_choices = len(self.choices)

        # Different consitency checks to ensure everything is coherent
        self._check_dataset()  # Should handle alone if np.arrays are squeezed
        self._return_types = self._check_types()
        self._check_names()
        # Build .iloc method
        self.indexer = ChoiceDatasetIndexer(self)

    def _build_features_by_ids(self):
        """Build mapping function.

        Those mapping functions are so that at indexing,
        the features are rebuilt with the features by id.

        Returns
        -------
        tuple
            indexes and features_by_id of shared_features_by_choice
        tuple
            indexes and features_by_id of items_features_by_choice
        """
        if len(self.features_by_ids) == 0:
            return {}, {}

        found_av_fid = False
        for fid in self.features_by_ids:
            if fid.name == "available_items_by_choice":
                logging.warning("FeaturesStorage for available_items_by_choice detected.")
                if self.available_items_by_choice is None:
                    raise ValueError(
                        """Cannot provide availabilities_by_choice as\
                        features_by_ids without indexes."""
                    )
                self.available_items_by_choice = (fid, self.available_items_by_choice)
                found_av_fid = True

        if (
            self.shared_features_by_choice_names is None
            and self.items_features_by_choice_names is None
            and not found_av_fid
        ):
            raise ValueError(
                """"Features names are needed to match id columns with features_by_id,
                and none were given. It is possible to either give them as arguments or to pass
                features as pandas.DataFrames."""
            )
        if (
            isinstance(self.shared_features_by_choice_names, tuple)
            and self.shared_features_by_choice_names[0] is None
            and isinstance(self.items_features_by_choice_names, tuple)
            and self.items_features_by_choice_names[0] is None
            and not found_av_fid
        ):
            raise ValueError(
                """"Features names are needed to match id columns with features_by_id,
                and none were given. It is possible to either give them as arguments or to pass
                features as pandas.DataFrames."""
            )

        shared_features_map = {}
        items_features_map = {}

        if self.shared_features_by_choice_names is not None:
            for i, feature in enumerate(self.shared_features_by_choice_names):
                if feature is not None:
                    for j, column_name in enumerate(feature):
                        for feature_by_id in self.features_by_ids:
                            if column_name == feature_by_id.name:
                                index_dict = shared_features_map.get(i, {})
                                index_dict[j] = feature_by_id
                                shared_features_map[i] = index_dict
                                logging.info(
                                    f"""Feature by ID found for shared_features_by_choice:
                                    {feature_by_id.name}"""
                                )

                                # We test a subset of IDs
                                test_values = self.shared_features_by_choice[i][:, j][
                                    np.arange(0, len(self.choices), 10)
                                ]
                                try:
                                    for val in test_values:
                                        feature_by_id.batch[val]
                                except KeyError:
                                    raise ValueError(
                                        f"""Key {val} in Shared Feature {column_name}
                                                     not found in {feature_by_id.name}"""
                                    )

        if self.items_features_by_choice_names is not None:
            for i, feature in enumerate(self.items_features_by_choice_names):
                if feature is not None:
                    for k, column_name in enumerate(feature):
                        for feature_by_id in self.features_by_ids:
                            if column_name == feature_by_id.name:
                                index_dict = items_features_map.get(i, {})
                                index_dict[k] = feature_by_id
                                items_features_map[i] = index_dict
                                logging.info(
                                    f"""Feature by ID found for items_features_by_choice:
                                    {feature_by_id.name}"""
                                )

                                # We test a subset of the IDs
                                if self.items_features_by_choice[i].ndim == 1:
                                    test_values = self.items_features_by_choice[i][
                                        np.arange(0, len(self.choices), 10)
                                    ]
                                else:
                                    test_values = self.items_features_by_choice[i][:, :, k][
                                        np.arange(0, len(self.choices), 10)
                                    ]
                                try:
                                    for val in test_values:
                                        feature_by_id.batch[val]
                                except KeyError:
                                    raise ValueError(
                                        f"""Key {val} in Items Feature {column_name}
                                                     not found in {feature_by_id.name}"""
                                    )
        # Checking number of found features_by_id
        num_ff_maps = sum([len(val) for val in shared_features_map.values()])
        num_if_maps = sum([len(val) for val in items_features_map.values()])
        if num_ff_maps + num_if_maps != len(self.features_by_ids) - found_av_fid:
            raise ValueError("Some features_by_ids were not matched with features_names.")

        return shared_features_map, items_features_map

    def check_features_by_ids(self, batch_size=128):
        """Verify that all IDs given in features exist in the corresponding FeaturesStorage.

        Parameters
        ----------
        batch_size : int, optional
            batch size used to sample the FeaturesStorage, by default 128

        Returns
        -------
        bool
            Whether the check was successful or not
        """
        for index_1 in self.shared_features_by_choice_map:
            for index_2 in self.shared_features_by_choice_map[index_1]:
                all_values = np.unique(self.shared_features_by_choice[index_1][:, index_2])
                for i in range(len(all_values) // batch_size + 1):
                    self.shared_features_by_choice_map[index_1][index_2].batch[
                        all_values[i * batch_size : (i + 1) * batch_size]
                    ]

        for index_1 in self.items_features_by_choice_map:
            for index_2 in self.items_features_by_choice_map[index_1]:
                if self.items_features_by_choice[index_1].ndim == 1:
                    all_values = np.unique(self.items_features_by_choice[index_1])
                else:
                    all_values = np.unique(self.items_features_by_choice[index_1][:, :, index_2])
                for i in range(len(all_values) // batch_size + 1):
                    self.items_features_by_choice_map[index_1][index_2].batch[
                        all_values[i * batch_size : (i + 1) * batch_size]
                    ]
        logging.info("Features by ID checked: all IDs have values")
        return True

    def _check_dataset(self):
        """Verify that the shapes of the different features are consistent.

        Particularly:
            - Over number of items
            - Over number of choices
        > Verifies that the choices have coherent values.
        """
        self._check_num_items_shapes()
        self._check_num_sessions_shapes()
        self._check_choices_coherence()

    def _check_num_items_shapes(self):
        """Verify that the shapes of the different features are consistent over number of items.

        Particularly:
            - items_features_by_choice
            - available_items_by_choice
        > Sets the argument base_num_items.
        """
        if self.items_features_by_choice is not None:
            if self.items_features_by_choice[0].ndim == 1:
                # items_features_by_choice fully integrated into a FeaturesStorage
                base_num_items = (
                    next(iter(next(iter(self.items_features_by_choice_map.values())).values()))
                    .get_element_from_index(0)
                    .shape[0]
                )
            else:
                base_num_items = self.items_features_by_choice[0].shape[1]
        elif self.available_items_by_choice is not None:
            if isinstance(self.available_items_by_choice, tuple):
                base_num_items = self.available_items_by_choice[0].get_element_by_index(0).shape[0]
            else:
                base_num_items = self.available_items_by_choice.shape[1]
        else:
            logging.warning(
                "No items features or items availabilities are defined. Using max value of choices"
            )
            base_num_items = len(np.unique(self.choices))

        logging.info(f"Number of detected items is {base_num_items}")
        self.base_num_items = base_num_items

        if self.items_features_by_choice is not None:
            for k, items_feature in enumerate(self.items_features_by_choice):
                if items_feature.ndim == 1:
                    batch = self.items_features_by_choice_map[k][0].batch[[0, 1]]
                    if batch.shape[1] != base_num_items:
                        raise ValueError(
                            f"""{k}-th 'items_features_by_choice' shape does not match the
                            detected number of items:
                            ({items_feature.shape[1]} and {base_num_items})"""
                        )
                else:
                    if items_feature.shape[1] != base_num_items:
                        raise ValueError(
                            f"""{k}-th 'items_features_by_choice' shape does not match the
                            detected number of items:
                            ({items_feature.shape[1]} and {base_num_items})"""
                        )
        if self.available_items_by_choice is not None:
            if isinstance(self.available_items_by_choice, tuple):
                extract = self.available_items_by_choice[0].batch[
                    self.available_items_by_choice[1][0]
                ]
            else:
                extract = self.available_items_by_choice[0]
            if len(extract) != base_num_items:
                raise ValueError(
                    f"""'available_items_by_choice' shape does not match the
                        detected number of items: ({len(extract)}
                        and {base_num_items})"""
                )

    def _check_num_sessions_shapes(self):
        """Verify that the shapes of the different features are consistent over nb of sessions.

        Particularly:
            - shared_features_by_choice
            - items_features_by_choice
            - available_items_by_choice
        > Sets self.base_num_choices argument.
        """
        self.n_choices = len(self.choices)

        if self.shared_features_by_choice is not None:
            for k, feature in enumerate(self.shared_features_by_choice):
                if feature.shape[0] != self.n_choices:
                    raise ValueError(
                        f"""{k}-th 'shared_features_by_choice' shape does not match
                         the number of choices detected: ({feature.shape[0]}, {self.n_choices})"""
                    )

        if self.items_features_by_choice is not None:
            for k, items_feature in enumerate(self.items_features_by_choice):
                if items_feature.shape[0] != self.n_choices:
                    raise ValueError(
                        f"""{k}-th 'items_features_by_choice' shape does not match
                         the number of choices detected: ({items_feature.shape[0]} and
                         {self.n_choices})"""
                    )
        if self.available_items_by_choice is not None:
            if isinstance(self.available_items_by_choice, tuple):
                if len(self.available_items_by_choice[1]) != self.n_choices:
                    raise ValueError(
                        f"""Given 'available_items_by_choice' shape does not match
                        the number of choices detected: ({len(self.available_items_by_choice[1])}
                        and {self.n_choices})"""
                    )
            else:
                if self.available_items_by_choice.shape[0] != self.n_choices:
                    raise ValueError(
                        f"""Given 'available_items_by_choice' shape does not match
                            the number of choices detected:
                            ({self.available_items_by_choice.shape[0]}
                            and {self.n_choices})"""
                    )

    def _check_choices_coherence(self):
        """Verify that the choices are coherent with the nb of items present in other features.

        Particularly:
            - There is no choice index higher than detected number of items
            - All items are present at least once in the choices
        """
        if np.max(self.choices) > self.base_num_items - 1:
            msg = f"Choices values not coherent with number of items given in features.  \
            In particular, max value of choices is {np.max(self.choices)} while number of  \
            items is {self.base_num_items}"
            raise ValueError(msg)

        unique_choices = set(np.unique(self.choices).flatten())
        missing_choices = set(np.arange(start=0, stop=self.base_num_items, step=1)) - unique_choices
        if len(missing_choices) > 0:
            logging.warning(f"Some choices never happen in the dataset: {missing_choices}")

    def _check_types(self):
        """Check types of elements and store it in order to return right types.

        Particularly:
            - Either int32 or float32 consistently for features.
                float32 is to be preferred unless One-Hot encoding is used.
            - float32 for available_items_by_choice
            - int32 for choices
        """
        return_types = []

        shared_features_types = []
        if self.shared_features_by_choice is not None:
            for feature in self.shared_features_by_choice:
                if np.issubdtype(feature[0].dtype, np.integer):
                    shared_features_types.append(np.int32)
                else:
                    shared_features_types.append(np.float32)
        for indexes, f_dict in self.shared_features_by_choice_map.items():
            sample_dtype = next(iter(f_dict.values())).get_storage_type()
            shared_features_types[indexes] = sample_dtype
        return_types.append(tuple(shared_features_types))

        items_features_types = []
        if self.items_features_by_choice is not None:
            for items_feat in self.items_features_by_choice:
                if np.issubdtype(items_feat[0].dtype, np.integer):
                    items_features_types.append(np.int32)
                else:
                    items_features_types.append(np.float32)
        for indexes, f_dict in self.items_features_by_choice_map.items():
            sample_dtype = next(iter(f_dict.values())).get_storage_type()
            items_features_types[indexes] = sample_dtype
        return_types.append(tuple(items_features_types))
        return_types.append(np.float32)
        return_types.append(np.int32)

        return return_types

    def _check_names(self):
        """Verify that names and features shapes are consistent with each other."""
        if self.shared_features_by_choice_names is not None:
            for k, (name, features) in enumerate(
                zip(self.shared_features_by_choice_names, self.shared_features_by_choice)
            ):
                if name is not None:
                    if len(name) != features.shape[1]:
                        raise ValueError(
                            f"""Specified {k}th shared_features_by_choice_name has length
                            {len(name)}while shared_features_by_choice has
                            {features.shape[1]} elements."""
                        )

        if self.items_features_by_choice_names is not None:
            for k, (
                name,
                features,
            ) in enumerate(zip(self.items_features_by_choice_names, self.items_features_by_choice)):
                if name is not None:
                    if features.ndim > 1:
                        if len(name) != features.shape[2]:
                            raise ValueError(
                                f"Specified {k}th\
                            items_features_by_choice_names has length {len(name)} while \
                            items_features_by_choice has {features.shape[2]} elements."
                            )
                    elif len(name) != 1:
                        raise ValueError(
                            f"Specified {k}th items_features_by_choice_names has length {len(name)}\
                            while items_features_by_choice has 1 element."
                        )

    def __len__(self):
        """Return length of the dataset e.g. total number of choices.

        Returns
        -------
        int
            total number of choices
        """
        return len(self.choices)

    def __str__(self):
        """Return short representation of ChoiceDataset.

        Returns
        -------
        str
            short representation of ChoiceDataset
        """
        template = """First choice is: Shared Features by choice: {}\n
                      Items Features by choice: {}\nAvailable items by choice: {}\n
                      Choices: {}"""
        return template.format(
            self.batch[0][0], self.batch[0][1], self.batch[0][2], self.batch[0][3]
        )

    def get_n_items(self):
        """Access the total number of different items.

        Returns
        -------
        int
            total number of different items
        """
        return self.base_num_items

    def get_n_choices(self):
        """Access the total number of different choices.

        Redundant with __len__ method.

        Returns
        -------
        int
            total number of different choices
        """
        return len(self)

    @classmethod
    def _long_df_to_items_features_array(
        cls,
        df,
        features,
        items_id_column="item_id",
        choices_id_column="choice_id",
        items_index=None,
        choices_index=None,
    ):
        """Build items_features_by_choice and available_items_by_choice from dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing all the features for each item and sessions
        items_index : list
            List of items identifiers
        choices_index : list
            List of unique identifiers of choices
        features : list
            List of columns of df that represents the items_features (for sessions_items_features)
        items_id_column: str, optional
            Name of the column containing the item ids, default is "items_id"
        choices_id_column: str, optional
            Name of the column containing the choices ids, default is "choice_id"

        Returns
        -------
        np.ndarray of shape (n_choices, n_items, n_features)
            Corresponding items_features_by_choice
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

        if choices_index is None:
            choices_index = np.sort(df[choices_id_column].unique().to_numpy())
        if items_index is None:
            items_index = np.sort(df[items_id_column].unique().to_numpy())

        items_features_by_choice = []
        available_items_by_choice = []
        for sess in choices_index:
            sess_df = df.loc[df[choices_id_column] == sess]

            # All items were available for the choice
            if len(sess_df) == len(items_index):
                sess_df = sess_df.T
                sess_df.columns = sess_df.loc[items_id_column]
                if features is not None:
                    items_features_by_choice.append(sess_df[items_index].loc[features].T.values)
                available_items_by_choice.append(np.ones(len(items_index)).astype("float32"))

            # Some items were not available for the choice
            else:
                sess_feats = []
                sess_av = []
                for item in items_index:
                    item_df = sess_df.loc[sess_df[items_id_column] == item]
                    if len(item_df) > 0:
                        if features is not None:
                            sess_feats.append(item_df[features].to_numpy()[0])
                        sess_av.append(1.0)
                    else:
                        if features is not None:
                            # Unavailable items features are filled with zeros
                            sess_feats.append(np.zeros(len(features)))
                        sess_av.append(0.0)
                items_features_by_choice.append(sess_feats)
                available_items_by_choice.append(sess_av)

        if features is not None:
            items_features_by_choice = np.array(items_features_by_choice)
        else:
            items_features_by_choice = None
        return items_features_by_choice, np.array(available_items_by_choice).astype("float32")

    @classmethod
    def from_single_wide_df(
        cls,
        df,
        items_id,
        shared_features_columns=None,
        items_features_suffixes=None,
        items_features_prefixes=None,
        available_items_suffix=None,
        available_items_prefix=None,
        delimiter="_",
        choices_column="choice",
        choice_format="items_id",
    ):
        """Build numpy arrays for ChoiceDataset from a single dataframe in wide format.

        Parameters
        ----------
        df : pandas.DataFrame
            dataframe in Wide format
        items_id : list
            List of items ids
        shared_features_columns : list, optional
            List of columns of the dataframe that are shared_features_by_choice, default is None
        items_features_prefixes : list, optional
            Prefixes of the columns of the dataframe that are items_features_by_choice,
            default is None
        items_features_suffixes : list, optional
            Suffixes of the columns of the dataframe that are items_features_by_choice,
            default is None
        available_items_prefix: str, optional
            Prefix of the columns of the dataframe that precise available_items_by_choice,
            default is None
        available_items_suffix: str, optional
            Suffix of the columns of the dataframe that precise available_items_by_choice,
            default is None
        delimiter: str, optional
            Delimiter used to separate the given prefix or suffixes and the features names,
            default is "_"
        choice_column: str, optional
            Name of the column containing the choices, default is "choice"
        choice_format: str, optional
            How choice is indicated in df, either "items_id" or "items_index",
            default is "items_id"

        Returns
        -------
        ChoiceDataset
            corresponding ChoiceDataset
        """
        if items_features_prefixes is not None and items_features_suffixes is not None:
            raise ValueError(
                "You cannot give both items_features_prefixes and\
                    items_features_suffixes."
            )
        if available_items_prefix is not None and available_items_suffix is not None:
            raise ValueError(
                "You cannot give both available_items_prefix and\
                    available_items_suffix."
            )
        if choice_format not in ["items_index", "items_id"]:
            logging.warning("choice_format not undersood, defaulting to 'items_index'")

        if shared_features_columns is not None:
            shared_features_by_choice = df[shared_features_columns].to_numpy()
            shared_features_by_choice_names = shared_features_columns
        else:
            shared_features_by_choice = None
            shared_features_by_choice_names = None

        if items_features_suffixes is not None:
            items_features_names = items_features_suffixes
            items_features_by_choice = []
            for item in items_id:
                columns = [f"{item}{delimiter}{feature}" for feature in items_features_suffixes]
                for col in columns:
                    if col not in df.columns:
                        logging.warning(
                            f"Column {col} was not in DataFrame,\
                            dummy creation of the feature with zeros."
                        )
                        df[col] = 0
                items_features_by_choice.append(df[columns].to_numpy())
            items_features_by_choice = np.stack(items_features_by_choice, axis=1)
        elif items_features_prefixes is not None:
            items_features_names = items_features_prefixes
            items_features_by_choice = []
            for item in items_id:
                columns = [f"{feature}{delimiter}{item}" for feature in items_features_prefixes]
                for col in columns:
                    if col not in df.columns:
                        logging.warning(
                            f"Column {col} was not in DataFrame,\
                            dummy creation of the feature with zeros."
                        )
                        df[col] = 0
                items_features_by_choice.append(df[columns].to_numpy())
            items_features_by_choice = np.stack(items_features_by_choice, axis=1)
        else:
            items_features_by_choice = None
            items_features_names = None

        if available_items_suffix is not None:
            if isinstance(available_items_suffix, list):
                if not len(available_items_suffix) == len(items_id):
                    raise ValueError(
                        "You have given a list of columns for availabilities."
                        "We consider that it is one for each item however lenghts do not match"
                    )
                logging.info("You have given a list of columns for availabilities.")
                logging.infog("Each column will be matched to an item, given their order")
                available_items_by_choice = df[available_items_suffix].to_numpy()
            else:
                columns = [f"{item}{delimiter}{available_items_suffix}" for item in items_id]
                available_items_by_choice = df[columns].to_numpy()
        elif available_items_prefix is not None:
            if isinstance(available_items_prefix, list):
                if not len(available_items_prefix) == len(items_id):
                    raise ValueError(
                        "You have given a list of columns for availabilities."
                        "We consider that it is one for each item however lenghts do not match"
                    )
                logging.info("You have given a list of columns for availabilities.")
                logging.info("Each column will be matched to an item, given their order")
                available_items_by_choice = df[available_items_prefix].to_numpy()
            else:
                columns = [f"{available_items_prefix}{delimiter}{item}" for item in items_id]
                available_items_by_choice = df[columns].to_numpy()
        else:
            available_items_by_choice = None

        choices = df[choices_column].to_numpy()
        if choice_format == "items_id":
            if items_id is None:
                raise ValueError("items_id must be given to use choice_format='items_id'")
            items_id = np.array(items_id)
            choices = np.squeeze([np.where(items_id == c)[0] for c in choices])
            if choices.size == 0:
                raise ValueError("No choice found in the items_id list")

        return ChoiceDataset(
            shared_features_by_choice=shared_features_by_choice,
            shared_features_by_choice_names=shared_features_by_choice_names,
            items_features_by_choice=items_features_by_choice,
            items_features_by_choice_names=items_features_names,
            available_items_by_choice=available_items_by_choice,
            choices=choices,
        )

    @classmethod
    def from_single_long_df(
        cls,
        df,
        choices_column="choice",
        items_id_column="item_id",
        choices_id_column="choice_id",
        shared_features_columns=None,
        items_features_columns=None,
        choice_format="items_id",
    ):
        """Build numpy arrays for ChoiceDataset from a single dataframe in long format.

        Parameters
        ----------
        df : pandas.DataFrame
            dataframe in Long format
        choices_column: str, optional
            Name of the column containing the choices, default is "choice"
        items_id_column: str, optional
            Name of the column containing the item ids, default is "items_id"
        choices_id_column: str, optional
            Name of the column containing the choice ids. It is used to identify all rows
            about a single choice, default is "choice_id"
        shared_features_columns : list
            Columns of the dataframe that are shared_features_by_choice, default is None
        items_features_columns : list
            Columns of the dataframe that are items_features_by_choice, default is None
        choice_format: str, optional
            How choice is indicated in df, either "items_name" or "one_zero",
            default is "items_id"

        Returns
        -------
        ChoiceDataset
            corresponding ChoiceDataset
        """
        # Ordering items and choices by id
        items = np.sort(df[items_id_column].unique())
        choices_ids = np.sort(df[choices_id_column].unique())

        if shared_features_columns is not None:
            shared_features_by_choice = df[
                shared_features_columns + [choices_id_column]
            ].drop_duplicates()
            shared_features_by_choice = shared_features_by_choice.set_index(choices_id_column)
            shared_features_by_choice = shared_features_by_choice.loc[choices_ids].to_numpy()

            shared_features_by_choice_names = shared_features_columns
        else:
            shared_features_by_choice = None
            shared_features_by_choice_names = None

        (
            items_features_by_choice,
            avaialble_items_by_choice,
        ) = cls._long_df_to_items_features_array(
            df,
            features=items_features_columns,
            items_id_column=items_id_column,
            choices_id_column=choices_id_column,
            items_index=items,
            choices_index=choices_ids,
        )

        items_features_by_choice_names = items_features_columns

        if choice_format == "items_id":
            choices = df[[choices_column, choices_id_column]].drop_duplicates(choices_id_column)
            choices = choices.set_index(choices_id_column)
            choices = choices.loc[choices_ids].to_numpy()
            # items is the value (str) of the item
            choices = np.squeeze([np.where(items == c)[0] for c in choices])
        elif choice_format == "one_zero":
            choices = df[[items_id_column, choices_column, choices_id_column]]
            choices = choices.loc[choices[choices_column] == 1]
            choices = choices.set_index(choices_id_column)
            choices = (
                choices.loc[choices_ids][items_id_column]
                .map({k: v for v, k in enumerate(items)})
                .to_numpy()
            )
        else:
            raise ValueError(
                f"choice_format {choice_format} not recognized. Must be in ['items_id', 'one_zero']"
            )
        return ChoiceDataset(
            shared_features_by_choice=shared_features_by_choice,
            items_features_by_choice=items_features_by_choice,
            available_items_by_choice=avaialble_items_by_choice,
            choices=choices,
            shared_features_by_choice_names=shared_features_by_choice_names,
            items_features_by_choice_names=items_features_by_choice_names,
        )

    def save(self):
        """Save the dataset."""
        raise NotImplementedError

    def summary(self):
        """Display a summary of the dataset."""
        print("%=====================================================================%")
        print("%%% Summary of the dataset:")
        print("%=====================================================================%")
        print("Number of items:", self.get_n_items())
        print(
            "Number of choices:",
            len(self),
        )
        print("%=====================================================================%")

        if self.shared_features_by_choice is not None:
            print(" Shared Features by Choice:")
            print(f" {sum([f.shape[1] for f in self.shared_features_by_choice])} shared features")
            if self.shared_features_by_choice_names is not None:
                if self.shared_features_by_choice_names[0] is not None:
                    print(f" with names: {self.shared_features_by_choice_names}")
        else:
            print(" No Shared Features by Choice registered")
        print("\n")

        if self.items_features_by_choice is not None:
            if self.items_features_by_choice[0] is not None:
                print(" Items Features by Choice:")
                print(
                    f"""{sum([f.shape[2] if f.ndim == 3 else 1
                              for f in self.items_features_by_choice])
                        } items features """
                )
                if self.items_features_by_choice_names is not None:
                    if self.items_features_by_choice_names[0] is not None:
                        print(f" with names: {self.items_features_by_choice_names}")
        else:
            print(" No Items Features by Choice registered")
        print("%=====================================================================%")
        return ""

    def get_choices_batch(self, choices_indexes, features=None):
        """Access a chunk of data within the ChoiceDataset from choice indexes.

        Parameters
        ----------
        choices_indexes : int or list of int or slice
            indexes of the choices (that will be mapped to choice & session indexes) to return
        features : list of str, optional
            list of features to return. None returns all of them, default is None.

        Returns
        -------
        tuple of (array_like, )
            tuple of arrays containing a batch of shared_features_by_choice
        tuple of (array_like, )
            tuple of arrays containing a batch of items_features_by_choice
        array_like
            array containing a batch of availables_items_by_choice
        array_like
            array containing a batch of choices
        """
        _ = features
        if isinstance(choices_indexes, list):
            if np.array(choices_indexes).ndim > 1:
                raise ValueError(
                    """ChoiceDataset unidimensional can only be batched along choices
                                 dimension received a list with several axis of indexing."""
                )
            if self.shared_features_by_choice is None:
                shared_features_by_choice = None
            else:
                shared_features_by_choice = list(
                    shared_features_by_choice[choices_indexes]
                    # .astype(self._return_types[1][i])
                    for i, shared_features_by_choice in enumerate(self.shared_features_by_choice)
                )

            if self.items_features_by_choice is None:
                items_features_by_choice = None
            else:
                items_features_by_choice = list(
                    items_features_by_choice[choices_indexes]
                    # .astype(self._return_types[2][i])
                    for _, items_features_by_choice in enumerate(self.items_features_by_choice)
                )
            if self.available_items_by_choice is None:
                available_items_by_choice = np.ones(
                    (len(choices_indexes), self.base_num_items)
                ).astype("float32")
            else:
                if isinstance(self.available_items_by_choice, tuple):
                    available_items_by_choice = self.available_items_by_choice[0].batch[
                        self.available_items_by_choice[1][choices_indexes]
                    ]
                else:
                    available_items_by_choice = self.available_items_by_choice[choices_indexes]
                # .astype(self._return_types[3])

            choices = self.choices[choices_indexes].astype(self._return_types[3])

            if len(self.shared_features_by_choice_map) > 0:
                mapped_features = []
                for tuple_index in range(len(shared_features_by_choice)):
                    if tuple_index in self.shared_features_by_choice_map.keys():
                        feat_ind_min = 0
                        unstacked_feat = []
                        for feature_index in np.sort(
                            list(self.shared_features_by_choice_map[tuple_index].keys())
                        ):
                            if feat_ind_min != feature_index:
                                unstacked_feat.append(
                                    shared_features_by_choice[tuple_index][
                                        :, feat_ind_min:feature_index
                                    ]
                                )
                            unstacked_feat.append(
                                self.shared_features_by_choice_map[tuple_index][
                                    feature_index
                                ].batch[shared_features_by_choice[tuple_index][:, feature_index]]
                            )
                            feat_ind_min = feature_index + 1
                        if feat_ind_min != shared_features_by_choice[tuple_index].shape[1]:
                            unstacked_feat.append(
                                shared_features_by_choice[tuple_index][:, feat_ind_min:]
                            )
                        mapped_features.append(np.concatenate(unstacked_feat, axis=1))
                    else:
                        mapped_features.append(shared_features_by_choice[tuple_index])

                shared_features_by_choice = mapped_features

            if len(self.items_features_by_choice_map) > 0:
                mapped_features = []
                for tuple_index in range(len(items_features_by_choice)):
                    if tuple_index in self.items_features_by_choice_map.keys():
                        if items_features_by_choice[tuple_index].ndim == 1:
                            mapped_features.append(
                                self.items_features_by_choice_map[tuple_index][0].batch[
                                    items_features_by_choice[tuple_index]
                                ]
                            )
                        else:
                            feat_ind_min = 0
                            unstacked_feat = []
                            for feature_index in np.sort(
                                list(self.items_features_by_choice_map[tuple_index].keys())
                            ):
                                if feat_ind_min != feature_index:
                                    unstacked_feat.append(
                                        items_features_by_choice[tuple_index][
                                            :, :, feat_ind_min:feature_index
                                        ]
                                    )
                                unstacked_feat.append(
                                    self.items_features_by_choice_map[tuple_index][
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
                        self._return_types[0][i]
                    )
                if not self._return_shared_features_by_choice_tuple:
                    shared_features_by_choice = shared_features_by_choice[0]
                else:
                    shared_features_by_choice = tuple(shared_features_by_choice)

            if items_features_by_choice is not None:
                for i in range(len(items_features_by_choice)):
                    items_features_by_choice[i] = items_features_by_choice[i].astype(
                        self._return_types[1][i]
                    )
                # items_features_by_choice were not given as a tuple, so we return do not return
                # it as a tuple
                if not self._return_items_features_by_choice_tuple:
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
            return self.get_choices_batch(
                list(range(*choices_indexes.indices(self.choices.shape[0])))
            )

        choices_indexes = [choices_indexes]
        (
            shared_features_by_choices,
            items_features_by_choice,
            available_items_by_choice,
            choice,
        ) = self.get_choices_batch(choices_indexes)
        if shared_features_by_choices is not None:
            if isinstance(shared_features_by_choices, tuple):
                shared_features_by_choices = tuple(feat[0] for feat in shared_features_by_choices)
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

    def __getitem__(self, choices_indexes):
        """Create a sub-ChoiceDataset with only a subset of choices, from their indexes.

        Parameters
        ----------
        choices_indexes : np.ndarray
            indexes of the choices to keep, shape should be (num_choices,)

        Returns
        -------
        ChoiceDataset
            ChoiceDataset with only the sessions indexed by indexes
        """
        if isinstance(choices_indexes, int):
            choices_indexes = [choices_indexes]
        elif isinstance(choices_indexes, slice):
            return self.__getitem__(list(range(*choices_indexes.indices(len(self.choices)))))

        try:
            if self.shared_features_by_choice[0] is None:
                shared_features_by_choice = None
            else:
                shared_features_by_choice = tuple(
                    self.shared_features_by_choice[i][choices_indexes]
                    for i in range(len(self.shared_features_by_choice))
                )
                if not self._return_shared_features_by_choice_tuple:
                    shared_features_by_choice = shared_features_by_choice[0]
        except TypeError:
            shared_features_by_choice = None

        try:
            if self.items_features_by_choice[0] is None:
                items_features_by_choice = None
            else:
                items_features_by_choice = tuple(
                    self.items_features_by_choice[i][choices_indexes]
                    for i in range(len(self.items_features_by_choice))
                )
                if not self._return_items_features_by_choice_tuple:
                    items_features_by_choice = items_features_by_choice[0]
        except TypeError:
            items_features_by_choice = None

        try:
            if self.shared_features_by_choice_names[0] is None:
                shared_features_by_choice_names = None
            else:
                shared_features_by_choice_names = self.shared_features_by_choice_names
                if not self._return_shared_features_by_choice_tuple:
                    shared_features_by_choice_names = shared_features_by_choice_names[0]
        except TypeError:
            shared_features_by_choice_names = None
        try:
            if self.items_features_by_choice_names[0] is None:
                items_features_by_choice_names = None
            else:
                items_features_by_choice_names = self.items_features_by_choice_names
                if not self._return_items_features_by_choice_tuple:
                    items_features_by_choice_names = items_features_by_choice_names[0]
        except TypeError:
            items_features_by_choice_names = None

        try:
            if isinstance(self.available_items_by_choice, tuple):
                available_items_by_choice = self.available_items_by_choice[1][choices_indexes]
            else:
                available_items_by_choice = self.available_items_by_choice[choices_indexes]
        except TypeError:
            available_items_by_choice = None

        return ChoiceDataset(
            shared_features_by_choice=shared_features_by_choice,
            items_features_by_choice=items_features_by_choice,
            available_items_by_choice=available_items_by_choice,
            choices=[self.choices[i] for i in choices_indexes],
            shared_features_by_choice_names=shared_features_by_choice_names,
            items_features_by_choice_names=items_features_by_choice_names,
            features_by_ids=self.features_by_ids,
        )

    @property
    def batch(self):
        """Indexer. Corresponds to get_choice_batch, but with [] logic."""
        return self.indexer

    def iter_batch(self, batch_size, shuffle=False, sample_weight=None):
        """Iterate over dataset return batches of length batch_size.

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
        if batch_size == -1 or batch_size == len(self):
            yield self.indexer.get_full_dataset()
        else:
            # Get indexes for each choice
            num_choices = len(self)
            indexes = np.arange(num_choices)

            # Shuffle indexes
            if shuffle and not batch_size == -1:
                indexes = np.random.permutation(indexes)

            yielded_size = 0
            while yielded_size < num_choices:
                # Return sample_weight if not None, for index matching
                batch_indexes = indexes[yielded_size : yielded_size + batch_size].tolist()
                if sample_weight is not None:
                    yield (
                        self.batch[batch_indexes],
                        sample_weight[batch_indexes],
                    )
                else:
                    yield self.batch[batch_indexes]
                yielded_size += batch_size

    def filter(self, bool_list):
        """Filter over sessions indexes following bool.

        Parameters
        ----------
        bool_list : list of boolean
            list of booleans of length self.get_n_choices() to filter choices.
            True to keep, False to discard.
        """
        indexes = [i for i, keep in enumerate(bool_list) if keep]
        return self[indexes]

    def get_n_shared_features(self):
        """Access the number of shared features.

        Returns
        -------
        int
            number of shared items features
        """
        if self.shared_features_by_choice is not None:
            n_features = 0
            for shared_features in self.shared_features_by_choice:
                n_features += shared_features.shape[1]
            return n_features
        return 0

    def get_n_items_features(self):
        """Access the number of items features.

        Returns
        -------
        int
            number of items features
        """
        if self.items_features_by_choice is not None:
            n_features = 0
            for items_features in self.items_features_by_choice:
                n_features += items_features.shape[2]
            return n_features
        return 0
