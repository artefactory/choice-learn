"""Conditional MNL model."""

import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from .base_model import ChoiceModel


class ModelSpecification(object):
    """Base class to specify the structure of a cMNL."""

    def __init__(self):
        """Instantiate a ModelSpecification object."""
        # User interface
        self.coefficients = {}
        # Handled by the model
        self.feature_to_weight = {}

    def add_coefficients(
        self, coefficient_name, feature_name, items_indexes=None, items_names=None
    ):
        """Adds a coefficient to the model throught the specification of the utility.

        Parameters
        ----------
        coefficient_name : str
            Name given to the coefficient.
        feature_name : str
            features name to which the coefficient is associated. It should work with
            the names given.
            in the ChoiceDataset that will be used for parameters estimation.
        items_indexes : list of int, optional
            list of items indexes (in the ChoiceDataset) for which we need to add a coefficient,
            by default None
        items_names : list of str, optional
            list of items names (in the ChoiceDataset) for which we need to add a coefficient,
            by default None

        Raises:
        -------
        ValueError
            When names or indexes are both not specified.
        """
        if items_indexes is None and items_names is None:
            raise ValueError("Either items_indexes or items_names must be specified")

        if isinstance(items_indexes, int):
            items_indexes = [items_indexes]
        if isinstance(items_names, str):
            items_names = [items_names]
        self.coefficients[coefficient_name] = {
            "feature_name": feature_name,
            "items_indexes": items_indexes,
            "items_names": items_names,
        }

    def add_shared_coefficient(
        self, coefficient_name, feature_name, items_indexes=None, items_names=None
    ):
        """Adds a single, shared coefficient to the model throught the specification of the utility.

        Parameters
        ----------
        coefficient_name : str
            Name given to the coefficient.
        feature_name : str
            features name to which the coefficient is associated. It should work with
            the names given.
            in the ChoiceDataset that will be used for parameters estimation.
        items_indexes : list of int, optional
            list of items indexes (in the ChoiceDataset) for which the coefficient will be used,
            by default None
        items_names : list of str, optional
            list of items names (in the ChoiceDataset) for which the coefficient will be used,
            by default None

        Raises:
        -------
        ValueError
            When names or indexes are both not specified.
        """
        if items_indexes is None and items_names is None:
            raise ValueError("Either items_indexes or items_names must be specified")

        if isinstance(items_indexes, int):
            print(
                "You have added a single index to a shared coefficient. This is not recommended.",
                "Returning to standard add_coefficients method.",
            )
            self.add_coefficients(coefficient_name, feature_name, items_indexes, items_names)
        if isinstance(items_names, str):
            print(
                "You have added a single name to a shared coefficient. This is not recommended.",
                "Returning to standard add_coefficients method.",
            )
            self.add_coefficients(coefficient_name, feature_name, items_indexes, items_names)
        self.coefficients[coefficient_name] = {
            "feature_name": feature_name,
            "items_indexes": [items_indexes] if items_indexes is not None else None,
            "items_names": items_names if items_names is not None else None,
        }

    def get_coefficient(self, coefficient_name):
        """Getter of a coefficient specification, from its name.

        Parameters
        ----------
        coefficient_name : str
            Name of the coefficient to get.

        Returns:
        --------
        dict
            specification of the coefficient.
        """
        return self.coefficients[coefficient_name]

    def add_weight(self, weight_name, weight_index):
        """Method used by cMNL class to create the Tensorflow weight corresponding.

        Parameters
        ----------
        weight_name : str
            Name of the weight to add.
        weight_index : int
            Index of the weight (in the conditionalMNL) to add.
        """
        if weight_name not in self.coefficients.keys():
            raise ValueError(f"Weight {weight_name} not in coefficients")

        if self.coefficients[weight_name]["feature_name"] in self.feature_to_weight.keys():
            self.feature_to_weight[self.coefficients[weight_name]["feature_name"]].append(
                (
                    weight_name,
                    weight_index,
                )
            )
        else:
            self.feature_to_weight[self.coefficients[weight_name]["feature_name"]] = [
                (
                    weight_name,
                    weight_index,
                ),
            ]

    def list_features_with_weights(self):
        """Get a list of the features that have a weight to be estimated.

        Returns:
        --------
        dict.keys
            List of the features that have a weight to be estimated.
        """
        return self.feature_to_weight.keys()

    def get_weight_item_indexes(self, feature_name):
        """Get the indexes of the concerned items for a given weight.

        Parameters
        ----------
        feature_name : str
            Features that is concerned by the weight.

        Returns:
        --------
        list
            List of indexes of the items concerned by the weight.
        int
            The index of the weight in the conditionalMNL weights.
        """
        weights_info = self.feature_to_weight[feature_name]
        weight_names = [weight_info[0] for weight_info in weights_info]
        weight_indexs = [weight_info[1] for weight_info in weights_info]
        return [
            self.coefficients[weight_name]["items_indexes"] for weight_name in weight_names
        ], weight_indexs

    @property
    def coefficients_list(self):
        """Returns the list of coefficients.

        Returns:
        --------
        dict keys
            List of coefficients in the specification.
        """
        return list(self.coefficients.keys())


class ConditionalMNL(ChoiceModel):
    """Conditional MNL that has a generic structure. It can be parametrized with a dictionnary.

    Arguments:
    ----------
    params: dict or ModelSpecification
        Specfication of the model to be estimated.
    """

    def __init__(
        self,
        parameters=None,
        add_exit_choice=False,
        optimizer="Adam",
        lr=0.001,
        **kwargs,
    ):
        """Initialization of Conditional-MNL.

        Parameters:
        -----------
        parameters : dict or ModelSpecification
            Dictionnary containing the parametrization of the model.
            The dictionnary must have the following structure:
            {feature_name_1: mode_1, feature_name_2: mode_2, ...}
            mode must be among "constant", "item", "item-full" for now
            (same specifications as torch-choice).
        add_exit_choice : bool, optional
            Whether or not to normalize the probabilities computation with an exit choice
            whose utility would be 1, by default True
        """
        super().__init__(normalize_non_buy=add_exit_choice, optimizer=optimizer, lr=lr, **kwargs)
        self.params = parameters
        self.instantiated = False

    def add_coefficients(
        self, coefficient_name, feature_name, items_indexes=None, items_names=None
    ):
        """Adds a coefficient to the model throught the specification of the utility.

        Parameters
        ----------
        coefficient_name : str
            Name given to the coefficient.
        feature_name : str
            features name to which the coefficient is associated. It should work with
            the names given.
            in the ChoiceDataset that will be used for parameters estimation.
        items_indexes : list of int, optional
            list of items indexes (in the ChoiceDataset) for which we need to add a coefficient,
            by default None
        items_names : list of str, optional
            list of items names (in the ChoiceDataset) for which we need to add a coefficient,
            by default None

        Raises:
        -------
        ValueError
            When names or indexes are both not specified.
        """
        if self.params is None:
            self.params = ModelSpecification()
        elif not isinstance(self.params, ModelSpecification):
            raise ValueError("Cannot add coefficient on top of a dict instantiation.")
        self.params.add_coefficients(
            coefficient_name=coefficient_name,
            feature_name=feature_name,
            items_indexes=items_indexes,
            items_names=items_names,
        )

    def add_shared_coefficient(
        self, coefficient_name, feature_name, items_indexes=None, items_names=None
    ):
        """Adds a single, shared coefficient to the model throught the specification of the utility.

        Parameters
        ----------
        coefficient_name : str
            Name given to the coefficient.
        feature_name : str
            features name to which the coefficient is associated. It should work with
            the names given.
            in the ChoiceDataset that will be used for parameters estimation.
        items_indexes : list of int, optional
            list of items indexes (in the ChoiceDataset) for which the coefficient will be used,
            by default None
        items_names : list of str, optional
            list of items names (in the ChoiceDataset) for which the coefficient will be used,
            by default None

        Raises:
        -------
        ValueError
            When names or indexes are both not specified.
        """
        if self.params is None:
            self.params = ModelSpecification()
        elif not isinstance(self.params, ModelSpecification):
            raise ValueError("Cannot add shared coefficient on top of a dict instantiation.")
        self.params.add_shared_coefficient(
            coefficient_name=coefficient_name,
            feature_name=feature_name,
            items_indexes=items_indexes,
            items_names=items_names,
        )

    def instantiate_from_specifications(self):
        """Instantiate the model from ModelSpecification object.

        Returns:
        --------
        list of tf.Tensor
            List of the weights created coresponding to the specification.
        """
        weights = []
        for weight_nb, weight_name in enumerate(self.params.coefficients_list):
            num_weights = (
                len(self.params.get_coefficient(weight_name)["items_indexes"])
                if self.params.get_coefficient(weight_name)["items_indexes"] is not None
                else len(self.params.get_coefficient(weight_name)["items_names"])
            )
            weight = tf.Variable(
                tf.random_normal_initializer(0.0, 0.02, seed=42)(shape=(1, num_weights)),
                name=weight_name,
            )
            weights.append(weight)
            """
            feat_to_weight[self.params[weight_name]["feature_name"]] = (
                weight,
                self.params[weight_name],
            )
            """
            self.params.add_weight(weight_name, weight_nb)

            ## Fill items_indexes here
            # Better organize feat_to_weight and specifications
        return weights

    def _store_dataset_features_names(self, dataset):
        """Registers the name of the features in the dataset. For later use in utility computation.

        Parameters
        ----------
        dataset : ChoiceDataset
            ChoiceDataset used to fit the model.
        """
        self._items_features_names = dataset.fixed_items_features_names
        self._contexts_features_names = dataset.contexts_features_names
        self._contexts_items_features_names = dataset.contexts_items_features_names

    def compute_utility_from_specification(
        self,
        items_batch,
        contexts_batch,
        contexts_items_batch,
        availabilities_batch,
        choices_batch,
        verbose=0,
    ):
        """Computes the utility when the model is constructed from a ModelSpecification object.

        Parameters
        ----------
        tems_batch : tuple of np.ndarray (items_features)
            Fixed-Item-Features: formatting from ChoiceDataset: a matrix representing the products
            constant/fixed features.
            Shape must be (n_items, n_items_features)
        contexts_batch : tuple of np.ndarray (contexts_features)
            Time-Features
            Shape must be (n_choices, n_contexts_features)
        contexts_items_batch : tuple of np.ndarray (contexts_items_features)
            Time-Item-Features
            Shape must be (n_choices, n_contexts_items_features)
        availabilities_batch : np.ndarray
            Availabilities (contexts_items_availabilities)
            Shape must be (n_choices, n_items)
        choices_batch : np.ndarray
            Choices
            Shape must be (n_choices, )
        verbose : int, optional
            Parametrization of the logging outputs, by default 0

        Returns:
        --------
        tf.Tensor
            Utilities corresponding of shape (n_choices, n_items)
        """
        _ = choices_batch

        num_items = availabilities_batch.shape[1]
        num_choices = availabilities_batch.shape[0]
        contexts_items_utilities = []
        # Items features
        if self._items_features_names is not None:
            for i, feat_tuple in enumerate(self._items_features_names):
                for j, feat in enumerate(feat_tuple):
                    if feat in self.params.list_features_with_weights():
                        item_index_list, weight_index_list = self.params.get_weight_item_indexes(
                            feat
                        )
                        for item_index, weight_index in zip(item_index_list, weight_index_list):
                            s_i_u = tf.zeros((num_items,))
                            for q, idx in enumerate(item_index):
                                if isinstance(idx, list):
                                    for k in idx:
                                        s_i_u = tf.concat(
                                            [
                                                s_i_u[:k],
                                                tf.multiply(
                                                    items_batch[i][k, j],
                                                    self.weights[weight_index][:, q],
                                                ),
                                                s_i_u[k + 1 :],
                                            ],
                                            axis=0,
                                        )
                                else:
                                    s_i_u = tf.concat(
                                        [
                                            s_i_u[:idx],
                                            tf.multiply(
                                                items_batch[i][idx, j],
                                                self.weights[weight_index][:, q],
                                            ),
                                            s_i_u[idx + 1 :],
                                        ],
                                        axis=0,
                                    )
                            s_i_u = tf.stack([s_i_u] * num_choices, axis=0)

                            ### Need reshaping here
                            contexts_items_utilities.append(tf.cast(s_i_u, tf.float32))
                    else:
                        if verbose > 1:
                            print(
                                f"Feature {feat} is in dataset but has no weight assigned\
                                        in utility computations"
                            )
        # Context features
        if self._contexts_features_names is not None:
            for i, feat_tuple in enumerate(self._contexts_features_names):
                for j, feat in enumerate(feat_tuple):
                    if feat in self.params.list_features_with_weights():
                        print("found feat", feat)
                        item_index_list, weight_index_list = self.params.get_weight_item_indexes(
                            feat
                        )
                        for item_index, weight_index in zip(item_index_list, weight_index_list):
                            s_i_u = tf.zeros((num_choices, num_items))
                            s_i_u = [tf.zeros(num_choices) for _ in range(num_items)]

                            for q, idx in enumerate(item_index):
                                if isinstance(idx, list):
                                    for k in idx:
                                        """
                                        s_i_u = tf.concat(
                                            [
                                                s_i_u[:, :k],
                                                tf.expand_dims(
                                                    tf.multiply(
                                                        contexts_batch[i][:, j],
                                                        self.weights[weight_index][:, q],
                                                    ),
                                                    axis=-1,
                                                ),
                                                s_i_u[:, k + 1 :],
                                            ],
                                            axis=1,
                                        )
                                        """
                                        contexts_batch[i][:, j]
                                        compute = tf.multiply(
                                            contexts_batch[i][:, j],
                                            self.weights[weight_index][:, q],
                                        )
                                        s_i_u[k] += compute
                                else:
                                    """
                                    s_i_u = tf.concat(
                                        [
                                            s_i_u[:, :idx],
                                            tf.expand_dims(
                                                tf.multiply(
                                                    contexts_batch[i][:, j],
                                                    self.weights[weight_index][:, q],
                                                ),
                                                axis=-1,
                                            ),
                                            s_i_u[:, idx + 1 :],
                                        ],
                                        axis=1,
                                    )
                                    """
                                    compute = tf.multiply(
                                        contexts_batch[i][:, j], self.weights[weight_index][:, q]
                                    )
                                    s_i_u[idx] += compute

                            contexts_items_utilities.append(
                                tf.cast(tf.stack(s_i_u, axis=1), tf.float32)
                            )
                    else:
                        print(
                            f"Feature {feat} is in dataset but has no weight assigned\
                                in utility computations"
                        )

        # context Items features
        if self._contexts_items_features_names is not None:
            for i, feat_tuple in enumerate(self._contexts_items_features_names):
                for j, feat in enumerate(feat_tuple):
                    if feat in self.params.list_features_with_weights():
                        print("found feat", feat)

                        item_index_list, weight_index_list = self.params.get_weight_item_indexes(
                            feat
                        )
                        for item_index, weight_index in zip(item_index_list, weight_index_list):
                            s_i_u = tf.zeros((num_choices, num_items))

                            for q, idx in enumerate(item_index):
                                if isinstance(idx, list):
                                    for k in idx:
                                        s_i_u = tf.concat(
                                            [
                                                s_i_u[:, :k],
                                                tf.expand_dims(
                                                    tf.multiply(
                                                        contexts_items_batch[i][:, k, j],
                                                        self.weights[weight_index][:, q],
                                                    ),
                                                    axis=-1,
                                                ),
                                                s_i_u[:, k + 1 :],
                                            ],
                                            axis=1,
                                        )
                                else:
                                    s_i_u = tf.concat(
                                        [
                                            s_i_u[:, :idx],
                                            tf.expand_dims(
                                                tf.multiply(
                                                    contexts_items_batch[i][:, idx, j],
                                                    self.weights[weight_index][:, q],
                                                ),
                                                axis=-1,
                                            ),
                                            s_i_u[:, idx + 1 :],
                                        ],
                                        axis=1,
                                    )

                            contexts_items_utilities.append(tf.cast(s_i_u, tf.float32))
                    else:
                        print(
                            f"Feature {feat} is in dataset but has no weight assigned\
                                in utility computations"
                        )

        if "intercept" in self.params.list_features_with_weights():
            print("found feat", "intercept")
            item_index_list, weight_index_list = self.params.get_weight_item_indexes("intercept")

            for item_index, weight_index in zip(item_index_list, weight_index_list):
                s_i_u = tf.zeros((num_items,))
                for q, idx in enumerate(item_index):
                    s_i_u = tf.concat(
                        [
                            s_i_u[:idx],
                            self.weights[weight_index][:, q],
                            s_i_u[idx + 1 :],
                        ],
                        axis=0,
                    )

                s_i_u = tf.stack([s_i_u] * num_choices, axis=0)

                ### Need reshaping here

                contexts_items_utilities.append(tf.cast(s_i_u, tf.float32))

        return tf.reduce_sum(contexts_items_utilities, axis=0)

    def instantiate_from_dict(self, num_items):
        """Instantiation of the model from a dictionnary specification.

        Parameters
        ----------
        num_items : int
            Number of different items in the assortment. Used to create the right number of weights.
        """
        spec = ModelSpecification()
        weight_counter = 0
        for feature, mode in self.params.items():
            if mode == "constant":
                spec.add_shared_coefficient(
                    feature + f"_w_{weight_counter}", feature, list(range(num_items))
                )
            elif mode == "item":
                spec.add_coefficients(
                    feature + f"_w_{weight_counter}", feature, list(range(1, num_items))
                )
            elif mode == "item-full":
                spec.add_coefficients(
                    feature + f"_w_{weight_counter}", feature, list(range(num_items))
                )

            weight_counter += 1
        self.params = spec
        self.instantiate_from_specifications()

    def instantiate(
        self,
        num_items,
        items_features_names,
        contexts_features_names,
        contexts_items_features_names,
    ):
        """Instantiate the model from self.params and a dataset.

        Model is thus instantiated at .fit() time.

        Parameters
        ----------
        num_items : int
            Number of different items in the assortment. Used to create the right number of weights.
        items_features_names : list of str
            Names of the items features in the dataset.
        contexts_features_names : list of str
            Names of the contexts features in the dataset.
        contexts_items_features_names : list of str
            Names of the contexts items features in the dataset.

        Raises:
        -------
        NotImplementedError
            When a mode is wrongly precised.
        """
        # Possibility to stack weights to be faster ????
        if items_features_names is None:
            items_features_names = []
        if contexts_features_names is None:
            contexts_features_names = []
        if contexts_items_features_names is None:
            contexts_items_features_names = []
        weights = []
        weights_count = 0
        self._items_features_names = []
        for feat_tuple in items_features_names:
            tuple_names = []
            for feat in feat_tuple:
                if feat in self.params.keys():
                    if self.params[feat] == "constant":
                        weight = tf.Variable(
                            tf.random_normal_initializer(0.0, 0.02, seed=42)(shape=(1, 1))
                        )
                    elif self.params[feat] == "item":
                        weight = tf.Variable(
                            tf.random_normal_initializer(0.0, 0.02, seed=42)(
                                shape=(1, num_items - 1)
                            )
                        )
                    elif self.params[feat] == "item-full":
                        weight = tf.Variable(
                            tf.random_normal_initializer(0.0, 0.02, seed=42)(shape=(1, num_items))
                        )
                    else:
                        raise NotImplementedError(f"Param {self.params[feat]} not implemented")
                    weights.append(weight)
                    tuple_names.append((feat, weights_count))
                    weights_count += 1

                else:
                    print(
                        f"Feature {feat} is in dataset but has no weight assigned in utility\
                            computations"
                    )
            if len(tuple_names) > 0:
                self._items_features_names.append(tuple_names)

        self._contexts_features_names = []
        for feat_tuple in contexts_features_names:
            tuple_names = []
            for feat in feat_tuple:
                if feat in self.params.keys():
                    if self.params[feat] == "constant":
                        weight = tf.Variable(
                            tf.random_normal_initializer(0.0, 0.02, seed=42)(shape=(1, 1)),
                            name=feat,
                        )
                    elif self.params[feat] == "item":
                        weight = tf.Variable(
                            tf.random_normal_initializer(0.0, 0.02, seed=42)(
                                shape=(1, num_items - 1)
                            ),
                            name=feat,
                        )
                    elif self.params[feat] == "item-full":
                        weight = tf.Variable(
                            tf.random_normal_initializer(0.0, 0.02, seed=42)(shape=(1, num_items)),
                            name=feat,
                        )
                    else:
                        raise NotImplementedError(f"Param {self.params[feat]} not implemented")
                    weights.append(weight)
                    tuple_names.append((feat, weights_count))
                    weights_count += 1
                else:
                    print(
                        f"Feature {feat} is in dataset but has no weight assigned in utility\
                            computations"
                    )
            if len(tuple_names) > 0:
                self._contexts_features_names.append(tuple_names)

        self._contexts_items_features_names = []
        for feat_tuple in contexts_items_features_names:
            tuple_names = []
            for feat in feat_tuple:
                if feat in self.params.keys():
                    if self.params[feat] == "constant":
                        weight = tf.Variable(
                            tf.random_normal_initializer(0.0, 0.02, seed=42)(shape=(1, 1)),
                            name=feat,
                        )
                    elif self.params[feat] == "item":
                        weight = tf.Variable(
                            tf.random_normal_initializer(0.0, 0.02, seed=42)(
                                shape=(1, num_items - 1)
                            ),
                            name=feat,
                        )
                    elif self.params[feat] == "item-full":
                        weight = tf.Variable(
                            tf.random_normal_initializer(0.0, 0.02, seed=42)(shape=(1, num_items)),
                            name=feat,
                        )
                    else:
                        for i, s_tuple in enumerate(contexts_features_names):
                            for j, s_feat in enumerate(s_tuple):
                                if s_feat == self.params[feat]:
                                    # Get num weights with unique values of this feature
                                    # Create a dictionary {value: weight}
                                    # mydict = {}
                                    # for i, j in enumerate(
                                    #     np.unique(dataset.sessions_features[i][:, j])
                                    # ):
                                    #     mydict[i] = j
                                    # weight = tf.Variable(
                                    #     tf.random_normal_initializer(0.0, 0.02, seed=42)(
                                    #         shape=(1, j + 1)
                                    #     ),
                                    #     name=feat,
                                    # )
                                    pass
                        raise NotImplementedError(f"Param {self.params[feat]} not implemented")
                    weights.append(weight)
                    tuple_names.append((feat, weights_count))
                    weights_count += 1
                else:
                    print(
                        f"Feature {feat} is in dataset but has no weight assigned in utility\
                            computations"
                    )

            if len(tuple_names) > 0:
                self._contexts_items_features_names.append(tuple_names)

        if "intercept" in self.params.keys():
            if self.params["intercept"] == "constant":
                weight = tf.Variable(
                    tf.random_normal_initializer(0.0, 0.02, seed=42)(shape=(1, 1)), name="intercept"
                )
            elif self.params["intercept"] == "item":
                weight = tf.Variable(
                    tf.random_normal_initializer(0.0, 0.02, seed=42)(shape=(1, num_items - 1)),
                    name="intercept",
                )
            elif self.params["intercept"] == "item-full":
                weight = tf.Variable(
                    tf.random_normal_initializer(0.0, 0.02, seed=42)(shape=(1, num_items)),
                    name="intercept",
                )
            else:
                # Is supposed to be in contexts_features_names
                raise NotImplementedError(f"Param {self.params['intercept']} not implemented")
            weights.append(weight)
        else:
            print("No Intercept specified... was it forgotten ?")

        if len(weights) > 0:
            self.instantiated = True
        else:
            raise ValueError("No weights instantiated")
        return weights

    def compute_utility(
        self, items_batch, contexts_batch, contexts_items_batch, availabilities_batch, choices_batch
    ):
        """Main method to compute the utility of the model. Selects the right method to compute.

        Parameters
        ----------
        items_batch : tuple of np.ndarray (items_features)
            Fixed-Item-Features: formatting from ChoiceDataset: a matrix representing the products
            constant/fixed features.
            Shape must be (n_items, n_items_features)
        contexts_batch : tuple of np.ndarray (contexts_features)
            Time-Features
            Shape must be (n_choices, n_contexts_features)
        contexts_items_batch : tuple of np.ndarray (contexts_items_features)
            Time-Item-Features
            Shape must be (n_choices, n_contexts_items_features)
        availabilities_batch : np.ndarray
            Availabilities (contexts_items_availabilities)
            Shape must be (n_choices, n_items)
        choices_batch : np.ndarray
            Choices Shape must be (n_choices, )

        Returns:
        --------
        tf.Tensor
            Computed utilities of shape (n_choices, n_items).
        """
        if isinstance(self.params, ModelSpecification):
            return self.compute_utility_from_specification(
                items_batch,
                contexts_batch,
                contexts_items_batch,
                availabilities_batch,
                choices_batch,
            )
        return self.compute_utility_from_dict(
            items_batch,
            contexts_batch,
            contexts_items_batch,
            availabilities_batch,
            choices_batch,
        )

    def compute_utility_from_dict(
        self, items_batch, contexts_batch, contexts_items_batch, availabilities_batch, choices_batch
    ):
        """Computes the utility when the model is constructed from a dictionnary object.

        Parameters
        ----------
        items_batch : tuple of np.ndarray (items_features)
            Fixed-Item-Features: formatting from ChoiceDataset: a matrix representing the products
            constant/fixed features.
            Shape must be (n_items, n_items_features)
        contexts_batch : tuple of np.ndarray (contexts_features)
            Time-Features
            Shape must be (n_choices, n_contexts_features)
        contexts_items_batch : tuple of np.ndarray (contexts_items_features)
            Time-Item-Features
            Shape must be (n_choices, n_contexts_items_features)
        availabilities_batch : np.ndarray
            Availabilities (contexts_items_availabilities)
            Shape must be (n_choices, n_items)
        choices_batch : np.ndarray
            Choices
            Shape must be (n_choices, )
        verbose : int, optional
            Parametrization of the logging outputs, by default 0

        Returns:
        --------
        tf.Tensor
            Utilities corresponding of shape (n_choices, n_items)
        """
        _, _ = availabilities_batch, choices_batch

        contexts_items_utilities = []
        if items_batch is not None:
            num_items = items_batch[0].shape[0]
        else:
            num_items = contexts_items_batch[0].shape[1]
        num_choices = availabilities_batch.shape[0]

        # Items features
        for i, feat_tuple in enumerate(self._items_features_names):
            for j, (feat, k) in enumerate(feat_tuple):
                if feat in self.params.keys():
                    weight = self.weights[k]
                    if self.params[feat] == "constant":
                        s_i_u = tf.concat(
                            [tf.multiply(items_batch[i][:, j], weight)] * num_choices, axis=0
                        )
                    elif self.params[feat] == "item":
                        weight = tf.concat([tf.constant([[0.0]]), weight], axis=-1)
                        s_i_u = tf.concat(
                            [tf.multiply(items_batch[i][:, j], weight)] * num_choices, axis=0
                        )
                    elif self.params[feat] == "item-full":
                        s_i_u = tf.concat(
                            [tf.multiply(items_batch[i][:, j], weight)] * num_choices, axis=0
                        )
                    else:
                        raise NotImplementedError(f"Param {self.params[feat]} not implemented")
                    contexts_items_utilities.append(s_i_u)
                else:
                    print(
                        f"Feature {feat} is in dataset but has no weight assigned in utility \
                        computations"
                    )

        # context features
        for i, feat_tuple in enumerate(self._contexts_features_names):
            for j, (feat, k) in enumerate(feat_tuple):
                if feat in self.params.keys():
                    weight = self.weights[k]
                    if self.params[feat] == "constant":
                        s_i_u = tf.concat(
                            [tf.multiply(contexts_batch[i][j], weight)] * num_items, axis=-1
                        )
                    elif self.params[feat] == "item":
                        weight = tf.concat([tf.constant([[0.0]]), weight], axis=-1)
                        s_i_u = tf.tensordot(contexts_batch[i][:, j : j + 1], weight, axes=1)
                    elif self.params[feat] == "item-full":
                        s_i_u = tf.tensordot(contexts_batch[i][:, j : j + 1], weight, axes=1)
                    else:
                        raise NotImplementedError(f"Param {self.params[feat]} not implemented")
                    contexts_items_utilities.append(s_i_u)
                else:
                    print(
                        f"Feature {feat} is in dataset but has no weight assigned in utility \
                        computations"
                    )

        # context Items features
        for i, feat_tuple in enumerate(self._contexts_items_features_names):
            for j, (feat, k) in enumerate(feat_tuple):
                if feat in self.params.keys():
                    weight = self.weights[k]
                    if self.params[feat] == "constant":
                        s_i_u = tf.multiply(contexts_items_batch[i][:, :, j], weight)
                    elif self.params[feat] == "item":
                        weight = tf.concat([tf.constant([[0.0]]), weight], axis=-1)
                        s_i_u = tf.multiply(contexts_items_batch[i][:, :, j], weight)
                    elif self.params[feat] == "item-full":
                        s_i_u = tf.multiply(contexts_items_batch[i][:, :, j], weight)
                    else:
                        raise NotImplementedError(f"Param {self.params[feat]} not implemented")
                    contexts_items_utilities.append(s_i_u)
                else:
                    print(
                        f"Feature {feat} is in dataset but has no weight assigned in utility \
                        computations"
                    )

        if "intercept" in self.params.keys():
            weight = self.weights[-1]
            if self.params["intercept"] == "constant":
                s_i_u = tf.concat([tf.concat([weight] * num_items, axis=0)] * num_choices, axis=0)
            elif self.params["intercept"] == "item":
                weight = tf.concat([tf.constant([[0.0]]), weight], axis=-1)
                s_i_u = tf.concat([weight] * num_choices, axis=0)
            elif self.params["intercept"] == "item-full":
                s_i_u = tf.concat([weight] * num_choices, axis=0)
            else:
                raise NotImplementedError(f"Param {self.params[feat]} not implemented")
            contexts_items_utilities.append(s_i_u)

        return tf.reduce_sum(contexts_items_utilities, axis=0)

    def fit(self, choice_dataset, get_report=False, **kwargs):
        """Main fit function to estimate the paramters.

        Parameters
        ----------
        choice_dataset : ChoiceDataset
            Choice dataset to use for the estimation.
        get_report: bool, optional
            Whether or not to compute a report of the estimation, by default False

        Returns:
        --------
        ConditionalMNL
            With estimated weights.
        """
        if not self.instantiated:
            if isinstance(self.params, ModelSpecification):
                self.weights = self.instantiate_from_specifications()
                self._store_dataset_features_names(choice_dataset)
            else:
                self.weights = self.instantiate(
                    num_items=choice_dataset.get_num_items(),
                    items_features_names=choice_dataset.fixed_items_features_names,
                    contexts_features_names=choice_dataset.contexts_features_names,
                    contexts_items_features_names=choice_dataset.contexts_items_features_names,
                )
            self.instantiated = True
        fit = super().fit(choice_dataset=choice_dataset, **kwargs)
        if get_report:
            self.report = self.compute_report(choice_dataset)
        return fit

    def _fit_with_lbfgs(self, choice_dataset, n_epochs, tolerance=1e-8, get_report=False):
        """Specific fit function to estimate the paramters with LBFGS.

        Parameters
        ----------
        choice_dataset : ChoiceDataset
            Choice dataset to use for the estimation.
        n_epochs : int
            Number of epochs to run.
        tolerance : float, optional
            Tolerance in the research of minimum, by default 1e-8
        get_report: bool, optional
            Whether or not to compute a report of the estimation, by default False

        Returns:
        --------
        conditionalMNL
            self with estimated weights.
        """
        if not self.instantiated:
            if isinstance(self.params, ModelSpecification):
                self.weights = self.instantiate_from_specifications()
                self._store_dataset_features_names(choice_dataset)
            else:
                self.weights = self.instantiate(
                    num_items=choice_dataset.get_num_items(),
                    items_features_names=choice_dataset.fixed_items_features_names,
                    contexts_features_names=choice_dataset.contexts_features_names,
                    contexts_items_features_names=choice_dataset.contexts_items_features_names,
                )
            self.instantiated = True
        fit = super()._fit_with_lbfgs(choice_dataset, n_epochs, tolerance)
        if get_report:
            self.report = self.compute_report(choice_dataset)
        return fit

    def compute_report(self, dataset):
        """Computes a report of the estimated weights.

        Parameters
        ----------
        dataset : ChoiceDataset
            ChoiceDataset used for the estimation of the weights that will be
            used to compute the Std Err of this estimation.

        Returns:
        --------
        pandas.DataFrame
            A DF with estimation, Std Err, z_value and p_value for each coefficient.
        """
        weights_std = self.get_weights_std(dataset)
        dist = tfp.distributions.Normal(loc=0.0, scale=1.0)

        names = []
        z_values = []
        estimations = []
        p_z = []
        i = 0
        for weight in self.weights:
            for j in range(weight.shape[1]):
                names.append(f"{weight.name}_{j}")
                estimations.append(weight.numpy()[0][j])
                z_values.append(weight.numpy()[0][j] / weights_std[i].numpy())
                p_z.append(2 * (1 - dist.cdf(tf.math.abs(z_values[-1])).numpy()))
                i += 1

        return pd.DataFrame(
            {
                "Coefficient Name": names,
                "Coefficient Estimation": estimations,
                "Std. Err": weights_std.numpy(),
                "z_value": z_values,
                "P(.>z)": p_z,
            },
        )

    def get_weights_std(self, dataset):
        """Approximates Std Err with Hessian matrix.

        Parameters
        ----------
        dataset : ChoiceDataset
            ChoiceDataset used for the estimation of the weights that will be
            used to compute the Std Err of this estimation.

        Returns:
        --------
        tf.Tensor
            Estimation of the Std Err for the weights.
        """
        # Loops of differentiation
        with tf.GradientTape() as tape_1:
            with tf.GradientTape(persistent=True) as tape_2:
                model = self.clone()
                w = tf.concat(self.weights, axis=1)
                tape_2.watch(w)
                tape_1.watch(w)
                mw = []
                index = 0
                for _w in self.weights:
                    mw.append(w[:, index : index + _w.shape[1]])
                    index += _w.shape[1]
                model.weights = mw
                for batch in dataset.iter_batch(batch_size=-1):
                    utilities = model.compute_utility(*batch)
                    probabilities = tf.nn.softmax(utilities, axis=-1)
                    loss = tf.keras.losses.CategoricalCrossentropy(reduction="sum")(
                        y_pred=probabilities,
                        y_true=tf.one_hot(dataset.choices, depth=4),
                    )
            # Compute the Jacobian
            jacobian = tape_2.jacobian(loss, w)
        # Compute the Hessian from the Jacobian
        hessian = tape_1.batch_jacobian(jacobian, w)
        return tf.sqrt([tf.linalg.inv(tf.squeeze(hessian))[i][i] for i in range(13)])

    def clone(self):
        """Returns a clone of the model."""
        clone = ConditionalMNL(
            parameters=self.params,
            add_exit_choice=self.normalize_non_buy,
            optimizer=self.optimizer_name,
        )
        if hasattr(self, "history"):
            clone.history = self.history
        if hasattr(self, "is_fitted"):
            clone.is_fitted = self.is_fitted
        if hasattr(self, "instantiated"):
            clone.instantiated = self.instantiated
        clone.loss = self.loss
        clone.label_smoothing = self.label_smoothing
        if hasattr(self, "report"):
            clone.report = self.report
        if hasattr(self, "weights"):
            clone.weights = self.weights
        if hasattr(self, "lr"):
            clone.lr = self.lr
        if hasattr(self, "_items_features_names"):
            clone._items_features_names = self._items_features_names
        if hasattr(self, "_contexts_features_names"):
            clone._contexts_features_names = self._contexts_features_names
        if hasattr(self, "_contexts_items_features_names"):
            clone._contexts_items_features_names = self._contexts_items_features_names
        return clone
