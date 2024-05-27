"""Conditional MNL model."""

import logging

import numpy as np
import pandas as pd
import tensorflow as tf

from .base_model import ChoiceModel


class MNLCoefficients(object):
    """Base class to specify the structure of a cLogit."""

    def __init__(self):
        """Instantiate a MNLCoefficients object."""
        # User interface
        self.coefficients = {}
        # Handled by the model
        self.feature_to_weight = {}

    def add(self, coefficient_name, feature_name, items_indexes=None, items_names=None):
        """Add a coefficient to the model throught the specification of the utility.

        Parameters
        ----------
        coefficient_name : str
            Name given to the coefficient.
        feature_name : str
            features name to which the coefficient is associated. It should work with
            the names given in the ChoiceDataset that will be used for parameters estimation.
        items_indexes : list of int, optional
            list of items indexes (in the ChoiceDataset) for which we need to add a coefficient,
            by default None
        items_names : list of str, optional
            list of items names (in the ChoiceDataset) for which we need to add a coefficient,
            by default None

        Raises
        ------
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

    def add_shared(self, coefficient_name, feature_name, items_indexes=None, items_names=None):
        """Add a single, shared coefficient to the model throught the specification of the utility.

        Parameters
        ----------
        coefficient_name : str
            Name given to the coefficient.
        feature_name : str
            features name to which the coefficient is associated. It should work with
            the names given in the ChoiceDataset that will be used for parameters estimation.
        items_indexes : list of int, optional
            list of items indexes (in the ChoiceDataset) for which the coefficient will be used,
            by default None
        items_names : list of str, optional
            list of items names (in the ChoiceDataset) for which the coefficient will be used,
            by default None

        Raises
        ------
        ValueError
            When names or indexes are both not specified.
        """
        if items_indexes is None and items_names is None:
            raise ValueError("Either items_indexes or items_names must be specified")

        if not coefficient_name:
            coefficient_name = f"beta_{feature_name}"

        if isinstance(items_indexes, int):
            logging.warning(
                "You have added a single index to a shared coefficient. This is not recommended.",
                "Returning to standard add_coefficients method.",
            )
            self.add_coefficients(coefficient_name, feature_name, items_indexes, items_names)

        if isinstance(items_names, str):
            logging.warning(
                "You have added a single name to a shared coefficient. This is not recommended.",
                "Returning to standard add_coefficients method.",
            )
            self.add_coefficients(coefficient_name, feature_name, items_indexes, items_names)

        self.coefficients[coefficient_name] = {
            "feature_name": feature_name,
            "items_indexes": [items_indexes] if items_indexes is not None else None,
            "items_names": items_names if items_names is not None else None,
        }

    def get(self, coefficient_name):
        """Getter of a coefficient specification, from its name.

        Parameters
        ----------
        coefficient_name : str
            Name of the coefficient to get.

        Returns
        -------
        dict
            specification of the coefficient.
        """
        return self.coefficients[coefficient_name]

    def _add_tf_weight(self, weight_name, weight_index):
        """Create the Tensorflow weight corresponding for cLogit.

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

    @property
    def features_with_weights(self):
        """Get a list of the features that have a weight to be estimated.

        Returns
        -------
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

        Returns
        -------
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
    def names(self):
        """Returns the list of coefficients.

        Returns
        -------
        dict keys
            List of coefficients in the specification.
        """
        return list(self.coefficients.keys())


class ConditionalLogit(ChoiceModel):
    """Conditional MNL that has a generic structure. It can be parametrized with a dictionnary.

    Arguments:
    ----------
    coefficients: dict or MNLCoefficients
        Specfication of the model to be estimated.
    """

    def __init__(
        self,
        coefficients=None,
        add_exit_choice=False,
        optimizer="lbfgs",
        lr=0.001,
        **kwargs,
    ):
        """Initialize of Conditional-MNL.

        Parameters
        ----------
        coefficients : dict or MNLCoefficients
            Dictionnary containing the coefficients parametrization of the model.
            The dictionnary must have the following structure:
            {feature_name_1: mode_1, feature_name_2: mode_2, ...}
            mode must be among "constant", "item", "item-full" for now
            (same specifications as torch-choice).
        add_exit_choice : bool, optional
            Whether or not to normalize the probabilities computation with an exit choice
            whose utility would be 1, by default True
        """
        super().__init__(add_exit_choice=add_exit_choice, optimizer=optimizer, lr=lr, **kwargs)
        self.coefficients = coefficients
        self.instantiated = False

    def add_coefficients(
        self, feature_name, coefficient_name="", items_indexes=None, items_names=None
    ):
        """Add a coefficient to the model throught the specification of the utility.

        Parameters
        ----------
        feature_name : str
            features name to which the coefficient is associated. It should work with
            the names given in the ChoiceDataset that will be used for parameters estimation.
        coefficient_name : str, optional
            Name given to the coefficient. If not provided, name will be "beta_feature_name".
        items_indexes : list of int, optional
            list of items indexes (in the ChoiceDataset) for which we need to add a coefficient,
            by default None
        items_names : list of str, optional
            list of items names (in the ChoiceDataset) for which we need to add a coefficient,
            by default None

        Raises
        ------
        ValueError
            When names or indexes are both not specified.
        """
        self._add_coefficient(
            coefficient_name=coefficient_name,
            feature_name=feature_name,
            items_indexes=items_indexes,
            items_names=items_names,
            shared=False,
        )

    def add_shared_coefficient(
        self, feature_name, coefficient_name="", items_indexes=None, items_names=None
    ):
        """Add a single, shared coefficient to the model throught the specification of the utility.

        Parameters
        ----------
        feature_name : str
            features name to which the coefficient is associated. It should work with
            the names given in the ChoiceDataset that will be used for parameters estimation.
        coefficient_name : str, optional
            Name given to the coefficient. If not provided, name will be "beta_feature_name".
        items_indexes : list of int, optional
            list of items indexes (in the ChoiceDataset) for which the coefficient will be used,
            by default None
        items_names : list of str, optional
            list of items names (in the ChoiceDataset) for which the coefficient will be used,
            by default None

        Raises
        ------
        ValueError
            When names or indexes are both not specified.
        """
        self._add_coefficient(
            coefficient_name=coefficient_name,
            feature_name=feature_name,
            items_indexes=items_indexes,
            items_names=items_names,
            shared=True,
        )

    def _add_coefficient(self, feature_name, coefficient_name, items_indexes, items_names, shared):
        if self.coefficients is None:
            self.coefficients = MNLCoefficients()
        elif not isinstance(self.coefficients, MNLCoefficients):
            raise ValueError("Cannot add coefficient on top of a dict instantiation.")

        coefficient_name = coefficient_name if coefficient_name else "beta_%s" % feature_name
        add_method = self.coefficients.add_shared if shared else self.coefficients.add
        add_method(
            coefficient_name=coefficient_name,
            feature_name=feature_name,
            items_indexes=items_indexes,
            items_names=items_names,
        )

    def instantiate(self, choice_dataset):
        """Instantiate the model using the features in the choice_dataset.

        Parameters
        ----------
        choice_dataset: ChoiceDataset
            Used to match the features names with the model coefficients.
        """
        if not self.instantiated:
            if not isinstance(self.coefficients, MNLCoefficients):
                self._build_coefficients_from_dict(n_items=choice_dataset.get_n_items())
            self.trainable_weights = self._instantiate_tf_weights()

            # Checking that no weight has been attributed to non existing feature in dataset
            dataset_stacked_features_names = []
            if choice_dataset.shared_features_by_choice_names is not None:
                for i, feat_tuple in enumerate(choice_dataset.shared_features_by_choice_names):
                    dataset_stacked_features_names.append(feat_tuple)
            if choice_dataset.items_features_by_choice_names is not None:
                for i, feat_tuple in enumerate(choice_dataset.items_features_by_choice_names):
                    dataset_stacked_features_names.append(feat_tuple)
            dataset_stacked_features_names = np.concatenate(dataset_stacked_features_names).ravel()

            for feature_with_weight in self.coefficients.features_with_weights:
                if feature_with_weight != "intercept":
                    if feature_with_weight not in dataset_stacked_features_names:
                        raise ValueError(
                            f"""Feature {feature_with_weight} has an attributed coefficient
                            but is not in dataset"""
                        )
            self._store_dataset_features_names(choice_dataset)
            self.instantiated = True

    def _instantiate_tf_weights(self):
        """Instantiate the model from MNLCoefficients object.

        Returns
        -------
        list of tf.Tensor
            List of the weights created coresponding to the specification.
        """
        weights = []
        for weight_nb, weight_name in enumerate(self.coefficients.names):
            n_weights = (
                len(self.coefficients.get(weight_name)["items_indexes"])
                if self.coefficients.get(weight_name)["items_indexes"] is not None
                else len(self.coefficients.get(weight_name)["items_names"])
            )
            weight = tf.Variable(
                tf.random_normal_initializer(0.0, 0.02, seed=42)(shape=(1, n_weights)),
                name=weight_name,
            )
            weights.append(weight)
            self.coefficients._add_tf_weight(weight_name, weight_nb)

        self.trainable_weights = weights

        return weights

    def _build_coefficients_from_dict(self, n_items):
        """Build coefficients when they are given as a dictionnay.

        Parameters
        ----------
        n_items : int
            Number of different items in the assortment. Used to create the right number of weights.
        """
        coefficients = MNLCoefficients()
        for weight_counter, (feature, mode) in enumerate(self.coefficients.items()):
            if mode == "constant":
                coefficients.add_shared(
                    feature + f"_w_{weight_counter}", feature, list(range(n_items))
                )
            elif mode == "item":
                coefficients.add(feature + f"_w_{weight_counter}", feature, list(range(1, n_items)))
            elif mode == "item-full":
                coefficients.add(feature + f"_w_{weight_counter}", feature, list(range(n_items)))
            else:
                raise ValueError(f"Mode {mode} not recognized.")

        self.coefficients = coefficients

    def _store_dataset_features_names(self, dataset):
        """Register the name of the features in the dataset. For later use in utility computation.

        Parameters
        ----------
        dataset : ChoiceDataset
            ChoiceDataset used to fit the model.
        """
        self._shared_features_by_choice_names = dataset.shared_features_by_choice_names
        self._items_features_by_choice_names = dataset.items_features_by_choice_names

    def compute_batch_utility(
        self,
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices,
        verbose=1,
    ):
        """Compute the utility when the model is constructed from a MNLCoefficients object.

        Parameters
        ----------
        shared_features_by_choice : tuple of np.ndarray (choices_features)
            a batch of shared features
            Shape must be (n_choices, n_shared_features)
        items_features_by_choice : tuple of np.ndarray (choices_items_features)
            a batch of items features
            Shape must be (n_choices, n_items_features)
        available_items_by_choice : np.ndarray
            A batch of items availabilities
            Shape must be (n_choices, n_items)
        choices: np.ndarray
            Choices
            Shape must be (n_choices, )
        verbose : int, optional
            Parametrization of the logging outputs, by default 1

        Returns
        -------
        tf.Tensor
            Utilities corresponding of shape (n_choices, n_items)
        """
        _ = choices

        n_items = available_items_by_choice.shape[1]
        n_choices = available_items_by_choice.shape[0]
        items_utilities_by_choice = []

        if not isinstance(shared_features_by_choice, tuple):
            shared_features_by_choice = (shared_features_by_choice,)
        if not isinstance(items_features_by_choice, tuple):
            items_features_by_choice = (items_features_by_choice,)

        # Shared features
        if self._shared_features_by_choice_names is not None:
            for i, feat_tuple in enumerate(self._shared_features_by_choice_names):
                for j, feat in enumerate(feat_tuple):
                    if feat in self.coefficients.features_with_weights:
                        (
                            item_index_list,
                            weight_index_list,
                        ) = self.coefficients.get_weight_item_indexes(feat)
                        for item_index, weight_index in zip(item_index_list, weight_index_list):
                            partial_items_utility_by_choice = tf.zeros((n_choices, n_items))
                            partial_items_utility_by_choice = [
                                tf.zeros(n_choices) for _ in range(n_items)
                            ]

                            for q, idx in enumerate(item_index):
                                if isinstance(idx, list):
                                    for k in idx:
                                        tf.cast(shared_features_by_choice[i][:, j], tf.float32)
                                        compute = tf.multiply(
                                            shared_features_by_choice[i][:, j],
                                            self.trainable_weights[weight_index][:, q],
                                        )
                                        partial_items_utility_by_choice[k] += compute
                                else:
                                    compute = tf.multiply(
                                        tf.cast(shared_features_by_choice[i][:, j], tf.float32),
                                        self.trainable_weights[weight_index][:, q],
                                    )
                                    partial_items_utility_by_choice[idx] += compute

                            items_utilities_by_choice.append(
                                tf.cast(
                                    tf.stack(partial_items_utility_by_choice, axis=1), tf.float32
                                )
                            )
                    elif verbose > 0:
                        logging.info(
                            f"Feature {feat} is in dataset but has no weight assigned\
                                in utility computations"
                        )

        # Items features
        if self._items_features_by_choice_names is not None:
            for i, feat_tuple in enumerate(self._items_features_by_choice_names):
                for j, feat in enumerate(feat_tuple):
                    if feat in self.coefficients.features_with_weights:
                        (
                            item_index_list,
                            weight_index_list,
                        ) = self.coefficients.get_weight_item_indexes(feat)
                        for item_index, weight_index in zip(item_index_list, weight_index_list):
                            partial_items_utility_by_choice = tf.zeros((n_choices, n_items))

                            for q, idx in enumerate(item_index):
                                if isinstance(idx, list):
                                    for k in idx:
                                        partial_items_utility_by_choice = tf.concat(
                                            [
                                                partial_items_utility_by_choice[:, :k],
                                                tf.expand_dims(
                                                    tf.multiply(
                                                        tf.cast(
                                                            items_features_by_choice[i][:, k, j],
                                                            tf.float32,
                                                        ),
                                                        self.trainable_weights[weight_index][:, q],
                                                    ),
                                                    axis=-1,
                                                ),
                                                partial_items_utility_by_choice[:, k + 1 :],
                                            ],
                                            axis=1,
                                        )
                                else:
                                    partial_items_utility_by_choice = tf.concat(
                                        [
                                            partial_items_utility_by_choice[:, :idx],
                                            tf.expand_dims(
                                                tf.multiply(
                                                    tf.cast(
                                                        items_features_by_choice[i][:, idx, j],
                                                        tf.float32,
                                                    ),
                                                    self.trainable_weights[weight_index][:, q],
                                                ),
                                                axis=-1,
                                            ),
                                            partial_items_utility_by_choice[:, idx + 1 :],
                                        ],
                                        axis=1,
                                    )

                            items_utilities_by_choice.append(
                                tf.cast(partial_items_utility_by_choice, tf.float32)
                            )
                    elif verbose > 0:
                        logging.info(
                            f"Feature {feat} is in dataset but has no weight assigned\
                                in utility computations"
                        )

        if "intercept" in self.coefficients.features_with_weights:
            item_index_list, weight_index_list = self.coefficients.get_weight_item_indexes(
                "intercept"
            )

            for item_index, weight_index in zip(item_index_list, weight_index_list):
                partial_items_utility_by_choice = tf.zeros((n_items,))
                for q, idx in enumerate(item_index):
                    partial_items_utility_by_choice = tf.concat(
                        [
                            partial_items_utility_by_choice[:idx],
                            self.trainable_weights[weight_index][:, q],
                            partial_items_utility_by_choice[idx + 1 :],
                        ],
                        axis=0,
                    )

                partial_items_utility_by_choice = tf.stack(
                    [partial_items_utility_by_choice] * n_choices, axis=0
                )

                items_utilities_by_choice.append(
                    tf.cast(partial_items_utility_by_choice, tf.float32)
                )

        return tf.reduce_sum(items_utilities_by_choice, axis=0)

    def fit(self, choice_dataset, get_report=False, **kwargs):
        """Fit function to estimate the paramters.

        Parameters
        ----------
        choice_dataset : ChoiceDataset
            Choice dataset to use for the estimation.
        get_report: bool, optional
            Whether or not to compute a report of the estimation, by default False

        Returns
        -------
        dict
            dict with fit history.
        """
        self.instantiate(choice_dataset)

        fit = super().fit(choice_dataset=choice_dataset, **kwargs)
        if get_report:
            self.report = self.compute_report(choice_dataset)
        return fit

    def _fit_with_lbfgs(
        self,
        choice_dataset,
        sample_weight=None,
        get_report=False,
        **kwargs,
    ):
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

        Returns
        -------
        dict
            dict with fit history.
        """
        self.instantiate(choice_dataset)

        fit = super()._fit_with_lbfgs(
            dataset=choice_dataset,
            sample_weight=sample_weight,
            **kwargs,
        )
        if get_report:
            self.report = self.compute_report(choice_dataset)
        return fit

    def compute_report(self, dataset):
        """Compute a report of the estimated weights.

        Parameters
        ----------
        dataset : ChoiceDataset
            ChoiceDataset used for the estimation of the weights that will be
            used to compute the Std Err of this estimation.

        Returns
        -------
        pandas.DataFrame
            A DF with estimation, Std Err, z_value and p_value for each coefficient.
        """
        import tensorflow_probability as tfp

        weights_std = self.get_weights_std(dataset)
        dist = tfp.distributions.Normal(loc=0.0, scale=1.0)

        names = []
        z_values = []
        estimations = []
        p_z = []
        i = 0
        for weight in self.trainable_weights:
            for j in range(weight.shape[1]):
                if weight.shape[1] > 1:
                    names.append(f"{weight.name[:-2]}_{j}")
                else:
                    names.append(f"{weight.name[:-2]}")
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

        Returns
        -------
        tf.Tensor
            Estimation of the Std Err for the weights.
        """
        # Loops of differentiation
        with tf.GradientTape() as tape_1:
            with tf.GradientTape(persistent=True) as tape_2:
                model = self.clone()
                w = tf.concat(self.trainable_weights, axis=1)
                tape_2.watch(w)
                tape_1.watch(w)
                mw = []
                index = 0
                for _w in self.trainable_weights:
                    mw.append(w[:, index : index + _w.shape[1]])
                    index += _w.shape[1]
                model.trainable_weights = mw
                batch = next(dataset.iter_batch(batch_size=-1))
                utilities = model.compute_batch_utility(*batch)
                probabilities = tf.nn.softmax(utilities, axis=-1)
                loss = tf.keras.losses.CategoricalCrossentropy(reduction="sum")(
                    y_pred=probabilities,
                    y_true=tf.one_hot(dataset.choices, depth=probabilities.shape[1]),
                )
            # Compute the Jacobian
            jacobian = tape_2.jacobian(loss, w)
        # Compute the Hessian from the Jacobian
        hessian = tape_1.batch_jacobian(jacobian, w)
        inv_hessian = tf.linalg.inv(tf.squeeze(hessian))
        return tf.sqrt([inv_hessian[i][i] for i in range(len(tf.squeeze(hessian)))])

    def clone(self):
        """Return a clone of the model."""
        clone = ConditionalLogit(
            coefficients=self.coefficients,
            add_exit_choice=self.add_exit_choice,
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
        if hasattr(self, "trainable_weights"):
            clone.trainable_weights = self.trainable_weights
        if hasattr(self, "lr"):
            clone.lr = self.lr
        if hasattr(self, "_shared_features_by_choice_names"):
            clone._shared_features_by_choice_names = self._shared_features_by_choice_names
        if hasattr(self, "_items_features_by_choice_names"):
            clone._items_features_by_choice_names = self._items_features_by_choice_names
        return clone
