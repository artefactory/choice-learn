"""Implementation of the Nested Logit model."""
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

from choice_learn.models.base_model import ChoiceModel
from choice_learn.models.conditional_logit import MNLCoefficients


def nested_softmax_with_availabilities(
    items_logit_by_choice,
    available_items_by_choice,
    items_nests,
    gammas,
    normalize_exit=False,
    eps=1e-17,
):
    """Compute softmax probabilities from utilities and items repartition within nests.

    Takes into account availabilties (1 if the product is available, 0 otherwise) to set
    probabilities to 0 for unavailable products and to renormalize the probabilities of
    available products.
    Takes also into account Items nest to compute a two step probability: first, probability
    to choose a given nest then probability to choose a product within this nest.
    See Nested Logit formulation for more details.

    Parameters
    ----------
    items_logit_by_choice : np.ndarray (n_choices, n_items)
        Utilities / Logits on which to compute the softmax
    available_items_by_choice : np.ndarray (n_choices, n_items)
        Matrix indicating the availabitily (1) or not (0) of the products
    items_nests : np.ndarray (n_items)
        Nest index for each item  # Beware that nest index matches well gammas,
        it is not verified.
    gammas : np.ndarray of shape (n_choices, n_items)
        Nest gammas value that must be reshaped so that it matches items_logit_by_choice
        items_gammas_by_choice ?
    normalize_exit : bool, optional
        Whether to normalize the probabilities of available products with an exit choice of
        utility 1, by default False
    eps : float, optional
        Value to avoid division by 0 when a product with probability almost 1 is unavailable,
        by default 1e-5

    Returns
    -------
    tf.Tensor (n_choices, n_items)
        Probabilities of each product for each choice computed from Logits
    """
    if tf.reduce_any(gammas < 0.05):
        logging.warning(
            """At least one gamma value for nests is below 0.05 and is
        clipped to 0.05 for numeric optimization purposes."""
        )
    gammas = tf.clip_by_value(gammas, 0.05, tf.float32.max)
    numerator = tf.exp(tf.clip_by_value(items_logit_by_choice / gammas, tf.float32.min, 50))
    # Set unavailable products utility to 0
    numerator = tf.multiply(numerator, available_items_by_choice)
    items_nest_utility = tf.zeros_like(numerator) + eps
    for nest_index in tf.unique(items_nests)[0]:
        stack = tf.boolean_mask(numerator, items_nests == nest_index, axis=1)
        nest_utility = tf.reduce_sum(
            stack,
            axis=-1,
            keepdims=True,
        )
        items_nest_utility += nest_utility * tf.cast(items_nests == nest_index, tf.float32)
    numerator = numerator * (items_nest_utility ** (gammas - 1))
    # Sum of total available utilities
    denominator = tf.reduce_sum(numerator, axis=-1, keepdims=True)
    # Add 1 to the denominator to take into account the exit choice
    if normalize_exit:
        denominator += 1
    # Avoir division by 0 when only unavailable items have highest utilities
    elif eps:
        denominator += eps
    # Compute softmax
    return numerator / denominator


class NestedLogit(ChoiceModel):
    """Nested Logit Model class."""

    def __init__(
        self,
        items_nests,
        shared_gammas_over_nests=False,
        coefficients=None,
        add_exit_choice=False,
        optimizer="lbfgs",
        lr=0.001,
        **kwargs,
    ):
        """Initialize the Nested Logit model.

        Parameters
        ----------
        items_nest: list
            list of nests lists, each containing the items indexes in the nest.
        shared_gammas_over_nests : bool, optional
            Whether or not to share the gammas over the nests, by default False.
            If True it means that only one gamma value is estimated, and used for
            all the nests.
        coefficients : dict or MNLCoefficients
            Dictionnary containing the coefficients parametrization of the model.
            The dictionnary must have the following structure:
            {feature_name_1: mode_1, feature_name_2: mode_2, ...}
            mode must be among "constant", "item", "item-full" and "nest" for now
            (same specifications as torch-choice).
        add_exit_choice : bool, optional
            Whether or not to normalize the probabilities computation with an exit choice
            whose utility would be 1, by default True
        optimizer: str, optional
            Optimizer to use for the estimation, by default "lbfgs"
        lr: float, optional
            Learning rate for the optimizer, by default 0.001
        **kwargs
            Additional arguments to pass to the ChoiceModel base class.
        """
        super().__init__(add_exit_choice=add_exit_choice, optimizer=optimizer, lr=lr, **kwargs)
        self.coefficients = coefficients
        self.instantiated = False

        # Checking the items_nests format:
        if len(items_nests) < 2:
            raise ValueError(f"At least two nests should be given, got {len(items_nests)}")
        for i_nest, nest in enumerate(items_nests):
            if len(nest) < 1:
                raise ValueError(f"Nest {i_nest} is empty.")
            logging.info(
                f"""Checking nest specification,
                         got nest nb {i_nest+1} / {len(items_nests)}
                         with {len(nest)} items within."""
            )
        flat_items = np.concatenate(items_nests).flatten()
        if np.max(flat_items) >= len(flat_items):
            raise ValueError(
                f"""{len(flat_items)} have been given,\
                             cannot have an item index greater than this."""
            )
        if len(np.unique(flat_items)) != len(flat_items):
            raise ValueError("Got at least one items in several nests, which is not possible.")

        # create mapping items -> nests
        self.items_nests = items_nests
        items_to_nest = []
        for item_index in range(len(np.unique(flat_items))):
            for i_nest, nest in enumerate(items_nests):
                if item_index in nest:
                    if len(nest) > 1:
                        items_to_nest.append(i_nest)
                    else:
                        items_to_nest.append(-1)
        for i in range(np.max(items_to_nest)):
            if i not in items_to_nest:
                items_to_nest = [j - 1 if j > i else j for j in items_to_nest]
        self.items_to_nest = items_to_nest
        self.shared_gammas_over_nests = shared_gammas_over_nests

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
            raise ValueError("Cannot add shared coefficient on top of a dict instantiation.")

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

        # Initialization of gammas a bit different, it's a sensible variable
        # which should be in [eps, 1] -> initialized at 0.5
        if self.shared_gammas_over_nests:
            weights.append(
                tf.Variable(
                    [[0.5]],
                    name="gamma_nests",
                )
            )
        else:
            weights.append(
                tf.Variable(
                    [[0.5] * np.sum([1 if len(nest) > 1 else 0 for nest in self.items_nests])],
                    name="gammas_nests",
                )
            )

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

            # Additional mode compared to Conditional Logit
            elif mode == "nest":
                for nest in self.items_nests:
                    items_in_nest = [i for (i, j) in enumerate(nest) if j == nest]
                    coefficients.add_shared(
                        feature + f"_w_{weight_counter}", feature, items_in_nest
                    )
            else:
                raise ValueError(f"Mode {mode} for coefficients not recognized.")

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
                    if isinstance(idx, list):
                        for idx_idx in idx:
                            partial_items_utility_by_choice = tf.concat(
                                [
                                    partial_items_utility_by_choice[:idx_idx],
                                    self.trainable_weights[weight_index][:, q],
                                    partial_items_utility_by_choice[idx_idx + 1 :],
                                ],
                                axis=0,
                            )
                    else:
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

    @tf.function
    def batch_predict(
        self,
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices,
        sample_weight=None,
    ):
        """Represent one prediction (Probas + Loss) for one batch of a ChoiceDataset.

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
        choices_batch : np.ndarray
            Choices
            Shape must be (n_choices, )
        sample_weight : np.ndarray, optional
            List samples weights to apply during the gradient descent to the batch elements,
            by default None

        Returns
        -------
        tf.Tensor (1, )
            Value of NegativeLogLikelihood loss for the batch
        tf.Tensor (batch_size, n_items)
            Probabilities for each product to be chosen for each choice
        """
        # Compute utilities from features
        utilities = self.compute_batch_utility(
            shared_features_by_choice,
            items_features_by_choice,
            available_items_by_choice,
            choices,
        )

        batch_size = utilities.shape[0]
        batch_gammas = []
        if self.shared_gammas_over_nests:
            batch_gammas = self.trainable_weights[-1][0, 0] * tf.ones_like(utilities)
        else:
            for i in range(len(self.items_to_nest)):
                if self.items_to_nest[i] == -1:
                    batch_gammas.append([tf.constant(1.0)] * batch_size)
                else:
                    batch_gammas.append(
                        [self.trainable_weights[-1][0, self.items_to_nest[i]]] * batch_size
                    )
            batch_gammas = tf.stack(batch_gammas, axis=-1)

        probabilities = nested_softmax_with_availabilities(
            items_logit_by_choice=utilities,
            available_items_by_choice=available_items_by_choice,
            items_nests=tf.constant(self.items_to_nest),
            gammas=batch_gammas,
            normalize_exit=self.add_exit_choice,
        )

        # Compute loss from probabilities & actual choices
        batch_loss = {
            "optimized_loss": self.loss(
                y_pred=probabilities,
                y_true=tf.one_hot(choices, depth=probabilities.shape[1]),
                sample_weight=sample_weight,
            ),
            "Exact-NegativeLogLikelihood": self.exact_nll(
                y_pred=probabilities,
                y_true=tf.one_hot(choices, depth=probabilities.shape[1]),
                sample_weight=sample_weight,
            ),
        }
        return batch_loss, probabilities

    @tf.function
    def train_step(
        self,
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices,
        sample_weight=None,
    ):
        """Represent one training step (= one gradient descent step) of the model.

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
        choices_batch : np.ndarray
            Choices
            Shape must be (n_choices, )
        sample_weight : np.ndarray, optional
            List samples weights to apply during the gradient descent to the batch elements,
            by default None

        Returns
        -------
        tf.Tensor
            Value of NegativeLogLikelihood loss for the batch
        """
        with tf.GradientTape() as tape:
            utilities = self.compute_batch_utility(
                shared_features_by_choice=shared_features_by_choice,
                items_features_by_choice=items_features_by_choice,
                available_items_by_choice=available_items_by_choice,
                choices=choices,
            )

            batch_size = utilities.shape[0]
            batch_gammas = []
            if self.shared_gammas_over_nests:
                batch_gammas = self.trainable_weights[-1][0, 0] * tf.ones_like(utilities)
            else:
                for i in range(len(self.items_to_nest)):
                    if self.items_to_nest[i] == -1:
                        batch_gammas.append([tf.constant(1.0)] * batch_size)
                    else:
                        batch_gammas.append(
                            [self.trainable_weights[-1][0, self.items_to_nest[i]]] * batch_size
                        )
                batch_gammas = tf.stack(batch_gammas, axis=-1)

            probabilities = nested_softmax_with_availabilities(
                items_logit_by_choice=utilities,
                available_items_by_choice=available_items_by_choice,
                items_nests=tf.constant(self.items_to_nest),
                gammas=batch_gammas,
                normalize_exit=self.add_exit_choice,
            )
            # Negative Log-Likelihood
            neg_loglikelihood = self.loss(
                y_pred=probabilities,
                y_true=tf.one_hot(choices, depth=probabilities.shape[1]),
                sample_weight=sample_weight,
            )
            if self.regularization is not None:
                regularization = tf.reduce_sum(
                    [self.regularizer(w) for w in self.trainable_weights]
                )
                neg_loglikelihood += regularization

        grads = tape.gradient(neg_loglikelihood, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return neg_loglikelihood

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

                batch_gammas = []
                if model.shared_gammas_over_nests:
                    batch_gammas = model.trainable_weights[-1][0, 0] * tf.ones_like(utilities)
                else:
                    for i in range(len(self.items_to_nest)):
                        if model.items_to_nest[i] == -1:
                            batch_gammas.append([tf.constant(1.0)] * len(dataset))
                        else:
                            batch_gammas.append(
                                [model.trainable_weights[-1][0, model.items_to_nest[i]]]
                                * len(dataset)
                            )
                    batch_gammas = tf.stack(batch_gammas, axis=-1)

                probabilities = nested_softmax_with_availabilities(
                    items_logit_by_choice=utilities,
                    available_items_by_choice=batch[2],
                    items_nests=tf.constant(model.items_to_nest),
                    gammas=batch_gammas,
                    normalize_exit=self.add_exit_choice,
                    eps=1e-15,
                )
                loss = tf.keras.losses.CategoricalCrossentropy(reduction="sum")(
                    y_pred=probabilities,
                    y_true=tf.one_hot(dataset.choices, depth=probabilities.shape[1]),
                )

            # Compute the Jacobian
            jacobian = tape_2.jacobian(loss, w)
        # Compute the Hessian from the Jacobian
        hessian = tape_1.batch_jacobian(jacobian, w)
        inv_hessian = tf.linalg.inv(tf.squeeze(hessian))
        return tf.sqrt(
            [
                tf.clip_by_value(inv_hessian[i][i], 0.0, tf.float32.max)
                for i in range(len(tf.squeeze(hessian)))
            ]
        )

    def clone(self):
        """Return a clone/deepcopy of the model."""
        clone = NestedLogit(
            coefficients=self.coefficients,
            add_exit_choice=self.add_exit_choice,
            optimizer=self.optimizer_name,
            items_nests=self.items_nests,
        )
        if hasattr(self, "history"):
            clone.history = self.history
        if hasattr(self, "is_fitted"):
            clone.is_fitted = self.is_fitted
        if hasattr(self, "instantiated"):
            clone.instantiated = self.instantiated
        if hasattr(self, "shared_gammas_over_nests"):
            clone.shared_gammas_over_nests = self.shared_gammas_over_nests
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
