"""Implementation of the Nested Logit model."""
import numpy as np
import tensorflow as tf

from choice_learn.models.base_model import ChoiceModel
from choice_learn.models.conditional_logit import MNLCoefficients


def nested_softmax_with_availabilities(
    items_logit_by_choice,
    available_items_by_choice,
    items_nests,
    gammas,
    normalize_exit=False,
    eps=1e-5,
):
    """Compute softmax probabilities from utilities.

    Takes into account availabilties (1 if the product is available, 0 otherwise) to set
    probabilities to 0 for unavailable products and to renormalize the probabilities of
    available products.

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
    numerator = tf.exp(items_logit_by_choice / gammas)
    # Set unavailable products utility to 0
    numerator = tf.multiply(numerator, available_items_by_choice)

    items_nest_utility = tf.zeros_like(numerator)
    for nest_index in tf.unique(items_nests[0]):
        nest_utility = tf.reduce_sum(numerator[items_nests == nest_index], keepdims=True)
        items_nest_utility += nest_utility * tf.cast(items_nests == nest_index, tf.float32)

    numerator = numerator * (items_nest_utility ** (gammas - 1))
    # Sum of total available utilities
    denominator = tf.reduce_sum((items_nest_utility**gammas), axis=-1, keepdims=True)
    # Add 1 to the denominator to take into account the exit choice
    if normalize_exit:
        denominator += 1
    # Avoir division by 0 when only unavailable items have highest utilities
    elif eps:
        denominator += eps

    # Compute softmax
    return numerator / denominator


class NestedLogit(ChoiceModel):
    """Nested Logit Model."""

    def __init__(
        self,
        items_nests,
        coefficients=None,
        add_exit_choice=False,
        optimizer="lbfgs",
        lr=0.001,
        **kwargs,
    ):
        """Initialize of Conditional-MNL.

        Parameters
        ----------
        items_nest: list
            list containing nest index for each item
        coefficients : dict or MNLCoefficients
            Dictionnary containing the coefficients parametrization of the model.
            The dictionnary must have the following structure:
            {feature_name_1: mode_1, feature_name_2: mode_2, ...}
            mode must be among "constant", "item", "item-full" and "nest" for now
            (same specifications as torch-choice).
        add_exit_choice : bool, optional
            Whether or not to normalize the probabilities computation with an exit choice
            whose utility would be 1, by default True
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
            print(f"Got nest {i_nest} on {len(nest)} with {len(nest)} items.")
        flat_items = np.stack(items_nests).flatten()
        if np.max(flat_items) >= len(flat_items):
            raise ValueError(
                f"""{len(flat_items)} have been given,\
                             cannot have an item index greater than this."""
            )
        if len(np.unique(flat_items)) != len(flat_items):
            raise ValueError("Got at least one items in several nests, which is not possible.")
        self.items_nests = items_nests

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
            elif mode == "nest":
                for nest in self.items_nests:
                    items_in_nest = [i for (i, j) in enumerate(nest) if j == nest]
                    coefficients.add_shared(
                        feature + f"_w_{weight_counter}", feature, items_in_nest
                    )
            else:
                raise ValueError(f"Mode {mode} not recognized.")

        self.coefficients = coefficients
