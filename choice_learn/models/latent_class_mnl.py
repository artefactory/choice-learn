"""Latent Class MNL models."""

import tensorflow as tf

from .base_model import BaseLatentClassModel
from .conditional_mnl import ConditionalMNL, ModelSpecification
from .simple_mnl import SimpleMNL


class LatentClassSimpleMNL(BaseLatentClassModel):
    """Latent Class for SimpleMNL."""

    def __init__(
        self,
        n_latent_classes,
        fit_method,
        epochs,
        add_exit_choice=False,
        tolerance=1e-6,
        intercept=None,
        optimizer="Adam",
        lr=0.001,
        **kwargs,
    ):
        """Initialization.

        Parameters
        ----------
        n_latent_classes : int
            Number of latent classes.
        fit_method : str
            Method to be used to estimate the model.
        epochs : int
            Number of epochs
        add_exit_choice : bool, optional
            Whether to normalize probabilities with exit choice, by default False
        tolerance : float, optional
            LBFG-S tolerance, by default 1e-6
        intercept : str, optional
            Type of intercept to include in the SimpleMNL.
            Must be in (None, 'item', 'item-full', 'constant'), by default None
        optimizer : str, optional
            tf.keras.optimizers to be used, by default "Adam"
        lr : float, optional
            Learning rate to use for optimizer if relevant, by default 0.001
        """
        self.n_latent_classes = n_latent_classes
        self.intercept = intercept
        model_params = {
            "add_exit_choice": add_exit_choice,
            "intercept": intercept,
            "optimizer": optimizer,
            "tolerance": tolerance,
            "lr": lr,
            "epochs": epochs,
        }

        super().__init__(
            model_class=SimpleMNL,
            model_parameters=model_params,
            n_latent_classes=n_latent_classes,
            fit_method=fit_method,
            epochs=epochs,
            add_exit_choice=add_exit_choice,
            tolerance=tolerance,
            optimizer=optimizer,
            lr=lr,
            **kwargs,
        )

    def instantiate_latent_models(
        self, n_items, n_fixed_items_features, n_contexts_features, n_contexts_items_features
    ):
        """Instantiation of the Latent Models that are SimpleMNLs.

        Parameters
        ----------
        n_items : int
            Number of items/aternatives to consider.
        n_fixed_items_features : int
            Number of fixed items features.
        n_contexts_features : int
            Number of contexts features
        n_contexts_items_features : int
            Number of contexts items features
        """
        for model in self.models:
            model.indexes, model.weights = model.instantiate(
                n_items, n_fixed_items_features, n_contexts_features, n_contexts_items_features
            )
            model.instantiated = True

    def instantiate(
        self, n_items, n_fixed_items_features, n_contexts_features, n_contexts_items_features
    ):
        """Instantiation of the Latent Class MNL model."""
        self.latent_logits = tf.Variable(
            tf.random_normal_initializer(0.0, 0.02, seed=42)(shape=(self.n_latent_classes - 1,)),
            name="Latent-Logits",
        )

        self.models = [self.model_class(**mp) for mp in self.model_parameters]

        self.instantiate_latent_models(
            n_items=n_items,
            n_fixed_items_features=n_fixed_items_features,
            n_contexts_features=n_contexts_features,
            n_contexts_items_features=n_contexts_items_features,
        )

    def fit(self, dataset, **kwargs):
        """Fit the model to the dataset.

        Parameters
        ----------
        dataset : ChoiceDataset
            Dataset to fit the model to.
        """
        if not self.instantiated:
            self.instantiate(
                n_items=dataset.get_n_items(),
                n_fixed_items_features=dataset.get_n_fixed_items_features(),
                n_contexts_features=dataset.get_n_contexts_features(),
                n_contexts_items_features=dataset.get_n_contexts_items_features(),
            )
        return super().fit(dataset, **kwargs)


class LatentClassConditionalMNL(BaseLatentClassModel):
    """Latent Class for ConditionalMNL."""

    def __init__(
        self,
        n_latent_classes,
        fit_method,
        parameters=None,
        epochs=1,
        add_exit_choice=False,
        tolerance=1e-6,
        optimizer="Adam",
        lr=0.001,
        **kwargs,
    ):
        """Initialization.

        Parameters
        ----------
        n_latent_classes : int
            Number of latent classes.
        fit_method : str
            Method to be used to estimate the model.
        parameters : dict or ModelSpecification
            Dictionnary containing the parametrization of the model.
            The dictionnary must have the following structure:
            {feature_name_1: mode_1, feature_name_2: mode_2, ...}
            mode must be among "constant", "item", "item-full" for now
            (same specifications as torch-choice).
        epochs : int
            Number of epochs
        add_exit_choice : bool, optional
            Whether to normalize probabilities with exit choice, by default False
        tolerance : float, optional
            LBFG-S tolerance, by default 1e-6
        optimizer : str, optional
            tf.keras.optimizers to be used, by default "Adam"
        lr : float, optional
            Learning rate to use for optimizer if relevant, by default 0.001
        """
        self.n_latent_classes = n_latent_classes
        self.fit_method = fit_method
        self.params = parameters
        self.epochs = epochs
        self.add_exit_choice = add_exit_choice
        self.tolerance = tolerance
        self.optimizer = optimizer
        self.lr = lr

        model_params = {
            "params": self.params,
            "add_exit_choice": self.add_exit_choice,
            "optimizer": self.optimizer,
            "tolerance": self.tolerance,
            "lr": self.lr,
            "epochs": self.epochs,
        }

        super().__init__(
            model_class=ConditionalMNL,
            model_parameters=model_params,
            n_latent_classes=n_latent_classes,
            fit_method=fit_method,
            epochs=epochs,
            add_exit_choice=add_exit_choice,
            tolerance=tolerance,
            optimizer=optimizer,
            lr=lr,
            **kwargs,
        )

    def instantiate_latent_models(
        self, n_items, n_fixed_items_features, n_contexts_features, n_contexts_items_features
    ):
        """Instantiation of the Latent Models that are SimpleMNLs.

        Parameters
        ----------
        n_items : int
            Number of items/aternatives to consider.
        n_fixed_items_features : int
            Number of fixed items features.
        n_contexts_features : int
            Number of contexts features
        n_contexts_items_features : int
            Number of contexts items features
        """
        for model in self.models:
            model.indexes, model.weights = model.instantiate(
                n_items, n_fixed_items_features, n_contexts_features, n_contexts_items_features
            )
            model.instantiated = True

    def instantiate(
        self, n_items, n_fixed_items_features, n_contexts_features, n_contexts_items_features
    ):
        """Instantiation of the Latent Class MNL model."""
        self.latent_logits = tf.Variable(
            tf.random_normal_initializer(0.0, 0.02, seed=42)(shape=(self.n_latent_classes - 1,)),
            name="Latent-Logits",
        )

        self.models = [self.model_class(**mp) for mp in self.model_parameters]

        self.instantiate_latent_models(
            n_items=n_items,
            n_fixed_items_features=n_fixed_items_features,
            n_contexts_features=n_contexts_features,
            n_contexts_items_features=n_contexts_items_features,
        )

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

    def fit(self, dataset, **kwargs):
        """Fit the model to the dataset.

        Parameters
        ----------
        dataset : ChoiceDataset
            Dataset to fit the model to.
        """
        if not self.instantiated:
            self.instantiate(
                n_items=dataset.get_n_items(),
                n_fixed_items_features=dataset.get_n_fixed_items_features(),
                n_contexts_features=dataset.get_n_contexts_features(),
                n_contexts_items_features=dataset.get_n_contexts_items_features(),
            )
        return super().fit(dataset, **kwargs)
