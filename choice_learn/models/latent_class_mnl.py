"""Latent Class MNL models."""

import copy

import tensorflow as tf

import choice_learn.tf_ops as tf_ops

from .conditional_logit import ConditionalLogit, MNLCoefficients
from .latent_class_base_model import BaseLatentClassModel
from .simple_mnl import SimpleMNL


class LatentClassSimpleMNL(BaseLatentClassModel):
    """Latent Class for SimpleMNL."""

    def __init__(
        self,
        n_latent_classes,
        fit_method,
        epochs=100,
        batch_size=128,
        add_exit_choice=False,
        lbfgs_tolerance=1e-6,
        intercept=None,
        optimizer="Adam",
        lr=0.001,
        epochs_maximization=1000,
        **kwargs,
    ):
        """Initialize model.

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
        lbfgs_tolerance : float, optional
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
        model_parameters = {
            "add_exit_choice": add_exit_choice,
            "intercept": intercept,
            "optimizer": optimizer,
            "batch_size": batch_size,
            "lbfgs_tolerance": lbfgs_tolerance,
            "lr": lr,
            "epochs": epochs_maximization,
        }

        super().__init__(
            model_class=SimpleMNL,
            model_parameters=model_parameters,
            n_latent_classes=n_latent_classes,
            fit_method=fit_method,
            epochs=epochs,
            add_exit_choice=add_exit_choice,
            lbfgs_tolerance=lbfgs_tolerance,
            optimizer=optimizer,
            lr=lr,
            **kwargs,
        )

    def instantiate_latent_models(self, n_items, n_shared_features, n_items_features):
        """Instantiate the Latent Models that are SimpleMNLs.

        Parameters
        ----------
        n_items : int
            Number of items/aternatives to consider.
        n_shared_features : int
            Number of shared features
        n_items_features : int
            Number of items features
        """
        for model in self.models:
            model.indexes, model.weights = model.instantiate(
                n_items, n_shared_features, n_items_features
            )
            model.exact_nll = tf_ops.CustomCategoricalCrossEntropy(
                from_logits=False,
                label_smoothing=0.0,
                sparse=False,
                axis=-1,
                epsilon=1e-25,
                name="exact_categorical_crossentropy",
                reduction="sum_over_batch_size",
            )
            model.instantiated = True

    def instantiate(self, n_items, n_shared_features, n_items_features):
        """Instantiate the Latent Class MNL model."""
        self.latent_logits = tf.Variable(
            tf.random_normal_initializer(0.0, 0.02, seed=42)(shape=(self.n_latent_classes - 1,)),
            name="Latent-Logits",
        )

        self.models = [self.model_class(**mp) for mp in self.model_parameters]

        self.instantiate_latent_models(
            n_items=n_items,
            n_shared_features=n_shared_features,
            n_items_features=n_items_features,
        )
        self.instantiated = True

    def fit(self, choice_dataset, **kwargs):
        """Fit the model to the choice_dataset.

        Parameters
        ----------
        choice_dataset : ChoiceDataset
            Dataset to fit the model to.
        """
        if not self.instantiated:
            self.instantiate(
                n_items=choice_dataset.get_n_items(),
                n_shared_features=choice_dataset.get_n_shared_features(),
                n_items_features=choice_dataset.get_n_items_features(),
            )
        return super().fit(choice_dataset, **kwargs)


class LatentClassConditionalLogit(BaseLatentClassModel):
    """Latent Class for ConditionalLogit."""

    def __init__(
        self,
        n_latent_classes,
        fit_method,
        coefficients=None,
        epochs=100,
        add_exit_choice=False,
        lbfgs_tolerance=1e-6,
        optimizer="Adam",
        lr=0.001,
        **kwargs,
    ):
        """Initialize model.

        Parameters
        ----------
        n_latent_classes : int
            Number of latent classes.
        fit_method : str
            Method to be used to estimate the model.
        coefficients : dict or MNLCoefficients
            Dictionnary containing the parametrization of the model.
            The dictionnary must have the following structure:
            {feature_name_1: mode_1, feature_name_2: mode_2, ...}
            mode must be among "constant", "item", "item-full" for now
            (same specifications as torch-choice).
        epochs : int
            Number of epochs
        add_exit_choice : bool, optional
            Whether to normalize probabilities with exit choice, by default False
        lbfgs_tolerance : float, optional
            LBFG-S tolerance, by default 1e-6
        optimizer : str, optional
            tf.keras.optimizers to be used, by default "Adam"
        lr : float, optional
            Learning rate to use for optimizer if relevant, by default 0.001
        """
        self.n_latent_classes = n_latent_classes
        self.fit_method = fit_method
        self.coefficients = coefficients
        self.epochs = epochs
        self.add_exit_choice = add_exit_choice
        self.lbfgs_tolerance = lbfgs_tolerance
        self.optimizer = optimizer
        self.lr = lr

        model_coefficients = {
            "coefficients": self.coefficients,
            "add_exit_choice": self.add_exit_choice,
            "optimizer": self.optimizer,
            "lbfgs_tolerance": self.lbfgs_tolerance,
            "lr": self.lr,
            "epochs": self.epochs,
        }

        super().__init__(
            model_class=ConditionalLogit,
            model_parameters=model_coefficients,
            n_latent_classes=n_latent_classes,
            fit_method=fit_method,
            epochs=epochs,
            add_exit_choice=add_exit_choice,
            lbfgs_tolerance=lbfgs_tolerance,
            optimizer=optimizer,
            lr=lr,
            **kwargs,
        )

    def instantiate_latent_models(self, choice_dataset):
        """Instantiate of the Latent Models that are SimpleMNLs.

        Parameters
        ----------
        choice_dataset: ChoiceDataset
            Used to match the features names with the model coefficients.
        """
        for model in self.models:
            model.coefficients = copy.deepcopy(self.coefficients)
            model.instantiate(choice_dataset)

    def instantiate(self, choice_dataset):
        """Instantiate of the Latent Class MNL model."""
        self.latent_logits = tf.Variable(
            tf.random_normal_initializer(0.0, 0.02, seed=42)(shape=(self.n_latent_classes - 1,)),
            name="Latent-Logits",
        )

        self.models = [self.model_class(**mp) for mp in self.model_parameters]

        self.instantiate_latent_models(choice_dataset)

    def add_coefficients(
        self, coefficient_name, feature_name, items_indexes=None, items_names=None
    ):
        """Add a coefficient to the model throught the specification of the utility.

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

        Raises
        ------
        ValueError
            When names or indexes are both not specified.
        """
        if self.coefficients is None:
            self.coefficients = MNLCoefficients()
        elif not isinstance(self.coefficients, MNLCoefficients):
            raise ValueError("Cannot add coefficient on top of a dict instantiation.")
        self.coefficients.add(
            coefficient_name=coefficient_name,
            feature_name=feature_name,
            items_indexes=items_indexes,
            items_names=items_names,
        )

    def add_shared_coefficient(
        self, coefficient_name, feature_name, items_indexes=None, items_names=None
    ):
        """Add a single, shared coefficient to the model throught the specification of the utility.

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

        Raises
        ------
        ValueError
            When names or indexes are both not specified.
        """
        if self.coefficients is None:
            self.coefficients = MNLCoefficients()
        elif not isinstance(self.coefficients, MNLCoefficients):
            raise ValueError("Cannot add shared coefficient on top of a dict instantiation.")
        self.coefficients.add_shared(
            coefficient_name=coefficient_name,
            feature_name=feature_name,
            items_indexes=items_indexes,
            items_names=items_names,
        )

    def fit(self, choice_dataset, **kwargs):
        """Fit the model to the choice_dataset.

        Parameters
        ----------
        choice_dataset : ChoiceDataset
            Dataset to fit the model to.
        """
        if not self.instantiated:
            self.instantiate(choice_dataset=choice_dataset)
        return super().fit(choice_dataset, **kwargs)
