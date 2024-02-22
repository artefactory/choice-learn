"""Latent Class MNL models."""

import tensorflow as tf

from .base_model import BaseLatentClassModel
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
        self.fit_method = fit_method
        self.epochs = epochs
        self.add_exit_choice = add_exit_choice
        self.tolerance = tolerance
        self.intercept = intercept
        self.optimizer = optimizer
        self.lr = lr

        model_params = {
            "add_exit_choice": self.add_exit_choice,
            "intercept": self.intercept,
            "optimizer": self.optimizer,
            "tolerance": self.tolerance,
            "lr": self.lr,
            "epochs": self.epochs,
        }

        super().__init__(model_class=SimpleMNL, model_params=model_params, **kwargs)

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
