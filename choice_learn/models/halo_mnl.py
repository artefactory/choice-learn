"""Halo MNL model."""

import logging
import math

import numpy as np
import pandas as pd
import tensorflow as tf

from .simple_mnl import SimpleMNL
from .conditional_logit import ConditionalLogit


class LowRankHaloMNL(SimpleMNL):
    def __init__(
        self,
        halo_latent_dim,
        add_exit_choice=False,
        intercept=None,
        optimizer="lbfgs",
        lr=0.001,
        **kwargs,
    ):
        """Initialize of Simple-MNL.

        Parameters
        ----------
        add_exit_choice : bool, optional
            Whether or not to normalize the probabilities computation with an exit choice
            whose utility would be 1, by default True
        intercept: str, optional
            Type of intercept to use, by default None
        optimizer: str
            TensorFlow optimizer to be used for estimation
        lr: float
            Learning Rate to be used with optimizer.
        """
        super().__init__(add_exit_choice=add_exit_choice, optimizer=optimizer, lr=lr, **kwargs)

        self.halo_latent_dim = halo_latent_dim
        self.instantiated = False
        self.intercept = intercept

    def instantiate(self, n_items, n_shared_features, n_items_features):
        """Instantiate the model from ModelSpecification object.

        Parameters
        ----------
        n_items : int
            Number of items/aternatives to consider.
        n_shared_features : int
            Number of contexts features
        n_items_features : int
            Number of contexts items features

        Returns
        -------
        list of tf.Tensor
            List of the weights created coresponding to the specification.
        """
        weights = []
        indexes = {}
        for n_feat, feat_name in zip(
            [n_shared_features, n_items_features],
            ["shared_features", "items_features"],
        ):
            if n_feat > 0:
                weights += [
                    tf.Variable(
                        tf.random_normal_initializer(0.0, 0.02, seed=42)(shape=(n_feat,)),
                        name=f"Weights_{feat_name}",
                    )
                ]
                indexes[feat_name] = len(weights) - 1
        if self.intercept is None:
            logging.info("No intercept in the model")
        elif self.intercept == "item":
            weights.append(
                tf.Variable(
                    tf.random_normal_initializer(0.0, 0.02, seed=42)(shape=(n_items - 1,)),
                    name="Intercept",
                )
            )
            indexes["intercept"] = len(weights) - 1
        elif self.intercept == "item-full":
            logging.info("Simple MNL intercept is not normalized to 0!")
            weights.append(
                tf.Variable(
                    tf.random_normal_initializer(0.0, 0.02, seed=42)(shape=(n_items,)),
                    name="Intercept",
                )
            )
            indexes["intercept"] = len(weights) - 1
        else:
            weights.append(
                tf.Variable(
                    tf.random_normal_initializer(0.0, 0.02, seed=42)(shape=(1,)),
                    name="Intercept",
                )
            )
            indexes["intercept"] = len(weights) - 1

        alpha = tf.Variable((tf.random.normal((1, n_items))), name="alpha")
        U = tf.Variable((tf.random.normal((n_items, self.halo_latent_dim))), name="U")
        V = tf.Variable((tf.random.normal((n_items, self.halo_latent_dim))), name="V")
        weights += [alpha, U, V]

        self.instantiated = True
        self.indexes = indexes
        self._trainable_weights = weights
        return indexes, weights

    def compute_batch_utility(
        self,
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices,
    ):
        """Compute the utility of the model. Selects the right method to compute.

        Parameters
        ----------
        shared_features_by_choice : tuple of np.ndarray (choices_features)
            a batch of shared features
            Shape must be (n_choices, n_shared_features)
        items_features_by_choice : tuple of np.ndarray (choices_items_features)
            a batch of items features
            Shape must be (n_choices, n_items, n_items_features)
        available_items_by_choice : np.ndarray
            A batch of items availabilities
            Shape must be (n_choices, n_items)
        choices : np.ndarray
            Choices
            Shape must be (n_choices, )

        Returns
        -------
        tf.Tensor
            Computed utilities of shape (n_choices, n_items).
        """
        _ = choices

        if "shared_features" in self.indexes.keys():
            if isinstance(shared_features_by_choice, tuple):
                shared_features_by_choice = tf.concat(*shared_features_by_choice, axis=1)
            shared_features_by_choice = tf.cast(shared_features_by_choice, tf.float32)
            shared_features_utilities = tf.tensordot(
                shared_features_by_choice,
                self.trainable_weights[self.indexes["shared_features"]],
                axes=1,
            )
            shared_features_utilities = tf.expand_dims(shared_features_utilities, axis=-1)
        else:
            shared_features_utilities = 0

        if "items_features" in self.indexes.keys():
            if isinstance(items_features_by_choice, tuple):
                items_features_by_choice = tf.concat([*items_features_by_choice], axis=2)
            items_features_by_choice = tf.cast(items_features_by_choice, tf.float32)
            items_features_utilities = tf.tensordot(
                items_features_by_choice,
                self.trainable_weights[self.indexes["items_features"]],
                axes=1,
            )
        else:
            items_features_utilities = tf.zeros(
                (available_items_by_choice.shape[0], available_items_by_choice.shape[1])
            )

        if "intercept" in self.indexes.keys():
            intercept = self.trainable_weights[self.indexes["intercept"]]
            if self.intercept == "item":
                intercept = tf.concat([tf.constant([0.0]), intercept], axis=0)
            if self.intercept in ["item", "item-full"]:
                intercept = tf.expand_dims(intercept, axis=0)
        else:
            intercept = 0

        items_utilities = shared_features_utilities + items_features_utilities + intercept

        halo = tf.tensordot(
            self.trainable_weights[-2], tf.transpose(self.trainable_weights[-1]), axes=1
        )
        return (
            items_utilities * self.trainable_weights[-3]
            + tf.tensordot(items_utilities, halo, axes=1) * available_items_by_choice
        )
