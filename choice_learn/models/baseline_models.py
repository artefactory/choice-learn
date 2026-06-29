"""Models to be used as baselines for choice modeling. Nothing smart here."""

import numpy as np
import tensorflow as tf

from .base_model import ChoiceModel


class RandomChoiceModel(ChoiceModel):
    """Dumb model that randomly attributes utilities to products."""

    def __init__(self, **kwargs):
        """Initialize of the model."""
        super().__init__(**kwargs)

    @property
    def trainable_weights(self):
        """Return an empty list - there is no trainable weight."""
        return []

    def compute_batch_utility(
        self,
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices,
    ):
        """Compute the random utility for each product of each choice.

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

        Returns
        -------
        tf.Tensor
            (n_choices, n_items) matrix of random utilities
        """
        # In order to avoid unused arguments warnings
        _ = shared_features_by_choice, items_features_by_choice, choices
        return np.squeeze(
            np.random.uniform(size=(available_items_by_choice.shape), low=0.0, high=1.0)
        ).astype(np.float32)

    def fit(self, *args, **kwargs):
        """Make sure that nothing happens during .fit."""
        _ = kwargs
        _ = args
        return {}

    def _fit_with_lbfgs(self, *args, **kwargs):
        """Make sure that nothing happens during .fit."""
        _ = kwargs
        _ = args
        return {}


class DistribMimickingModel(ChoiceModel):
    """Dumb class model that mimicks the probabilities.

    It stores the encountered in the train datasets and always returns them
    """

    def __init__(self, **kwargs):
        """Initialize of the model."""
        super().__init__(**kwargs)
        self._trainable_weights = []
        self.is_fitted = False

    @property
    def trainable_weights(self):
        """Trainable weights of the model."""
        return [self._trainable_weights]

    def fit(self, choice_dataset, *args, **kwargs):
        """Compute the choice frequency of each product and defines it as choice probabilities.

        Parameters
        ----------
        choice_dataset : ChoiceDataset
            Dataset to be used for fitting
        """
        _ = kwargs
        _ = args
        choices = choice_dataset.choices
        for i in range(choice_dataset.get_n_items()):
            self._trainable_weights.append(tf.reduce_sum(tf.cast(choices == i, tf.float32)))
        self._trainable_weights = tf.stack(self._trainable_weights) / len(choices)
        self.is_fitted = True

    def _fit_with_lbfgs(self, choice_dataset, *args, **kwargs):
        """Compute the choice frequency of each product and defines it as choice probabilities.

        Parameters
        ----------
        choice_dataset : ChoiceDataset
            Dataset to be used for fitting
        """
        _ = kwargs
        _ = args
        choices = choice_dataset.choices
        for i in range(choice_dataset.get_n_items()):
            self._trainable_weights.append(tf.reduce_sum(tf.cast(choices == i, tf.float32)))
        self._trainable_weights = tf.stack(self._trainable_weights) / len(choices)
        self.is_fitted = True

    def compute_batch_utility(
        self,
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices,
    ):
        """Return utility that is fixed. U = log(P).

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

        Returns
        -------
        np.ndarray (n_choices, n_items)
            Utilities

        Raises
        ------
        ValueError
            If the model has not been fitted cannot evaluate the utility
        """
        # In order to avoid unused arguments warnings
        _ = items_features_by_choice, shared_features_by_choice, available_items_by_choice
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return tf.stack([tf.math.log(self.trainable_weights[0])] * len(choices), axis=0)
