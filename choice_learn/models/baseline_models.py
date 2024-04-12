"""Models to be used as baselines for choice modeling. Nothing smart here."""
import numpy as np
import tensorflow as tf

from .base_model import ChoiceModel


class RandomChoiceModel(ChoiceModel):
    """Dumb model that randomly attributes utilities to products."""

    def __init__(self, **kwargs):
        """Initialize of the model."""
        super().__init__(**kwargs)

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
            np.random.uniform(shape=(available_items_by_choice.shape), minval=0, maxval=1)
        )

    def fit(**kwargs):
        """Make sure that nothing happens during .fit."""
        _ = kwargs
        return {}


class DistribMimickingModel(ChoiceModel):
    """Dumb class model that mimicks the probabilities.

    It stores the encountered in the train datasets and always returns them
    """

    def __init__(self, **kwargs):
        """Initialize of the model."""
        super().__init__(**kwargs)
        self.weights = []

    def fit(self, choice_dataset, **kwargs):
        """Compute the choice frequency of each product and defines it as choice probabilities."""
        _ = kwargs
        choices = choice_dataset.choices
        for i in range(choice_dataset.get_num_items()):
            self.weights.append(tf.reduce_sum(tf.cast(choices == i, tf.float32)))
        self.weights = tf.stack(self.weights) / len(choices)

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
        if self.weights is None:
            raise ValueError("Model not fitted")
        return np.stack([np.log(self.weights.numpy())] * len(choices), axis=0)
