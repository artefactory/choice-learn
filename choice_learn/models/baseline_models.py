"""Models to be used as baselines for choice modeling. Nothing smart here."""
import numpy as np
import tensorflow as tf

from .base_model import ChoiceModel


class RandomChoiceModel(ChoiceModel):
    """Dumb model that randomly attributes utilities to products."""

    def __init__(self, **kwargs):
        """Initialization of the model."""
        super().__init__(**kwargs)

    def compute_batch_utility(
        self,
        fixed_items_features,
        contexts_features,
        contexts_items_features,
        contexts_items_availabilities,
        choices,
    ):
        """Computes the random utility for each product of each context.

        Parameters
        ----------
        fixed_items_features : tuple of np.ndarray
            Fixed-Item-Features: formatting from ChoiceDataset: a matrix representing the products
            constant/fixed features.
            Shape must be (n_items, n_items_features)
        contexts_features : tuple of np.ndarray (contexts_features)
            a batch of contexts features
            Shape must be (n_contexts, n_contexts_features)
        contexts_items_features : tuple of np.ndarray (contexts_items_features)
            a batch of contexts items features
            Shape must be (n_contexts, n_contexts_items_features)
        contexts_items_availabilities : np.ndarray
            A batch of contexts items availabilities
            Shape must be (n_contexts, n_items)
        choices_batch : np.ndarray
            Choices
            Shape must be (n_contexts, )

        Returns:
        --------
        tf.Tensor
            (n_contexts, n_items) matrix of random utilities
        """
        # In order to avoid unused arguments warnings
        _ = fixed_items_features, contexts_features, contexts_items_availabilities, choices
        return np.squeeze(
            np.random.uniform(shape=(contexts_items_features.shape), minval=0, maxval=1)
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
        """Initialization of the model."""
        super().__init__(**kwargs)
        self.weights = []

    def fit(self, choice_dataset, **kwargs):
        """Computes the choice frequency of each product and defines it as choice probabilities."""
        _ = kwargs
        choices = choice_dataset.choices
        for i in range(choice_dataset.get_num_items()):
            self.weights.append(tf.reduce_sum(tf.cast(choices == i, tf.float32)))
        self.weights = tf.stack(self.weights) / len(choices)

    def compute_batch_utility(
        self,
        fixed_items_features,
        contexts_features,
        contexts_items_features,
        contexts_items_availabilities,
        choices,
    ):
        """Returns utility that is fixed. U = log(P).

        Parameters
        ----------
        fixed_items_features : tuple of np.ndarray
            Fixed-Item-Features: formatting from ChoiceDataset: a matrix representing the products
            constant/fixed features.
            Shape must be (n_items, n_items_features)
        contexts_features : tuple of np.ndarray (contexts_features)
            a batch of contexts features
            Shape must be (n_contexts, n_contexts_features)
        contexts_items_features : tuple of np.ndarray (contexts_items_features)
            a batch of contexts items features
            Shape must be (n_contexts, n_contexts_items_features)
        contexts_items_availabilities : np.ndarray
            A batch of contexts items availabilities
            Shape must be (n_contexts, n_items)
        choices_batch : np.ndarray
            Choices
            Shape must be (n_contexts, )

        Returns:
        --------
        np.ndarray (n_contexts, n_items)
            Utilities

        Raises:
        -------
        ValueError
            If the model has not been fitted cannot evaluate the utility
        """
        # In order to avoid unused arguments warnings
        _ = fixed_items_features, contexts_features, contexts_items_availabilities
        _ = contexts_items_features
        if self.weights is None:
            raise ValueError("Model not fitted")
        return np.stack([np.log(self.weights.numpy())] * len(choices), axis=0)
