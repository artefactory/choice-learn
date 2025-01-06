"""Halo MNL model."""

import tensorflow as tf

# from .conditional_logit import ConditionalLogit
from .simple_mnl import SimpleMNL


class LowRankHaloMNL(SimpleMNL):
    """Implementation of Low Rank Halo MNL model."""

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
        indexes, weights = super().instantiate(n_items, n_shared_features, n_items_features)

        u_mat = tf.Variable((tf.zeros((n_items, self.halo_latent_dim))), name="U")
        v_mat = tf.Variable((tf.zeros((self.halo_latent_dim, n_items))), name="V")
        weights += [u_mat, v_mat]

        self.zero_diag = tf.zeros(n_items)
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
        items_utilities = super().compute_batch_utility(
            shared_features_by_choice, items_features_by_choice, available_items_by_choice, choices
        )

        halo = tf.linalg.matmul(self.trainable_weights[-2], self.trainable_weights[-1])
        tf.linalg.set_diag(halo, self.zero_diag)
        return items_utilities + halo
