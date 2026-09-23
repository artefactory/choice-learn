"""Halo MNL model."""

import math

import pandas as pd
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

        u_mat = tf.Variable((tf.random.normal((n_items, self.halo_latent_dim))), name="U")
        v_mat = tf.Variable((tf.random.normal((self.halo_latent_dim, n_items))), name="V")
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
        halo = tf.linalg.set_diag(halo, self.zero_diag)
        halo = tf.linalg.matmul(available_items_by_choice, halo)
        return items_utilities + halo

    def get_weights_std(self, choice_dataset):
        """Approximates Std Err with Hessian matrix.

        Parameters
        ----------
        choice_dataset : ChoiceDataset
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
                w = tf.concat(self.trainable_weights[:-2], axis=0)
                tape_2.watch(w)
                tape_1.watch(w)
                mw = []
                index = 0
                for _w in self.trainable_weights:
                    mw.append(w[index : index + _w.shape[0]])
                    index += _w.shape[0]
                model._trainable_weights = mw + [
                    self.trainable_weights[-2],
                    self.trainable_weights[-1],
                ]
                for batch in choice_dataset.iter_batch(batch_size=-1):
                    utilities = model.compute_batch_utility(*batch)
                    probabilities = tf.nn.softmax(utilities, axis=-1)
                    loss = tf.keras.losses.CategoricalCrossentropy(reduction="sum")(
                        y_pred=probabilities,
                        y_true=tf.one_hot(choice_dataset.choices, depth=probabilities.shape[-1]),
                    )
            # Compute the Jacobian
            jacobian = tape_2.jacobian(loss, w)
        # Compute the Hessian from the Jacobian
        hessian = tape_1.jacobian(jacobian, w)
        hessian = tf.linalg.inv(tf.squeeze(hessian))
        return tf.sqrt([hessian[i][i] for i in range(len(tf.squeeze(hessian)))])

    def compute_report(self, choice_dataset):
        """Compute a report of the estimated weights.

        Parameters
        ----------
        choice_dataset : ChoiceDataset
            ChoiceDataset used for the estimation of the weights that will be
            used to compute the Std Err of this estimation.

        Returns
        -------
        pandas.DataFrame
            A DF with estimation, Std Err, z_value and p_value for each coefficient.
        """

        def phi(x):
            """Cumulative distribution function for the standard normal distribution."""
            return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

        weights_std = self.get_weights_std(choice_dataset)

        names = []
        z_values = []
        estimations = []
        p_z = []
        i = 0
        for weight in self.trainable_weights[:-2]:
            for j in range(weight.shape[0]):
                if weight.shape[0] > 1:
                    names.append(f"{weight.name[:-2]}_{j}")
                else:
                    names.append(f"{weight.name[:-2]}")
                estimations.append(weight.numpy()[j])
                z_values.append(weight.numpy()[j] / weights_std[i].numpy())
                p_z.append(2 * (1 - phi(tf.math.abs(z_values[-1]).numpy())))
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


class HaloMNL(SimpleMNL):
    """Implementation of Low Rank Halo MNL model."""

    def __init__(
        self,
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

        halo_matrix = tf.Variable((tf.random.normal((n_items, n_items))), name="halo_matrix")
        self.zero_diag = tf.zeros(n_items)
        # halo_matrix = tf.linalg.set_diag(halo_matrix, self.zero_diag)
        weights += [halo_matrix]

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
        halo = tf.linalg.matmul(
            available_items_by_choice,
            tf.linalg.set_diag(self.trainable_weights[-1], self.zero_diag),
        )
        return items_utilities + halo

    def get_weights_std(self, choice_dataset):
        """Approximates Std Err with Hessian matrix.

        Parameters
        ----------
        choice_dataset : ChoiceDataset
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
                w = tf.concat(self.trainable_weights[:-1], axis=0)
                tape_2.watch(w)
                tape_1.watch(w)
                mw = []
                index = 0
                for _w in self.trainable_weights:
                    mw.append(w[index : index + _w.shape[0]])
                    index += _w.shape[0]
                model._trainable_weights = mw + [
                    self.trainable_weights[-1],
                ]
                for batch in choice_dataset.iter_batch(batch_size=-1):
                    utilities = model.compute_batch_utility(*batch)
                    probabilities = tf.nn.softmax(utilities, axis=-1)
                    loss = tf.keras.losses.CategoricalCrossentropy(reduction="sum")(
                        y_pred=probabilities,
                        y_true=tf.one_hot(choice_dataset.choices, depth=probabilities.shape[-1]),
                    )
            # Compute the Jacobian
            jacobian = tape_2.jacobian(loss, w)
        # Compute the Hessian from the Jacobian
        hessian = tape_1.jacobian(jacobian, w)
        hessian = tf.linalg.inv(tf.squeeze(hessian))
        return tf.sqrt([hessian[i][i] for i in range(len(tf.squeeze(hessian)))])

    def compute_report(self, choice_dataset):
        """Compute a report of the estimated weights.

        Parameters
        ----------
        choice_dataset : ChoiceDataset
            ChoiceDataset used for the estimation of the weights that will be
            used to compute the Std Err of this estimation.

        Returns
        -------
        pandas.DataFrame
            A DF with estimation, Std Err, z_value and p_value for each coefficient.
        """

        def phi(x):
            """Cumulative distribution function for the standard normal distribution."""
            return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

        weights_std = self.get_weights_std(choice_dataset)

        names = []
        z_values = []
        estimations = []
        p_z = []
        i = 0
        for weight in self.trainable_weights[:-1]:
            for j in range(weight.shape[0]):
                if weight.shape[0] > 1:
                    names.append(f"{weight.name[:-2]}_{j}")
                else:
                    names.append(f"{weight.name[:-2]}")
                estimations.append(weight.numpy()[j])
                z_values.append(weight.numpy()[j] / weights_std[i].numpy())
                p_z.append(2 * (1 - phi(tf.math.abs(z_values[-1]).numpy())))
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
