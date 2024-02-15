"""Implementation of the simple linear multinomial logit model.

It is a multi output logistic regression.
"""

import pandas as pd
import tensorflow as tf

from .base_model import ChoiceModel


class SimpleMNL(ChoiceModel):
    """Simple MNL with one linear coefficient to estimate by feature."""

    def __init__(
        self,
        add_exit_choice=False,
        optimizer="Adam",
        lr=0.001,
        **kwargs,
    ):
        """Initialization of Simple-MNL.

        Parameters:
        -----------
        add_exit_choice : bool, optional
            Whether or not to normalize the probabilities computation with an exit choice
            whose utility would be 1, by default True
        optimizer: str
            TensorFlow optimizer to be used for estimation
        lr: float
            Learning Rate to be used with optimizer.
        """
        super().__init__(normalize_non_buy=add_exit_choice, optimizer=optimizer, lr=lr, **kwargs)
        self.instantiated = False

    def instantiate(self, n_fixed_items_features, n_contexts_features, n_contexts_items_features):
        """Instantiate the model from ModelSpecification object.

        Paramters
        --------
        n_weights: int
            Number of weights to be estimated. Corresponds to the number of features.

        Returns:
        --------
        list of tf.Tensor
            List of the weights created coresponding to the specification.
        """
        weights = [
            tf.Variable(
                tf.random_normal_initializer(0.0, 0.02, seed=42)(shape=(1, n_feat)),
                name="Weights",
            )
            for n_feat in [n_fixed_items_features, n_contexts_features, n_contexts_items_features]
        ]
        self.instantiated = True
        return weights

    def compute_batch_utility(
        self,
        fixed_items_features,
        contexts_features,
        contexts_items_features,
        contexts_items_availabilities,
        choices,
    ):
        """Main method to compute the utility of the model. Selects the right method to compute.

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
            Computed utilities of shape (n_choices, n_items).
        """
        _, _ = contexts_items_availabilities, choices
        if len(self.weights[0]) > 0:
            fixed_items_features = tf.concat(*fixed_items_features, axis=1)
            fixed_items_utilities = tf.tensordot(fixed_items_features, self.weights[0], axes=1)
        else:
            fixed_items_utilities = 0

        if len(self.weights[1]) > 0:
            contexts_features = tf.concat(*contexts_features, axis=1)
            contexts_utilities = tf.tensordot(contexts_features, self.weights[1], axes=1)
            contexts_utilities = tf.expand_dims(contexts_utilities, axis=0)
        else:
            contexts_utilities = 0

        if len(self.weights[2]) > 0:
            contexts_items_features = tf.concat([*contexts_items_features], axis=2)
            contexts_items_utilities = tf.tensordot(contexts_items_features, self.weights[2])
        else:
            contexts_utilities = tf.zeros(
                (contexts_utilities.shape[0], fixed_items_utilities.shape[1], 1)
            )

        return fixed_items_utilities + contexts_utilities + contexts_items_utilities

    def fit(self, choice_dataset, get_report=False, **kwargs):
        """Main fit function to estimate the paramters.

        Parameters
        ----------
        choice_dataset : ChoiceDataset
            Choice dataset to use for the estimation.
        get_report: bool, optional
            Whether or not to compute a report of the estimation, by default False

        Returns:
        --------
        ConditionalMNL
            With estimated weights.
        """
        if not self.instantiated:
            # Lazy Instantiation
            self.weights = self.instantiate(
                n_items_features=choice_dataset.get_n_fixed_items_features(),
                n_context_features=choice_dataset.get_n_contexts_features(),
                n_contexts_items_features=choice_dataset.get_n_contexts_items_features(),
            )
            self.instantiated = True
        fit = super().fit(choice_dataset=choice_dataset, **kwargs)
        if get_report:
            self.report = self.compute_report(choice_dataset)
        return fit

    def compute_report(self, dataset):
        """Computes a report of the estimated weights.

        Parameters
        ----------
        dataset : ChoiceDataset
            ChoiceDataset used for the estimation of the weights that will be
            used to compute the Std Err of this estimation.

        Returns:
        --------
        pandas.DataFrame
            A DF with estimation, Std Err, z_value and p_value for each coefficient.
        """
        import tensorflow_probability as tfp

        weights_std = self.get_weights_std(dataset)
        dist = tfp.distributions.Normal(loc=0.0, scale=1.0)

        names = []
        z_values = []
        estimations = []
        p_z = []
        i = 0
        for weight in self.weights:
            for j in range(weight.shape[1]):
                names.append(f"{weight.name}_{j}")
                estimations.append(weight.numpy()[0][j])
                z_values.append(weight.numpy()[0][j] / weights_std[i].numpy())
                p_z.append(2 * (1 - dist.cdf(tf.math.abs(z_values[-1])).numpy()))
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

    def get_weights_std(self, dataset):
        """Approximates Std Err with Hessian matrix.

        Parameters
        ----------
        dataset : ChoiceDataset
            ChoiceDataset used for the estimation of the weights that will be
            used to compute the Std Err of this estimation.

        Returns:
        --------
        tf.Tensor
            Estimation of the Std Err for the weights.
        """
        # Loops of differentiation
        with tf.GradientTape() as tape_1:
            with tf.GradientTape(persistent=True) as tape_2:
                model = self.clone()
                w = tf.concat(self.weights, axis=1)
                tape_2.watch(w)
                tape_1.watch(w)
                mw = []
                index = 0
                for _w in self.weights:
                    mw.append(w[:, index : index + _w.shape[1]])
                    index += _w.shape[1]
                model.weights = mw
                for batch in dataset.iter_batch(batch_size=-1):
                    utilities = model.compute_batch_utility(*batch)
                    probabilities = tf.nn.softmax(utilities, axis=-1)
                    loss = tf.keras.losses.CategoricalCrossentropy(reduction="sum")(
                        y_pred=probabilities,
                        y_true=tf.one_hot(dataset.choices, depth=4),
                    )
            # Compute the Jacobian
            jacobian = tape_2.jacobian(loss, w)
        # Compute the Hessian from the Jacobian
        hessian = tape_1.batch_jacobian(jacobian, w)
        return tf.sqrt(
            [tf.linalg.inv(tf.squeeze(hessian))[i][i] for i in range(len(tf.squeeze(hessian)))]
        )

    def clone(self):
        """Returns a clone of the model."""
        clone = SimpleMNL(
            add_exit_choice=self.normalize_non_buy,
            optimizer=self.optimizer_name,
        )
        if hasattr(self, "history"):
            clone.history = self.history
        if hasattr(self, "is_fitted"):
            clone.is_fitted = self.is_fitted
        if hasattr(self, "instantiated"):
            clone.instantiated = self.instantiated
        clone.loss = self.loss
        clone.label_smoothing = self.label_smoothing
        if hasattr(self, "report"):
            clone.report = self.report
        if hasattr(self, "weights"):
            clone.weights = self.weights
        if hasattr(self, "lr"):
            clone.lr = self.lr
        if hasattr(self, "_items_features_names"):
            clone._items_features_names = self._items_features_names
        if hasattr(self, "_contexts_features_names"):
            clone._contexts_features_names = self._contexts_features_names
        if hasattr(self, "_contexts_items_features_names"):
            clone._contexts_items_features_names = self._contexts_items_features_names
        return clone
