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
        intercept=None,
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
        self.intercept = intercept

    def instantiate(
        self, n_items, n_fixed_items_features, n_contexts_features, n_contexts_items_features
    ):
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
        weights = []
        indexes = {}
        for n_feat, feat_name in zip(
            [n_fixed_items_features, n_contexts_features, n_contexts_items_features],
            ["items", "contexts", "contexts_items"],
        ):
            if n_feat > 0:
                weights = [
                    tf.Variable(
                        tf.random_normal_initializer(0.0, 0.02, seed=42)(shape=(n_feat,)),
                        name=f"Weights_{feat_name}",
                    )
                ]
                indexes[feat_name] = len(weights) - 1
        if self.intercept is None:
            print("No intercept in the model")
        elif self.intercept == "item":
            weights.append(
                tf.Variable(
                    tf.random_normal_initializer(0.0, 0.02, seed=42)(shape=(n_items - 1,)),
                    name="Intercept",
                )
            )
            indexes["intercept"] = len(weights) - 1
        elif self.intercept == "item-full":
            print("Are you sure you do not want to normalize an intercept to 0?")
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

        self.instantiated = True
        return indexes, weights

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
        if "items" in self.indexes.keys():
            if isinstance(fixed_items_features, tuple):
                fixed_items_features = tf.concat(*fixed_items_features, axis=1)
            fixed_items_utilities = tf.tensordot(
                fixed_items_features, self.weights[self.indexes["items"]], axes=1
            )
        else:
            fixed_items_utilities = 0

        if "contexts" in self.indexes.keys():
            if isinstance(contexts_features, tuple):
                contexts_features = tf.concat(*contexts_features, axis=1)
            contexts_utilities = tf.tensordot(
                contexts_features, self.weights[self.indexes["contexts"]], axes=1
            )
            contexts_utilities = tf.expand_dims(contexts_utilities, axis=0)
        else:
            contexts_utilities = 0

        if "contexts_items" in self.indexes.keys():
            if isinstance(contexts_items_features, tuple):
                contexts_items_features = tf.concat([*contexts_items_features], axis=2)
            contexts_items_utilities = tf.tensordot(
                contexts_items_features, self.weights[self.indexes["contexts_items"]], axes=1
            )
        else:
            contexts_utilities = tf.zeros(
                (contexts_utilities.shape[0], fixed_items_utilities.shape[1], 1)
            )

        if "intercept" in self.indexes.keys():
            intercept = self.weights[self.indexes["intercept"]]
            if self.intercept == "item":
                intercept = tf.concat([tf.constant([0.0]), intercept], axis=0)
            if self.intercept in ["item", "item-full"]:
                intercept = tf.expand_dims(intercept, axis=0)
        else:
            intercept = 0

        return fixed_items_utilities + contexts_utilities + contexts_items_utilities + intercept

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
            print("Instantiation")
            self.indexes, self.weights = self.instantiate(
                n_items=choice_dataset.get_n_items(),
                n_fixed_items_features=choice_dataset.get_n_fixed_items_features(),
                n_contexts_features=choice_dataset.get_n_contexts_features(),
                n_contexts_items_features=choice_dataset.get_n_contexts_items_features(),
            )
            self.instantiated = True
        fit = super().fit(choice_dataset=choice_dataset, **kwargs)
        if get_report:
            self.report = self.compute_report(choice_dataset)
        return fit

    def _fit_with_lbfgs(
        self, choice_dataset, epochs=None, sample_weight=None, get_report=False, **kwargs
    ):
        """Specific fit function to estimate the paramters with LBFGS.

        Parameters
        ----------
        choice_dataset : ChoiceDataset
            Choice dataset to use for the estimation.
        n_epochs : int
            Number of epochs to run.
        sample_weight: Iterable, optional
            list of each sample weight, by default None meaning that all samples have weight 1.
        get_report: bool, optional
            Whether or not to compute a report of the estimation, by default False.

        Returns:
        --------
        conditionalMNL
            self with estimated weights.
        """
        if not self.instantiated:
            # Lazy Instantiation
            print("Instantiation")
            self.indexes, self.weights = self.instantiate(
                n_items=choice_dataset.get_n_items(),
                n_fixed_items_features=choice_dataset.get_n_fixed_items_features(),
                n_contexts_features=choice_dataset.get_n_contexts_features(),
                n_contexts_items_features=choice_dataset.get_n_contexts_items_features(),
            )
            self.instantiated = True
        if epochs is None:
            epochs = self.epochs
        fit = super()._fit_with_lbfgs(
            dataset=choice_dataset, epochs=epochs, sample_weight=sample_weight, **kwargs
        )
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
            for j in range(weight.shape[0]):
                names.append(f"{weight.name}_{j}")
                estimations.append(weight.numpy()[j])
                z_values.append(weight.numpy()[j] / weights_std[i].numpy())
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
                w = tf.concat(self.weights, axis=0)
                tape_2.watch(w)
                tape_1.watch(w)
                mw = []
                index = 0
                for _w in self.weights:
                    mw.append(w[index : index + _w.shape[0]])
                    index += _w.shape[0]
                model.weights = mw
                for batch in dataset.iter_batch(batch_size=-1):
                    utilities = model.compute_batch_utility(*batch)
                    probabilities = tf.nn.softmax(utilities, axis=-1)
                    loss = tf.keras.losses.CategoricalCrossentropy(reduction="sum")(
                        y_pred=probabilities,
                        y_true=tf.one_hot(dataset.choices, depth=probabilities.shape[-1]),
                    )
            # Compute the Jacobian
            jacobian = tape_2.jacobian(loss, w)
        # Compute the Hessian from the Jacobian
        hessian = tape_1.jacobian(jacobian, w)
        hessian = tf.linalg.inv(tf.squeeze(hessian))
        return tf.sqrt([hessian[i][i] for i in range(len(tf.squeeze(hessian)))])

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
        if hasattr(self, "indexes"):
            clone.indexes = self.indexes
        if hasattr(self, "intercept"):
            clone.intercept = self.intercept
        if hasattr(self, "lr"):
            clone.lr = self.lr
        if hasattr(self, "_items_features_names"):
            clone._items_features_names = self._items_features_names
        if hasattr(self, "_contexts_features_names"):
            clone._contexts_features_names = self._contexts_features_names
        if hasattr(self, "_contexts_items_features_names"):
            clone._contexts_items_features_names = self._contexts_items_features_names
        return clone