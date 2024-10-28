"""
Implementation of Enhancing Discrete Choice Models with Representation Learning.

https://arxiv.org/abs/1812.09747 .
"""

import logging

import tensorflow as tf

from .conditional_logit import ConditionalLogit


class LearningMNL(ConditionalLogit):
    """Learning MNL from paper https://arxiv.org/abs/1812.09747 .

    Arguments:
    ----------
    coefficients: dict or MNLCoefficients
        Specfication of the model to be estimated.
    """

    def __init__(
        self,
        coefficients=None,
        nn_features=[],
        nn_layers_widths=[],
        nn_activation="relu",
        add_exit_choice=False,
        optimizer="Adam",
        lr=0.001,
        **kwargs,
    ):
        """Initialize of Conditional-MNL.

        Parameters
        ----------
        coefficients : dict or MNLCoefficients
            Dictionnary containing the coefficients parametrization of the model.
            The dictionnary must have the following structure:
            {feature_name_1: mode_1, feature_name_2: mode_2, ...}
            mode must be among "constant", "item", "item-full" for now
            (same specifications as torch-choice).
        nn_features: list of str
            List of features names that will be used in the neural network.
            Features used as NN inputs must be shared_features !
        nn_layers_widths: list of int
            List of integers representing the width of each hidden layer in the neural network.
        add_exit_choice : bool, optional
            Whether or not to normalize the probabilities computation with an exit choice
            whose utility would be 1, by default True
        """
        super().__init__(add_exit_choice=add_exit_choice, optimizer=optimizer, lr=lr, **kwargs)
        self.coefficients = coefficients
        self.nn_features = nn_features
        self.nn_layers_widths = nn_layers_widths
        self.nn_activation = nn_activation
        self.instantiated = False

    def instantiate(self, choice_dataset):
        """Instantiate the model using the features in the choice_dataset.

        Parameters
        ----------
        choice_dataset: ChoiceDataset
            Used to match the features names with the model coefficients.
        """
        if not self.instantiated:
            # Instantiate NN
            nn_input = tf.keras.Input(shape=(len(self.nn_features), 1, 1))
            nn_output = tf.keras.layers.Conv2D(
                filters=self.nn_layers_widths[0],
                kernel_size=[len(self.nn_features), 1],
                activation="relu",
                padding="valid",
                name="Dense_NN_per_frame",
            )(nn_input)
            nn_output = tf.keras.layers.Dropout(0.2, name="Regularizer")(nn_output)
            nn_output = tf.reshape(nn_output, (-1, self.nn_layers_widths[0]))

            for i in range(len(self.nn_layers_widths) - 1):
                nn_output = tf.keras.layers.Dense(
                    units=self.nn_layers_widths[i + 1], activation="relu", name="Dense{}".format(i)
                )(nn_output)
                nn_output = tf.keras.layers.ropout(0.2, name="Drop{}".format(i))(nn_output)
            nn_output = tf.keras.layers.Dense(
                units=choice_dataset.get_n_items(), name="Output_new_feature"
            )(nn_output)

            # nn_input = tf.keras.Input(shape=(len(self.nn_features), ))
            # x = nn_input
            # for width in self.nn_layers_widths:
            #     x = tf.keras.layers.Dense(width, activation=self.nn_activation)(x)
            #     x = tf.keras.layers.Dropout(0.2, name="Regularizer")(x)
            # nn_output = tf.keras.layers.Dense(choice_dataset.get_n_items())(x)
            self.nn_model = tf.keras.Model(inputs=nn_input, outputs=nn_output)

            super().instantiate(choice_dataset)

    @property
    def trainable_weights(self):
        """Trainable weights of the model."""
        return self._trainable_weights + self.nn_model.trainable_variables

    def compute_batch_utility(
        self,
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices,
        verbose=1,
    ):
        """Compute the utility when the model is constructed from a MNLCoefficients object.

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
        choices: np.ndarray
            Choices
            Shape must be (n_choices, )
        verbose : int, optional
            Parametrization of the logging outputs, by default 1

        Returns
        -------
        tf.Tensor
            Utilities corresponding of shape (n_choices, n_items)
        """
        if not isinstance(shared_features_by_choice, tuple):
            shared_features_by_choice = (shared_features_by_choice,)
        if not isinstance(items_features_by_choice, tuple):
            items_features_by_choice = (items_features_by_choice,)
        knowledge_driven_utilities = super().compute_batch_utility(
            shared_features_by_choice,
            items_features_by_choice,
            available_items_by_choice,
            choices,
            verbose=verbose,
        )
        data_driven_inputs = []
        if self._shared_features_by_choice_names is not None:
            for nn_feature in self.nn_features:
                for i, feat_tuple in enumerate(self._shared_features_by_choice_names):
                    for j, feat in enumerate(feat_tuple):
                        if feat == nn_feature:
                            data_driven_inputs.append(shared_features_by_choice[i][:, j])
        else:
            logging.warn("No shared features found in the dataset.")
        data_driven_utilities = self.nn_model(
            tf.expand_dims(tf.expand_dims(tf.stack(data_driven_inputs, axis=1), axis=-1), axis=-1)
        )

        return tf.reduce_sum(knowledge_driven_utilities, axis=0) + data_driven_utilities

    def clone(self):
        """Return a clone of the model."""
        clone = LearningMNL(
            coefficients=self.coefficients,
            add_exit_choice=self.add_exit_choice,
            optimizer=self.optimizer_name,
            nn_features=self.nn_features,
            nn_layers_widths=self.nn_layers_widths,
            nn_activation=self.nn_activation,
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
        if hasattr(self, "trainable_weights"):
            clone._trainable_weights = self.trainable_weights
        if hasattr(self, "nn_model"):
            clone.nn_model = self.nn_model
        if hasattr(self, "lr"):
            clone.lr = self.lr
        if hasattr(self, "_shared_features_by_choice_names"):
            clone._shared_features_by_choice_names = self._shared_features_by_choice_names
        if hasattr(self, "_items_features_by_choice_names"):
            clone._items_features_by_choice_names = self._items_features_by_choice_names
        if hasattr(self, "_items_features_by_choice_names"):
            clone._items_features_by_choice_names = self._items_features_by_choice_names
        return clone
