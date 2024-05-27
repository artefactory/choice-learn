"""TasteNet model unofficial implementation."""
import logging

import tensorflow as tf

from choice_learn.models.base_model import ChoiceModel


def get_feed_forward_net(input_width, output_width, layers_width, activation):
    """Get a feed-forward neural network."""
    net_input = tf.keras.layers.Input(shape=(input_width,))
    net_output = net_input
    for n_units in layers_width:
        net_output = tf.keras.layers.Dense(n_units, activation=activation)(net_output)
    net_output = tf.keras.layers.Dense(output_width, activation=None)(net_output)
    return tf.keras.Model(inputs=net_input, outputs=net_output)


class TasteNet(ChoiceModel):
    """UnOfficial implementation of the TasteNet model.

    A neural-embedded discrete choice model: Learning taste representation with strengthened
    interpretability, by Han, Y.; Calara Oereuran F.; Ben-Akiva, M.; Zegras, C. (2020).
    """

    def __init__(
        self,
        taste_net_layers,
        taste_net_activation,
        items_features_by_choice_parametrization,
        exp_paramater_mu=1.0,
        **kwargs,
    ):
        """Initialize of the model.

        Parameters
        ----------
        taste_net_layers : list of ints
            Width of the different layer to use in the taste network.
        taste_net_activation : str
            Activation function to use in the taste network.
        items_features_by_choice_parametrization : list of lists
            List of list of strings or floats. Each list corresponds to the features of an item.
            Each string is the name of an activation function to apply to the feature.
            Each float is a constant to multiply the feature by.
            e.g. for the swissmetro that has 3 items with 4 features each:
            [[-1., "-exp", "-exp", 0., "linear", 0., 0.],
            [-1., "-exp", "-exp", "linear", 0., "linear", 0.],
            [-1., "-exp", 0., 0., 0., 0., 0.]]
        exp_paramater_mu : float
            Parameter of the exponential function to use in the
            items_features_by_choice_parametrization.
            x = exp(x / exp_paramater_mu), default is 1.0.
        """
        super().__init__(**kwargs)
        self.taste_net_layers = taste_net_layers
        self.taste_net_activation = taste_net_activation
        self.items_features_by_choice_parametrization = items_features_by_choice_parametrization
        self.exp_paramater_mu = exp_paramater_mu

        for item_params in items_features_by_choice_parametrization:
            if len(item_params) != len(items_features_by_choice_parametrization[0]):
                raise ValueError(
                    f"""All items must have the same number of features parametrization.
                                 Found {len(item_params)} and
                                 {len(items_features_by_choice_parametrization[0])}"""
                )
        self.n_items = len(items_features_by_choice_parametrization)
        self.n_items_features = len(items_features_by_choice_parametrization[0])
        logging.info(
            """TasteNet model is instantiated for {self.n_items} items and
                     {self.n_items_features} items_features."""
        )
        self.instantiated = False

    def get_activation_function(self, name):
        """Get a normalization function from its str name.

        Parameters
        ----------
        name : str
            Name of the function to apply.

        Returns
        -------
        function
            Tensorflow function to apply.
        """
        if name == "linear":
            return lambda x: x
        if name == "relu":
            return tf.nn.relu
        if name == "-relu":
            return lambda x: -tf.nn.relu(-x)
        if name == "exp":
            return lambda x: tf.exp(x / self.exp_paramater_mu)
        if name == "-exp":
            return lambda x: -tf.exp(-x / self.exp_paramater_mu)
        if name == "tanh":
            return tf.nn.tanh
        if name == "sigmoid":
            return tf.nn.sigmoid
        raise ValueError(f"Activation function {name} not supported.")

    def instantiate(self, n_shared_features):
        """Instantiate the model.

        Parameters
        ----------
        n_shared_features : int
            Number of shared_features or customer features.
            It is needed to set-up the neural network input shape.
        """
        # TODO: Add possibility for MNL-type weights
        items_features_to_weight_index = {}
        for i, item_param in enumerate(self.items_features_by_choice_parametrization):
            for j, param in enumerate(item_param):
                if isinstance(param, str):
                    items_features_to_weight_index[(i, j)] = len(items_features_to_weight_index)
        self.items_features_to_weight_index = items_features_to_weight_index

        self.taste_params_module = get_feed_forward_net(
            n_shared_features,
            len(items_features_to_weight_index),
            self.taste_net_layers,
            self.taste_net_activation,
        )
        self.instantiated = True

    @property
    def trainable_weights(self):
        """Argument to access the future trainable_weights throught the taste net.

        Returns
        -------
        list
            List of trainable weights.
        """
        if self.instantiated:
            return self.taste_params_module.trainable_variables
        return []

    def compute_batch_utility(
        self,
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices,
    ):
        """Define how the model computes the utility of a product.

        MUST be implemented in children classe !
        For simpler use-cases this is the only method to be user-defined.

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
        np.ndarray
            Utility of each product for each choice.
            Shape must be (n_choices, n_items)
        """
        _ = available_items_by_choice
        # Restacking and dtyping of the item features
        if isinstance(shared_features_by_choice, tuple):
            shared_features_by_choice = tf.concat(
                [
                    tf.cast(shared_feature, tf.float32)
                    for shared_feature in shared_features_by_choice
                ],
                axis=-1,
            )
        if isinstance(items_features_by_choice, tuple):
            items_features_by_choice = tf.concat(
                [tf.cast(items_feature, tf.float32) for items_feature in items_features_by_choice],
                axis=-1,
            )

        taste_weights = self.taste_params_module(shared_features_by_choice)
        item_utility_by_choice = []
        for i, item_param in enumerate(self.items_features_by_choice_parametrization):
            utility = tf.zeros_like(choices, dtype=tf.float32)
            for j, param in enumerate(item_param):
                if isinstance(param, str):
                    weight = taste_weights[:, self.items_features_to_weight_index[(i, j)]]
                    weight = self.get_activation_function(param)(weight)
                    item_feature = items_features_by_choice[:, i, j] * weight

                elif isinstance(param, float):
                    item_feature = param * items_features_by_choice[:, i, j]
                utility += item_feature
            item_utility_by_choice.append(utility)
        return tf.stack(item_utility_by_choice, axis=1)

    def predict_tastes(self, shared_features_by_choice):
        """Predict the tastes of the model for a given dataset.

        Parameters
        ----------
        shared_features_by_choice : np.ndarray
            Shared Features by choice.

        Returns
        -------
        np.ndarray
            Taste of each product for each choice.
            Shape is (n_choices, n_taste_parameters)
        """
        return self.taste_params_module(shared_features_by_choice)

    def fit(self, choice_dataset, **kwargs):
        """Fit to estimate the paramters.

        Parameters
        ----------
        choice_dataset : ChoiceDataset
            Choice dataset to use for the estimation.

        Returns
        -------
        dict
            dict with fit history.
        """
        if not self.instantiated:
            # Lazy Instantiation
            if choice_dataset.get_n_items() != self.n_items:
                raise ValueError(
                    """Number of items in the dataset does not match
                    the number of items of the model."""
                )
            if choice_dataset.get_n_items_features() != self.n_items_features:
                raise ValueError(
                    """Number of items features in the dataset does not match the
                    number of items features in the model."""
                )
            self.instantiate(
                n_shared_features=choice_dataset.get_n_shared_features(),
            )
            self.instantiated = True
        return super().fit(choice_dataset=choice_dataset, **kwargs)

    def _fit_with_lbfgs(self, choice_dataset, sample_weight=None, **kwargs):
        """Specific fit function to estimate the paramters with LBFGS.

        Parameters
        ----------
        choice_dataset : ChoiceDataset
            Choice dataset to use for the estimation.
        n_epochs : int
            Number of epochs to run.
        sample_weight: Iterable, optional
            list of each sample weight, by default None meaning that all samples have weight 1.

        Returns
        -------
        dict
            dict with fit history.
        """
        if not self.instantiated:
            if choice_dataset.get_n_items() != self.n_items:
                raise ValueError(
                    """Number of items in the dataset does not match
                    the number of items of the model."""
                )
            if choice_dataset.get_n_items_features() != self.n_items_features:
                raise ValueError(
                    """Number of items features in the dataset does not match
                    the number of items features in the model."""
                )
            # Lazy Instantiation
            self.instantiate(
                n_shared_features=choice_dataset.get_n_shared_features(),
            )
            self.instantiated = True
        return super()._fit_with_lbfgs(
            dataset=choice_dataset, sample_weight=sample_weight, **kwargs
        )
