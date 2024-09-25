"""Implementation of ResLogit for easy use."""

import logging

import tensorflow as tf

from choice_learn.models.base_model import ChoiceModel


class ResNetLayer(tf.keras.layers.Layer):
    """The ResNet layer class."""

    def __init__(self):
        """Initialize the ResNetLayer class."""
        super().__init__()

    def build(self, input_shape):
        """Create the state of the layer (weights).

        Parameters
        ----------
        input_shape : tuple
            Shape of the input of the layer. Typically (batch_size, num_features).
            Batch_size (None) is ignored, but num_features is the shape of the input.
        """
        print(f"Inside build() function of ResNetLayer: {input_shape=}")
        n_items = input_shape[-1]

        self.residual_weights = self.add_weight(
            shape=(n_items, n_items),
            initializer="random_normal",
            trainable=True,
            name="resnet_weight",
        )

    def call(self, input):
        """Return the output of the ResNet layer.

        Parameters
        ----------
        inputs : tf.Variable
            Input of the residual layer
        """
        lin_output = tf.matmul(input, self.residual_weights)

        return input - tf.math.softplus(
            tf.cast(lin_output, tf.float32)
        )  # Not the same softplus function as in PyTorch???


class ResLogit(ChoiceModel):
    """The ResLogit class."""

    def __init__(
        self,
        intercept="item",  # TODO: check if it still works when intercept is not None
        optimizer="SGD",
        n_layers=16,
        **kwargs,
    ):
        """Initialize the ResLogit class.

        Parameters
        ----------
        add_exit_choice : bool, optional
            Whether or not to normalize the probabilities computation with an exit choice
            whose utility would be 1, by default False
        intercept : str, optional
            ????, by default None
        optimizer: str
            TensorFlow optimizer to be used for estimation
        lr: float
            Learning Rate to be used with optimizer.
        n_layers : int
            Number of residual layers.
        """
        super().__init__(
            self,
            optimizer=optimizer,
            **kwargs,
        )
        self.intercept = intercept
        self.n_layers = n_layers

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
        alphas : tf.Variable
            Alphas parameters (aka intercepts or asc variable) of the model.
        betas : tf.Variable
            Betas parameters of the model.
        resnet : tf.keras.Model
            List of the weights created coresponding to the specification.
        """
        mnl_weights = []
        indexes = {}

        # Create the betas parameters for the shared and items features
        for n_feat, feat_name in zip(
            [n_shared_features, n_items_features],
            ["shared_features", "items_features"],
        ):
            if n_feat > 0:
                mnl_weights += [
                    tf.Variable(
                        tf.random_normal_initializer(0.0, 0.02, seed=42)(shape=(n_feat,)),
                        name=f"Betas_{feat_name}",
                    )
                ]
                indexes[feat_name] = len(mnl_weights) - 1

        # Create the alphas parameters
        if self.intercept is None:
            logging.info("No intercept in the model")
        elif self.intercept == "item":
            mnl_weights.append(
                tf.Variable(
                    tf.random_normal_initializer(0.0, 0.02, seed=42)(shape=(n_items - 1,)),
                    name="Intercept",
                )
            )
            indexes["intercept"] = len(mnl_weights) - 1
        elif self.intercept == "item-full":
            logging.info("Simple MNL intercept is not normalized to 0!")
            mnl_weights.append(
                tf.Variable(
                    tf.random_normal_initializer(0.0, 0.02, seed=42)(shape=(n_items,)),
                    name="Intercept",
                )
            )
            indexes["intercept"] = len(mnl_weights) - 1
        else:
            mnl_weights.append(
                tf.Variable(
                    tf.random_normal_initializer(0.0, 0.02, seed=42)(shape=(1,)),
                    name="Intercept",
                )
            )
            indexes["intercept"] = len(mnl_weights) - 1

        # Create the ResNet layer
        input_shape = (n_items,)
        input = tf.keras.layers.Input(shape=input_shape)
        residual_weights = []
        layers = [ResNetLayer() for _ in range(self.n_layers)]
        output = input
        for layer in layers:
            layer.build(input_shape=(n_items,))  # /!\ Not sure about this line
            residual_weights.append(layer.residual_weights)  # /!\ Not sure about this line
            output = layer(output)
        resnet_model = tf.keras.Model(
            inputs=input, outputs=output, name=f"resnet_with_{self.n_layers}_layers"
        )
        # resnet_model.build(input_shape=(n_items,))

        # Concatenation of all the trainable weights
        weights = mnl_weights + residual_weights

        self.instantiated = True
        self.resnet_model = resnet_model
        self.indexes = indexes
        self.mnl_weights = mnl_weights
        self.residual_weights = residual_weights
        self._trainable_weights = weights
        return indexes, weights

    @property
    def trainable_weights(self):
        """Trainable weights of the model."""
        return self._trainable_weights

    def compute_batch_utility(
        self,
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices,
    ):
        """Compute utility from a batch of ChoiceDataset.

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
        choices : np.ndarray
            Choices
            Shape must be (n_choices, )

        Returns
        -------
        tf.Tensor
            Computed utilities of shape (n_choices, n_items).
        """
        (_, _) = available_items_by_choice, choices  # Avoid unused variable warning

        batch_size = shared_features_by_choice.shape[0]
        n_items = items_features_by_choice.shape[1]

        # Deterministic component of the utility
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
        shared_features_utilities = tf.squeeze(shared_features_utilities)

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
            items_features_utilities = tf.zeros((batch_size, n_items))

        if "intercept" in self.indexes.keys():
            intercept = self.trainable_weights[self.indexes["intercept"]]
            if self.intercept == "item":
                intercept = tf.concat([tf.constant([0.0]), intercept], axis=0)
            if self.intercept in ["item", "item-full"]:
                intercept = tf.expand_dims(intercept, axis=0)
        else:
            intercept = 0

        # /!\ Not sure about the next line shared_features_utilities = tf.tile(...)
        shared_features_utilities = tf.tile(
            tf.expand_dims(shared_features_utilities, axis=-1), [1, n_items]
        )
        deterministic_utilities_without_intercept = tf.add(
            shared_features_utilities, items_features_utilities
        )  # Work with a simple "+" instead of tf.add()???
        deterministic_utilities = tf.add(
            deterministic_utilities_without_intercept, intercept
        )  # Work with a simple "+" instead of tf.add()???

        # Residual component of the utility
        input_shape = (n_items,)
        print(
            "Inside compute_batch_utility function of ResLogit before defining",
            f"'input_data':{input_shape=}",
        )
        input_data = deterministic_utilities_without_intercept
        print(
            "Inside compute_batch_utility function of ResLogit after defining",
            f"'input_data':{input_data.shape=}",
        )

        resnet_model = self.resnet_model
        residual_utilities = resnet_model(input_data)
        residual_utilities = tf.convert_to_tensor(residual_utilities)  # Useless???
        residual_utilities = tf.reshape(
            residual_utilities,
            [batch_size, n_items],
        )  # Useless???
        residual_utilities = tf.cast(residual_utilities, tf.float32)

        return tf.add(
            deterministic_utilities, residual_utilities
        )  # Work with a simple "+" instead of tf.add()???

    def fit(
        self, choice_dataset, get_report=False, **kwargs
    ):  # Not necessary to redefine this method, can be deleted
        """Fit to estimate the parameters.

        Parameters
        ----------
        choice_dataset : ChoiceDataset
            Choice dataset to use for the estimation.
        get_report: bool, optional
            Whether or not to compute a report of the estimation, by default False

        Returns
        -------
        dict
            dict with fit history.
        """
        if not self.instantiated:
            # Lazy Instantiation
            self.indexes, self._trainable_weights = self.instantiate(
                n_items=choice_dataset.get_n_items(),
                n_shared_features=choice_dataset.get_n_shared_features(),
                n_items_features=choice_dataset.get_n_items_features(),
            )
            self.instantiated = True
        fit = super().fit(choice_dataset=choice_dataset, **kwargs)
        if get_report:
            self.report = self.compute_report(choice_dataset)
        return fit
