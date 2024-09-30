"""Implementation of ResLogit for easy use."""

import logging

import tensorflow as tf

import choice_learn.tf_ops as tf_ops
from choice_learn.models.base_model import ChoiceModel


class ResNetLayer(tf.keras.layers.Layer):
    """The ResNet layer class."""

    def __init__(self):
        """Initialize the ResNetLayer class."""
        super().__init__()

    def get_activation_function(self, name):
        """Get an activation function from its str name.

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
        if name == "tanh":
            return tf.nn.tanh
        if name == "sigmoid":
            return tf.nn.sigmoid
        if name == "softplus":
            return tf.math.softplus
        raise ValueError(f"Activation function {name} not supported.")

    def build(self, input_shape, layer_width=None, activation="softplus"):
        """Create the state of the layer (weights).

        Parameters
        ----------
        input_shape : tuple
            Shape of the input of the layer. Typically (batch_size, num_features)
            Batch_size (None) is ignored, but num_features is the shape of the input
        layer_width : int, optional
            Width of the layer, by default None
            If None, the width of the layer is the same as the input shape
        activation : str, optional
            Activation function to use in the layer, by default "softplus"
        """
        self.num_features = input_shape[-1]

        if layer_width is None:
            self.layer_width = input_shape[-1]
        else:
            self.layer_width = layer_width

        self.activation = self.get_activation_function(activation)

        # Random normal initialization of the weights
        # Shape of the weights: (num_features, layer_width)
        self.residual_weights = self.add_weight(
            shape=(self.num_features, self.layer_width),
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

        Returns
        -------
        tf.Variable
            Output of the residual layer
        """
        lin_output = tf.matmul(input, self.residual_weights)

        # Ensure the dimensions are compatible for subtraction
        if input.shape != lin_output.shape:
            # Then perform a linear projection to match the dimensions
            input = tf.matmul(input, tf.ones((self.num_features, self.layer_width)))

        # Softplus: smooth approximation of ReLU
        return input - self.activation(tf.cast(lin_output, tf.float32))

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer.

        Automatically used when calling ResNetLayer.call() to infer the shape of the output.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input of the layer. Typically (batch_size, num_features)
            Batch_size (None) is ignored, but num_features is the shape of the input

        Returns
        -------
        tuple
            Shape of the output of the layer
        """
        return (input_shape[0], self.layer_width)


class ResLogit(ChoiceModel):
    """The ResLogit class."""

    def __init__(
        self,
        intercept="item",
        n_layers=16,
        res_layers_width=None,
        activation="softplus",
        label_smoothing=0.0,
        optimizer="SGD",
        tolerance=1e-8,
        lr=0.001,
        epochs=1000,
        batch_size=32,
        logmin=1e-5,
        **kwargs,
    ):
        """Initialize the ResLogit class.

        Parameters
        ----------
        intercept: str, optional
            Type of intercept to use, by default None
        n_layers : int
            Number of residual layers.
        res_layers_width : list of int, optional
            Width of the *hidden* residual layers, by default None
            If None, all the residual layers have the same width (n_items)
            The length of the list should be equal to n_layers - 1
            The last element of the list should be equal to n_items
        activation : str, optional
            Activation function to use in the residual layers, by default "softplus"
        label_smoothing : float, optional
            Whether (then is ]O, 1[ value) or not (then can be None or 0) to use label smoothing
        optimizer: str
            String representation of the TensorFlow optimizer to be used for estimation,
            by default "SGD"
            Should be within tf.keras.optimizers
        tolerance : float, optional
            Tolerance for the L-BFGS optimizer if applied, by default 1e-8
        lr: float, optional
            Learning rate for the optimizer if applied, by default 0.001
        epochs: int, optional
            (Max) Number of epochs to train the model, by default 1000
        batch_size: int, optional
            Batch size in the case of stochastic gradient descent optimizer
            Not used in the case of L-BFGS optimizer, by default 32
        logmin : float, optional
            Value to be added within log computation to avoid infinity, by default 1e-5
        """
        super().__init__(
            self,
            optimizer=optimizer,
            **kwargs,
        )
        self.intercept = intercept
        self.n_layers = n_layers
        self.res_layers_width = res_layers_width
        self.activation = activation

        # Optimization parameters
        self.label_smoothing = label_smoothing
        self.tolerance = tolerance
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.logmin = logmin

        if optimizer == "Adam":
            self.optimizer = tf.keras.optimizers.Adam(lr)
        elif optimizer == "SGD":
            self.optimizer = tf.keras.optimizers.SGD(lr)
        elif optimizer == "Adamax":
            self.optimizer = tf.keras.optimizers.Adamax(lr)
        else:
            print(f"Optimizer {optimizer} not implemented, switching for default Adam")
            self.optimizer = tf.keras.optimizers.Adam(lr)

        self.instantiated = False

    def instantiate(self, n_items, n_shared_features, n_items_features):
        """Instantiate the model from ModelSpecification object.

        Parameters
        ----------
        n_items : int
            Number of items/aternatives to consider
        n_shared_features : int
            Number of contexts features
        n_items_features : int
            Number of contexts items features

        Returns
        -------
        indexes : dict
            Dictionary of the indexes of the weights created
        weights : list of tf.Variable
            List of the weights created coresponding to the specification
        """
        # Instantiate the loss function
        self.loss = tf_ops.CustomCategoricalCrossEntropy(
            from_logits=False,
            label_smoothing=self.label_smoothing,
            epsilon=self.logmin,
        )

        # Instantiate the weights
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
        if self.res_layers_width is None:
            # Common width for all the residual layers by default: n_items
            # (Like in the original paper of ResLogit)
            for layer in layers:
                layer.build(input_shape=(n_items,), activation=self.activation)
                residual_weights.append(layer.residual_weights)
                output = layer(output)
        else:
            # Different width for each *hidden* residual layer
            if self.n_layers > 0 and len(self.res_layers_width) != self.n_layers - 1:
                raise ValueError(
                    "The length of the res_layers_width list should be equal to n_layers - 1"
                )
            if self.n_layers > 1 and self.res_layers_width[-1] != n_items:
                raise ValueError("The width of the last residual layer should be equal to n_items")
            for i, layer in enumerate(layers):
                if i == 0:
                    # The first layer has the same width as the input
                    layer.build(input_shape=(n_items,), activation=self.activation)
                    residual_weights.append(layer.residual_weights)
                # The other layers have a width defined by the
                # res_layers_width parameter and an input shape
                # depending on the width of the previous layer
                elif i == 1:
                    layer.build(
                        input_shape=(n_items,),
                        layer_width=self.res_layers_width[i - 1],
                        activation=self.activation,
                    )
                    residual_weights.append(layer.residual_weights)
                else:
                    layer.build(
                        input_shape=(self.res_layers_width[i - 2],),
                        layer_width=self.res_layers_width[i - 1],
                        activation=self.activation,
                    )
                    residual_weights.append(layer.residual_weights)
                output = layer(output)
        resnet_model = tf.keras.Model(
            inputs=input, outputs=output, name=f"resnet_with_{self.n_layers}_layers"
        )

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
            Computed utilities of shape (n_choices, n_items)
        """
        (_, _) = available_items_by_choice, choices  # Avoid unused variable warning

        batch_size = shared_features_by_choice.shape[0]  # Other name: n_choices
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

        shared_features_utilities = tf.tile(
            tf.expand_dims(shared_features_utilities, axis=-1), [1, n_items]
        )
        deterministic_utilities_without_intercept = (
            shared_features_utilities + items_features_utilities
        )
        deterministic_utilities = deterministic_utilities_without_intercept + intercept

        # Residual component of the utility
        input_data = deterministic_utilities_without_intercept

        resnet_model = self.resnet_model
        residual_utilities = resnet_model(input_data)
        residual_utilities = tf.cast(residual_utilities, tf.float32)

        return deterministic_utilities + residual_utilities

    def fit(self, choice_dataset, get_report=False, **kwargs):
        """Fit to estimate the parameters.

        Parameters
        ----------
        choice_dataset : ChoiceDataset
            Choice dataset to use for the estimation.
        get_report: bool, optional
            Whether or not to compute a report of the estimation, by default False

        Returns
        -------
        fit : dict
            dict with fit history
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
