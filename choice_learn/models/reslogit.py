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

    def build(self, input_shape):
        """Create the state of the layer (weights).

        Parameters
        ----------
        input_shape : tuple
            Shape of the input of the layer. Typically (batch_size, num_features).
            Batch_size (None) is ignored, but num_features is the shape of the input.
        """
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
        n_layers=16,
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
        intercept : str, optional
            ????, by default None
        n_layers : int
            Number of residual layers.
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
        input_data = deterministic_utilities_without_intercept

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
