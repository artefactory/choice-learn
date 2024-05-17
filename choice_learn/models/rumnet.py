"""Implementation of RUMnet for easy use."""
import tensorflow as tf

import choice_learn.tf_ops as tf_ops
from choice_learn.models.base_model import ChoiceModel


def create_ff_network(
    input_shape, depth, width, activation="elu", add_last=False, l2_regularization_coeff=0.0
):
    """Create a simple fully connected (Dense) network.

    Parameters
    ----------
    input_shape : tuple of int
        shape of the input of the network. Typically (num_features, )
    depth : int
        Number of dense/fully-connected of the network to create.
    width : int
        Neurons number for all dense layers.
    add_last : bool, optional
        Whether to add a Dense layer with a single output at the end, by default False
        Typically to be used when creating the utility network, that outputs a single number:
        the utility.
    l2_regularization_coeff : float, optional
        Regularization coefficient for Dense layers weights during training, by default 0.0

    Returns
    -------
    tf.keras.Model
        Dense Neural Network with tensorflow backend.
    """
    input = tf.keras.layers.Input(shape=input_shape)
    regularizer = tf.keras.regularizers.L2(l2_regularization_coeff)
    out = input
    for _ in range(depth):
        out = tf.keras.layers.Dense(
            width, activation=activation, kernel_regularizer=regularizer, use_bias=True
        )(out)
    if add_last:
        out = tf.keras.layers.Dense(1, activation="linear", use_bias=False)(out)
    return tf.keras.Model(inputs=input, outputs=out)


def recreate_official_nets(
    num_products_features,
    x_width,
    x_depth,
    x_eps,
    num_customer_features,
    z_width,
    z_depth,
    z_eps,
    width_u,
    depth_u,
    l2_regularization_coeff=0.0,
):
    """Create the three nets used in RUMnet: X_net, Z_net and U_net.

    Parameters
    ----------
    num_products_features : int
        Number of features each product will be described with.
        In terms of ChoiceDataset it is the number of { items_features + contexts_items_features }
        for one product.
    num_customer_features : int
        Number of features each customer will be described with.
        In terms of ChoiceDataset it is the number of contexts_features.
    width_eps_x : int
        Number of neurons for each dense layer for the products encoding net.
    depth_eps_x : int
        Number of dense layers for the products encoding net.
    heterogeneity_x : int
        Number of nets of products features encoding.
    width_eps_z : int
        Number of neurons for each dense layer for the customers encoding net.
    depth_eps_z : int
        Number of dense layers for the customers encoding net.
    heterogeneity_z : int
        Number of nets of customers features encoding.
    width_u : int
        Number of neurons for each dense layer for the utility net.
    depth_u : int
        Number of dense layers for the utility net.
    l2_regularization_coef : float, optional
        Value of dense layers weights regulariation to apply during training, by default 0.0

    Returns
    -------
    tf.keras.Model
        Product features encoding network
    tf.keras.Model
        Customer features encoding network
    tf.keras.Model
        Features and encoding to utility computation network
    """
    # Products and Customers embeddings nets, quiet symmetrical
    products_input = tf.keras.layers.Input(shape=(num_products_features))
    customer_input = tf.keras.layers.Input(shape=(num_customer_features))
    x_embeddings = []
    z_embeddings = []

    # Creating independant nets for each heterogeneity
    for _ in range(x_eps):
        x_embedding = create_ff_network(
            input_shape=num_products_features,
            depth=x_depth,
            width=x_width,
            l2_regularization_coeff=l2_regularization_coeff,
        )(products_input)
        x_embeddings.append(x_embedding)

    # Creating independant nets for each heterogeneity
    for _ in range(z_eps):
        z_embedding = create_ff_network(
            input_shape=num_customer_features,
            depth=z_depth,
            width=z_width,
            l2_regularization_coeff=l2_regularization_coeff,
        )(customer_input)

        z_embeddings.append(z_embedding)

    x_net = tf.keras.Model(inputs=products_input, outputs=x_embeddings, name="X_embedding")
    z_net = tf.keras.Model(inputs=customer_input, outputs=z_embeddings, name="Z_embedding")

    # Utility network
    u_net = create_ff_network(
        input_shape=(
            x_width + z_width + num_products_features + num_customer_features
        ),  # Input shape from previous nets
        width=width_u,
        depth=depth_u,
        add_last=True,  # Add last for utility
        l2_regularization_coeff=l2_regularization_coeff,
    )

    return x_net, z_net, u_net


class ParallelDense(tf.keras.layers.Layer):
    """Layer that represents several Dense layers in Parallel.

    Parallel means that they have the same input, but then are not intricated and
    are totally independant from each other.
    """

    def __init__(self, width, depth, heterogeneity, activation="relu", **kwargs):
        """Instantiate the layer.

        Following tf.keras.Layer API. Note that there will be width * depth * heterogeneity
        number of neurons in the layer.

        Parameters
        ----------
        width : int
            Number of neurons for each dense layer.
        depth : int
            Number of neuron layers.
        heterogeneity : int
            Number of dense layers that are in parallel
        activation : str, optional
            activation function at the end of each layer, by default "relu"
        """
        super().__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.heterogeneity = heterogeneity
        self.activation = tf.keras.layers.Activation(activation)

    def build(self, input_shape):
        """Lazy build of the layer.

        Parameters
        ----------
        input_shape : tuple
            shape of the input of the layer. Typically (batch_size, num_features).
            Batch_size (None) is ignored, but num_features is the shape of the input.
        """
        super().build(input_shape)

        weights = [
            (
                self.add_weight(
                    shape=(input_shape[-1], self.width, self.heterogeneity),
                    initializer="glorot_normal",
                    trainable=True,
                ),
                self.add_weight(
                    shape=(self.width, self.heterogeneity),
                    initializer="glorot_normal",
                    trainable=True,
                ),
            )
        ]
        for i in range(self.depth - 1):
            weights.append(
                (
                    self.add_weight(
                        shape=(self.width, self.width, self.heterogeneity),
                        initializer="glorot_normal",
                        trainable=True,
                    ),
                    self.add_weight(
                        shape=(self.width, self.heterogeneity),
                        initializer="glorot_normal",
                        trainable=True,
                    ),
                )
            )

        self.w = weights

    def call(self, inputs):
        """Predict of the layer.

        Follows tf.keras.Layer API.

        Parameters
        ----------
        inputs : tf.Tensor, np.ndarray
            Tensor of shape (batch_size, n_features) as input of the model.

        Returns
        -------
        outputs
            tensor of shape (batch_size, width, heterogeneity)
        """
        outputs = tf.tensordot(inputs, self.w[0][0], axes=1) + self.w[0][1]
        outputs = self.activation(outputs)
        # tf.nn.bias_add(y, weights[0][1], data_format="NC...")

        for w, b in self.w[1:]:
            outputs = tf.einsum("ijk,jlk->ilk", outputs, w) + b
            outputs = self.activation(outputs)

        return outputs


class AssortmentParallelDense(tf.keras.layers.Layer):
    """Several Dense layers in Parallel applied to an Assortment.

    Parallel means that they have the same input, but then are not intricated and
    are totally independant from each other. The layer applies the same Dense layers
    to an assortment of items.
    """

    def __init__(self, width, depth, heterogeneity, activation="relu", **kwargs):
        """Inialization of the layer.

        Parameters
        ----------
        width : int
            Number of neurons of each dense layer.
        depth : int
            Number of dense layers
        heterogeneity : int
            Number of dense networks in parallel.
        activation : str, optional
            activation function of each dense, by default "relu"
        """
        super().__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.heterogeneity = heterogeneity
        self.activation = tf.keras.layers.Activation(activation)

    def build(self, input_shape):
        """Lazy build of the layer.

        Follows tf.keras API.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input of the layer.
            Typically (batch_size, num_items, num_features).
        """
        super().build(input_shape)

        weights = [
            (
                self.add_weight(
                    shape=(input_shape[-1], self.width, self.heterogeneity),
                    initializer="glorot_normal",
                    trainable=True,
                ),
                self.add_weight(
                    shape=(self.width, self.heterogeneity),
                    initializer="glorot_normal",
                    trainable=True,
                ),
            )
        ]
        for i in range(self.depth - 1):
            weights.append(
                (
                    self.add_weight(
                        shape=(self.width, self.width, self.heterogeneity),
                        initializer="glorot_normal",
                        trainable=True,
                    ),
                    self.add_weight(
                        shape=(self.width, self.heterogeneity),
                        initializer="glorot_normal",
                        trainable=True,
                    ),
                )
            )

        self.w = weights

    def call(self, inputs):
        """Predict of the layer.

        Follows tf.keras.Layer API.

        Parameters
        ----------
        inputs : tf.Tensor, np.ndarray
            Tensor of shape (batch_size, n_items, n_features) as input of the model.

        Returns
        -------
        tf.Tensor
            Embeddings of shape (batch_size, n_items, width, heterogeneity)
        """
        outputs = tf.tensordot(inputs, self.w[0][0], axes=[[2], [0]]) + self.w[0][1]
        outputs = self.activation(outputs)

        for w, b in self.w[1:]:
            outputs = tf.einsum("imjk,jlk->imlk", outputs, w) + b
            outputs = self.activation(outputs)

        return outputs


class AssortmentUtilityDenseNetwork(tf.keras.layers.Layer):
    """Dense Network that is applied to an assortment of items.

    We apply to the same network over several items and several heterogeneitites.
    """

    def __init__(self, width, depth, activation="relu", add_last=True, **kwargs):
        """Initialize the layer.

        Parameters
        ----------
        width : int
            Nnumber of neurons of each dense layer.
        depth : int
            Number of dense layers.
        activation : str, optional
            Activation function for each layer, by default "relu"
        add_last : bool, optional
            Whether to add a final dense layer with 1 neuron, by default True
        """
        super().__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.activation = tf.keras.layers.Activation(activation)
        self.add_last = add_last

    def build(self, input_shape):
        """Lazy build of the layer.

        Follows tf.keras.Layer API.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input of the layer.
            Typically (batch_size, num_items, width, heterogeneity).
        """
        super().build(input_shape)

        weights = [
            (
                self.add_weight(
                    shape=(input_shape[-2], self.width),
                    initializer="glorot_normal",
                    trainable=True,
                ),
                self.add_weight(
                    shape=(self.width, 1),
                    initializer="glorot_normal",
                    trainable=True,
                ),
            )
        ]
        for i in range(self.depth - 1):
            weights.append(
                (
                    self.add_weight(
                        shape=(self.width, self.width),
                        initializer="glorot_normal",
                        trainable=True,
                    ),
                    self.add_weight(
                        shape=(self.width, 1),
                        initializer="glorot_normal",
                        trainable=True,
                    ),
                )
            )
        if self.add_last:
            self.last = self.add_weight(
                shape=(self.width, 1), initializer="glorot_normal", trainable=True
            )

        self.w = weights

    def call(self, inputs):
        """Predict of the layer.

        Parameters
        ----------
        inputs : tf.Tensor, np.ndarray
            Input Tensor of shape (batch_size, num_items, width, heterogeneity)

        Returns
        -------
        tf.Tensor
            Utilities of shape (batch_size, num_items, heterogeneity)
        """
        outputs = inputs

        for w, b in self.w:
            # bs, items, features, heterogeneities
            outputs = tf.einsum("ijlk, lm->ijmk", outputs, w) + b
            outputs = self.activation(outputs)

        if self.add_last:
            outputs = tf.einsum("ijlk, lm->ijmk", outputs, self.last)

        return outputs


class PaperRUMnet(ChoiceModel):
    """Re-Implementation of the RUMnet model.

    Re-implemented from the paper:
    Representing Random Utility Choice Models with Neural Networks from Ali Aouad and Antoine Désir.
    https://arxiv.org/abs/2207.12877

    --- Attention: ---
    Note that the model uses two type of features that are treated differently:
        - customer features
        - product features
    >>> In this implementation, please make sure that the features are correctly formatted:
        - customer features: (n_choices, n_features) are given as 'shared_features_by_choice'
        in the ChoiceDataset used to fit the model
        - product features: (n_choices, n_items, n_features) are given as
        'items_features_by_choice' in the ChoiceDataset used to fit the model
    ---
    Inherits from base_model.ChoiceModel
    """

    def __init__(
        self,
        num_products_features,
        num_customer_features,
        width_eps_x,
        depth_eps_x,
        heterogeneity_x,
        width_eps_z,
        depth_eps_z,
        heterogeneity_z,
        width_u,
        depth_u,
        tol,
        optimizer,
        lr,
        add_exit_choice=False,
        logmin=1e-5,
        l2_regularization_coef=0.0,
        label_smoothing=0.0,
        **kwargs,
    ):
        """Initialize the RUMnet Model.

        Parameters
        ----------
        num_products_features : int
            Number of features each product will be described with.
            In terms of ChoiceDataset it is the number of
            { items_features + contexts_items_features } for one product.
        num_customer_features : int
            Number of features each customer will be described with.
            In terms of ChoiceDataset it is the number of contexts_features.
        width_eps_x : int
            Number of neurons for each dense layer for the products encoding net.
        depth_eps_x : int
            Number of dense layers for the products encoding net.
        heterogeneity_x : int
            Number of nets of products features encoding.
        width_eps_z : int
            Number of neurons for each dense layer for the customers encoding net.
        depth_eps_z : int
            Number of dense layers for the customers encoding net.
        heterogeneity_z : int
            Number of nets of customers features encoding.
        width_u : int
            Number of neurons for each dense layer for the utility net.
        depth_u : int
            Number of dense layers for the utility net.
        tol : float
            # To be Implemented
        optimizer : str
            String representation of the optimizer to use. By default is Adam if not specified.
            Should be within tf.keras.optimizers.
        lr : float
            Starting learning rate to associate with optimizer.
        add_exit_choice : bool, optional
            Whether or not to add exit option with utility 1, by default True
        logmin : float, optional
            Value to be added within log computation to avoid infinity, by default 1e-5
        l2_regularization_coef : float, optional
            Value of dense layers weights regulariation to apply during training, by default 0.0
        label_smoothing : float, optional
            Value of smoothing to apply in CrossEntropy loss computation, by default 0.0
        """
        super().__init__(add_exit_choice=add_exit_choice, optimizer=optimizer, **kwargs)
        # Number of features
        if num_customer_features <= 0:
            raise ValueError("Number of customer features must be at least 1.")
        if num_products_features <= 0:
            raise ValueError("Number of product features must be at least 1.")
        self.num_products_features = num_products_features
        self.num_customer_features = num_customer_features

        # Dimension of encoding nets
        self.width_eps_x = width_eps_x
        self.depth_eps_x = depth_eps_x
        self.heterogeneity_x = heterogeneity_x

        self.width_eps_z = width_eps_z
        self.depth_eps_z = depth_eps_z
        self.heterogeneity_z = heterogeneity_z

        # Dimension of utility net
        self.width_u = width_u
        self.depth_u = depth_u

        # Optimization parameters
        self.logmin = logmin
        self.tol = tol
        self.lr = lr
        self.add_exit_choice = add_exit_choice
        self.l2_regularization_coef = l2_regularization_coef
        self.label_smoothing = label_smoothing

        if optimizer == "Adam":
            self.optimizer = tf.keras.optimizers.Adam(lr)
        elif optimizer == "SGD":
            self.optimizer = tf.keras.optimizers.SGD(lr)
        elif optimizer == "Adamax":
            self.optimizer = tf.keras.optimizers.Adamax(lr)
        else:
            print(f"Optimizer {optimizer} not implemnted, switching for default Adam")
            self.optimizer = tf.keras.optimizers.Adam(lr)

        self.instantiated = False

    def instantiate(self):
        """Instatiation of the RUMnet model.

        Creation of :
            - x_model encoding products features,
            - z_model encoding customers features,
            - u_model computing utilities from product, customer features and their embeddings
        """
        # Instatiation of the different nets
        self.x_model, self.z_model, self.u_model = recreate_official_nets(
            num_products_features=self.num_products_features,
            num_customer_features=self.num_customer_features,
            x_width=self.width_eps_x,
            x_depth=self.depth_eps_x,
            x_eps=self.heterogeneity_x,
            z_width=self.width_eps_z,
            z_depth=self.depth_eps_z,
            z_eps=self.heterogeneity_z,
            width_u=self.width_u,
            depth_u=self.depth_u,
            l2_regularization_coeff=self.l2_regularization_coef,
        )

        # Storing weights for back-propagation
        self.trainable_weights = self.x_model.weights + self.z_model.weights + self.u_model.weights
        self.loss = tf_ops.CustomCategoricalCrossEntropy(
            from_logits=False,
            label_smoothing=self.label_smoothing,
            epsilon=self.logmin,
        )
        self.instantiated = True

    def compute_batch_utility(
        self,
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices,
    ):
        """Compute utility from a batch of ChoiceDataset.

        Here we asssume that: item features = {fixed item features + contexts item features}
                              user features = {contexts features}

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
        (_, _) = available_items_by_choice, choices
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

        # Computation of utilities
        utilities = []

        # Computation of the customer features embeddings
        z_embeddings = self.z_model(shared_features_by_choice)

        # Iterate over items in assortment
        for item_i in range(items_features_by_choice.shape[1]):
            # Computation of item features embeddings
            x_embeddings = self.x_model(items_features_by_choice[:, item_i, :])

            utilities.append([])

            # Computation of utilites from embeddings, iteration over heterogeneities
            # (eps_x * eps_z)
            for _x in x_embeddings:
                for _z in z_embeddings:
                    _u = tf.keras.layers.Concatenate()(
                        [items_features_by_choice[:, item_i, :], _x, shared_features_by_choice, _z]
                    )
                    utilities[-1].append(self.u_model(_u))

        # Reshape utilities: (batch_size, num_items, heterogeneity)
        return tf.transpose(tf.squeeze(tf.stack(utilities, axis=0), -1))

    @tf.function
    def train_step(
        self,
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices,
        sample_weight=None,
    ):
        """Update model's weight with a step of gradient descent.

        Function that represents one training step (= one gradient descent step) of the model.
        Handles a batch of data of size n_contexts = n_choices = batch_size

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
        sample_weight : np.ndarray, optional
            List samples weights to apply during the gradient descent to the batch elements,
            by default None

        Returns
        -------
        tf.Tensor
            Value of NegativeLogLikelihood loss for the batch
        """
        with tf.GradientTape() as tape:
            # Computation of utilities
            all_u = self.compute_batch_utility(
                shared_features_by_choice=shared_features_by_choice,
                items_features_by_choice=items_features_by_choice,
                available_items_by_choice=available_items_by_choice,
                choices=choices,
            )
            probabilities = []

            # Iterate over heterogeneities
            eps_probabilities = tf.nn.softmax(all_u, axis=1)

            # Average probabilities over heterogeneities
            probabilities = tf.reduce_mean(eps_probabilities, axis=-1)

            # It is not in the paper, but let's normalize with availabilities
            probabilities = tf.multiply(probabilities, available_items_by_choice)
            probabilities = tf.divide(
                probabilities, tf.reduce_sum(probabilities, axis=1, keepdims=True) + 1e-5
            )
            if self.tol > 0:
                probabilities = (1 - self.tol) * probabilities + self.tol * tf.ones_like(
                    probabilities
                ) / probabilities.shape[-1]

            # Probabilities of selected products

            # Negative Log-Likelihood
            batch_nll = self.loss(
                y_pred=probabilities,
                y_true=tf.one_hot(choices, depth=probabilities.shape[1]),
                sample_weight=sample_weight,
            )

            if self.regularization_type is not None:
                regularization = tf.reduce_sum(
                    [self.regularizer(w) for w in self.trainable_weights]
                )
                batch_nll += regularization

        grads = tape.gradient(batch_nll, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return batch_nll

    @tf.function
    def batch_predict(
        self,
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices,
        sample_weight=None,
    ):
        """Represent one prediction (Probas + Loss) for one batch of a ChoiceDataset.

        Specifically recoded for RUMnet because it is needed to average probabilities over
        heterogeneities.

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
        sample_weight : np.ndarray, optional
            List samples weights to apply during the gradient descent to the batch elements,
            by default None

        Returns
        -------
        tf.Tensor (1, )
            Value of NegativeLogLikelihood loss for the batch
        tf.Tensor (batch_size, n_items)
            Probabilities for each product to be chosen for each contexts
        """
        utilities = self.compute_batch_utility(
            shared_features_by_choice=shared_features_by_choice,
            items_features_by_choice=items_features_by_choice,
            available_items_by_choice=available_items_by_choice,
            choices=choices,
        )
        probabilities = tf.nn.softmax(utilities, axis=1)
        probabilities = tf.reduce_mean(probabilities, axis=-1)

        # Normalization with availabilties
        probabilities = tf.multiply(probabilities, available_items_by_choice)
        probabilities = tf.divide(
            probabilities, tf.reduce_sum(probabilities, axis=1, keepdims=True) + 1e-5
        )

        batch_loss = {
            "optimized_loss": self.loss(
                y_pred=probabilities,
                y_true=tf.one_hot(choices, depth=probabilities.shape[1]),
                sample_weight=sample_weight,
            ),
            "NegativeLogLikelihood": tf.keras.losses.CategoricalCrossentropy()(
                y_pred=probabilities,
                y_true=tf.one_hot(choices, depth=probabilities.shape[1]),
                sample_weight=sample_weight,
            ),
        }
        return batch_loss, probabilities


class CPURUMnet(PaperRUMnet):
    """CPU-optimized Re-Implementation of the RUMnet model.

    Re-implemented from the paper:
    Representing Random Utility Choice Models with Neural Networks from Ali Aouad and Antoine Désir.
    https://arxiv.org/abs/2207.12877

    --- Attention: ---
    Note that the model uses two type of features that are treated differently:
        - customer features
        - product features
    >>> In this implementation, please make sure that the features are correctly formatted:
        - customer features: (n_choices, n_features) are given as 'shared_features_by_choice'
        in the ChoiceDataset used to fit the model
        - product features: (n_choices, n_items, n_features) are given as
        'items_features_by_choice' in the ChoiceDataset used to fit the model
    """

    def compute_batch_utility(
        self,
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices,
    ):
        """Compute utility from a batch of ChoiceDataset.

        Here we asssume that: item features = {fixed item features + contexts item features}
                                user features = {contexts features}

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
            Utility of each product for each contexts.
            Shape must be (n_choices, n_items)
        """
        (_, _) = available_items_by_choice, choices
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

        # Computation of utilities
        utilities = []
        batch_size = shared_features_by_choice.shape[0]

        # Computation of the customer features embeddings
        z_embeddings = self.z_model(shared_features_by_choice)

        # Iterate over items in assortment
        for item_i in range(items_features_by_choice.shape[1]):
            # Computation of item features embeddings
            x_embeddings = self.x_model(items_features_by_choice[:, item_i, :])

            stacked_heterogeneities = []
            # Computation of utilites from embeddings, iteration over heterogeneities
            # eps_x * eps_z
            for _x in x_embeddings:
                for _z in z_embeddings:
                    full_embedding = tf.keras.layers.Concatenate()(
                        [items_features_by_choice[:, item_i, :], _x, shared_features_by_choice, _z]
                    )
                    stacked_heterogeneities.append(full_embedding)
            item_utilities = self.u_model(tf.concat(stacked_heterogeneities, axis=0))
            item_utilities = tf.stack(
                [
                    item_utilities[batch_size * i : batch_size * (i + 1)]
                    for i in range(len(x_embeddings) * len(z_embeddings))
                ],
                axis=1,
            )
            utilities.append(item_utilities)
        # Reshape utilities: (batch_size, num_items, heterogeneity)
        return tf.squeeze(tf.stack(utilities, axis=1), -1)


class GPURUMnet(PaperRUMnet):
    """GPU-optimized Re-Implementation of the RUMnet model.

    Re-implemented from the paper:
    Representing Random Utility Choice Models with Neural Networks from Ali Aouad and Antoine Désir.
    https://arxiv.org/abs/2207.12877

    --- Attention: ---
    Note that the model uses two type of features that are treated differently:
        - customer features
        - product features
    >>> In this implementation, please make sure that the features are correctly formatted:
        - customer features: (n_choices, n_features) are given as 'shared_features_by_choice'
        in the ChoiceDataset used to fit the model
        - product features: (n_choices, n_items, n_features) are given as
        'items_features_by_choice' in the ChoiceDataset used to fit the model
    """

    def instantiate(self):
        """Instatiation of the RUMnet model.

        Instantiation of the three nets:
            - x_model encoding products features,
            - z_model encoding customers features,
            - u_model computing utilities from product, customer features and their embeddings
        """
        # Instatiation of the different nets
        self.x_model = AssortmentParallelDense(
            width=self.width_eps_x, depth=self.depth_eps_x, heterogeneity=self.heterogeneity_x
        )
        self.z_model = ParallelDense(
            width=self.width_eps_z, depth=self.depth_eps_z, heterogeneity=self.heterogeneity_z
        )
        self.u_model = AssortmentUtilityDenseNetwork(
            width=self.width_u, depth=self.depth_u, add_last=True
        )

        # Storing weights for back-propagation
        self.trainable_weights = (
            self.x_model.trainable_variables
            + self.z_model.trainable_variables
            + self.u_model.trainable_variables
        )
        self.loss = tf_ops.CustomCategoricalCrossEntropy(
            from_logits=False, label_smoothing=self.label_smoothing
        )
        self.time_dict = {}
        self.instantiated = True

    def compute_batch_utility(
        self,
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices,
    ):
        """Compute utility from a batch of ChoiceDataset.

        Here we asssume that: item features = {fixed item features + contexts item features}
                                user features = {contexts features}

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
            Utility of each product for each contexts.
            Shape must be (n_choices, n_items)
        """
        (_, _) = available_items_by_choice, choices

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

        item_utility_by_choice = []

        # Computation of the customer features embeddings
        z_embeddings = self.z_model(shared_features_by_choice)
        x_embeddings = self.x_model(items_features_by_choice)
        # Reshaping
        big_z = tf.tile(
            tf.expand_dims(shared_features_by_choice, axis=2),
            multiples=[1, 1, self.heterogeneity_z],
        )
        big_z = tf.repeat(
            tf.concat([big_z, z_embeddings], axis=1), repeats=self.heterogeneity_x, axis=2
        )

        # Iterate over items in assortment
        for item_i in range(items_features_by_choice.shape[1]):
            # Computation of item features embeddings
            # utilities.append([])

            # Computation of utilites from embeddings, iteration over heterogeneities
            # (eps_x * eps_z)
            x_fixed_features = tf.repeat(
                tf.expand_dims(items_features_by_choice[:, item_i, :], axis=2),
                repeats=self.heterogeneity_x * self.heterogeneity_z,
                axis=2,
            )
            big_x = tf.repeat(x_embeddings[:, item_i], repeats=self.heterogeneity_z, axis=2)

            item_utility_by_choice.append(tf.concat([big_z, x_fixed_features, big_x], axis=1))

        # Computing resulting utilitiies
        utilities = self.u_model(tf.stack(item_utility_by_choice, axis=1))
        utilities = tf.squeeze(utilities, -2)

        # Reshape & return
        return tf.transpose(utilities, perm=[0, 2, 1])

    @tf.function
    def train_step(
        self,
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices,
        sample_weight=None,
    ):
        """Represent one training step (= one gradient descent step) of the model.

        Recoded because heterogeneities generate different shapes of tensors.
        # TODO: verify that it is indeed different than PaperRUMnet

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
        sample_weight : np.ndarray, optional
            List samples weights to apply during the gradient descent to the batch elements,
            by default None

        Returns
        -------
        tf.Tensor
            Value of NegativeLogLikelihood loss for the batch
        """
        with tf.GradientTape() as tape:
            # Computation of utilities
            utilities = self.compute_batch_utility(
                shared_features_by_choice=shared_features_by_choice,
                items_features_by_choice=items_features_by_choice,
                available_items_by_choice=available_items_by_choice,
                choices=choices,
            )
            eps_probabilities = tf.nn.softmax(utilities, axis=2)
            # Average probabilities over heterogeneities
            probabilities = tf.reduce_mean(eps_probabilities, axis=1)

            # Availability normalization
            probabilities = tf.multiply(probabilities, available_items_by_choice)
            probabilities = tf.divide(
                probabilities, tf.reduce_sum(probabilities, axis=1, keepdims=True) + 1e-5
            )
            if self.tol > 0:
                probabilities = (1 - self.tol) * probabilities + self.tol * tf.ones_like(
                    probabilities
                ) / probabilities.shape[-1]

            # Negative Log-Likelihood
            batch_nll = self.loss(
                y_pred=probabilities,
                y_true=tf.one_hot(choices, depth=probabilities.shape[1]),
                sample_weight=sample_weight,
            )

        grads = tape.gradient(
            batch_nll,
            self.x_model.trainable_variables
            + self.z_model.trainable_variables
            + self.u_model.trainable_variables,
        )
        self.optimizer.apply_gradients(
            zip(
                grads,
                self.x_model.trainable_variables
                + self.z_model.trainable_variables
                + self.u_model.trainable_variables,
            )
        )
        return batch_nll

    @tf.function
    def batch_predict(
        self,
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices,
        sample_weight=None,
    ):
        """RUMnet batch_predict.

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
        sample_weight : np.ndarray, optional
            List samples weights to apply during the gradient descent to the batch elements,
            by default None

        Returns
        -------
        tf.Tensor (1, )
            Value of NegativeLogLikelihood loss for the batch
        tf.Tensor (batch_size, n_items)
            Probabilities for each product to be chosen for each contexts
        """
        utilities = self.compute_batch_utility(
            shared_features_by_choice=shared_features_by_choice,
            items_features_by_choice=items_features_by_choice,
            available_items_by_choice=available_items_by_choice,
            choices=choices,
        )
        probabilities = tf.nn.softmax(utilities, axis=2)
        probabilities = tf.reduce_mean(probabilities, axis=1)

        # Test with availability normalization
        probabilities = tf.multiply(probabilities, available_items_by_choice)
        probabilities = tf.divide(
            probabilities, tf.reduce_sum(probabilities, axis=1, keepdims=True) + 1e-5
        )

        batch_loss = {
            "optimized_loss": self.loss(
                y_pred=probabilities,
                y_true=tf.one_hot(choices, depth=probabilities.shape[1]),
                sample_weight=sample_weight,
            ),
            "NegativeLogLikelihood": tf.keras.losses.CategoricalCrossentropy()(
                y_pred=probabilities,
                y_true=tf.one_hot(choices, depth=probabilities.shape[1]),
                sample_weight=sample_weight,
            ),
        }
        return batch_loss, probabilities
