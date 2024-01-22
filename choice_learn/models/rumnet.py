"""Implementation of RUMnet for easy use."""
import tensorflow as tf

from choice_learn.models.base_model import ChoiceModel
from choice_learn.tf_ops import CustomCategoricalCrossEntropy


class PaperRUMnet(ChoiceModel):
    """Re-Implementation of the RUMnet model.

    Re-implemented from the paper:
    Representing Random Utility Choice Models with Neural Networks from Ali Aouad and Antoine Désir
    https://arxiv.org/abs/2207.12877

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
        normalize_non_buy=True,
        logmin=1e-5,
        l2_regularization_coef=0.0,
        label_smoothing=0.0,
        **kwargs,
    ):
        """Initiation of the RUMnet Model.

        Parameters
        ----------
        num_products_features : int
            Number of features each product will be described with.
            In terms of ChoiceDataset it is the number of
            { items_features + sessions_items_features } for one product.
        num_customer_features : int
            Number of features each customer will be described with.
            In terms of ChoiceDataset it is the number of sessions_features.
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
        normalize_non_buy : bool, optional
            Whether or not to add exit option with utility 1, by default True
        logmin : float, optional
            Value to be added within log computation to avoid infinity, by default 1e-5
        l2_regularization_coef : float, optional
            Value of dense layers weights regulariation to apply during training, by default 0.0
        label_smoothing : float, optional
            Value of smoothing to apply in CrossEntropy loss computation, by default 0.0
        """
        super().__init__(normalize_non_buy=normalize_non_buy, **kwargs)
        # Number of features
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
        self.normalize_non_buy = normalize_non_buy
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
        self.weights = self.x_model.weights + self.z_model.weights + self.u_model.weights
        self.loss = CustomCategoricalCrossEntropy(
            from_logits=False, label_smoothing=self.label_smoothing
        )

    def compute_batch_utility(
        self,
        fixed_items_features,
        contexts_features,
        contexts_items_features,
        contexts_items_availabilities,
        choices,
    ):
        """Compute utility from a batch of ChoiceDataset.

        Here we asssume that: item features = {fixed item features + session item features}
                              user features = {session features}

        Parameters
        ----------
        fixed_items_features : tuple of np.ndarray (n_items, n_features)
            Items-Features: formatting from ChoiceDataset: a matrix representing the
            products fixed features.
        contexts_features : tuple of np.ndarray (n_contexts, n_features)
            Contexts-Features: features varying with contexts, shared by all products
        contexts_items_features :tuple of np.ndarray (n_contexts, n_items, n_features)
            Features varying with contexts and products
        contexts_items_availabilities : np.ndarray (n_contexts, n_items)
            Availabilities: here for ChoiceModel signature
        choices :  np.ndarray (n_contexts, )
            Choices: here for ChoiceModel signature

        Returns:
        --------
        np.ndarray
            Utility of each product for each session.
            Shape must be (n_sessions, n_items)
        """
        (_, _) = contexts_items_availabilities, choices
        ### Restacking of the item features
        items_features_batch = tf.concat([*fixed_items_features], axis=-1)
        session_features_batch = tf.concat([*contexts_features], axis=-1)
        session_items_features_batch = tf.concat([*contexts_items_features], axis=-1)

        full_item_features = tf.stack(
            [items_features_batch] * session_items_features_batch.shape[0], axis=0
        )
        full_item_features = tf.concat([session_items_features_batch, full_item_features], axis=-1)

        ### Computation of utilities
        utilities = []

        # Computation of the customer features embeddings
        z_embeddings = self.z_model(session_features_batch)

        # Iterate over items in assortment
        for item_i in range(full_item_features.shape[1]):
            # Computation of item features embeddings
            x_embeddings = self.x_model(full_item_features[:, item_i, :])

            utilities.append([])

            # Computation of utilites from embeddings, iteration over heterogeneities
            # (eps_x * eps_z)
            for _x in x_embeddings:
                for _z in z_embeddings:
                    _u = tf.keras.layers.Concatenate()(
                        [full_item_features[:, item_i, :], _x, session_features_batch, _z]
                    )
                    utilities[-1].append(self.u_model(_u))

        ### Reshape utilities: (batch_size, num_items, heterogeneity)
        return tf.transpose(tf.squeeze(tf.stack(utilities, axis=0), -1))

    @tf.function
    def train_step(
        self,
        fixed_items_features,
        contexts_features,
        contexts_items_features,
        contexts_items_availabilities,
        choices,
        sample_weight=None,
    ):
        """Modified version of train step, as we have to average probabilities over heterogeneities.

        Function that represents one training step (= one gradient descent step) of the model.
        Handles a batch of data of size n_contexts = n_choices = batch_size

        Parameters
        ----------
        fixed_items_features : tuple of np.ndarray (n_items, n_features)
            Items-Features: formatting from ChoiceDataset: a matrix representing the
            products fixed features.
        contexts_features : tuple of np.ndarray (n_contexts, n_features)
            Contexts-Features: features varying with contexts, shared by all products
        contexts_items_features :tuple of np.ndarray (n_contexts, n_items, n_features)
            Features varying with contexts and products
        contexts_items_availabilities : np.ndarray (n_contexts, n_items)
            Availabilities of items
        choices :  np.ndarray (n_contexts, )
            Choices
        sample_weight : np.ndarray, optional
            List samples weights to apply during the gradient descent to the batch elements,
            by default None

        Returns:
        --------
        tf.Tensor
            Value of NegativeLogLikelihood loss for the batch
        """
        with tf.GradientTape() as tape:
            ### Computation of utilities
            all_u = self.compute_batch_utility(
                fixed_items_features=fixed_items_features,
                contexts_features=contexts_features,
                contexts_items_features=contexts_items_features,
                contexts_items_availabilities=contexts_items_availabilities,
                choices=choices,
            )
            probabilities = []

            # Iterate over heterogeneities
            # for i in range(all_u.shape[2]):
            # Assortment(t) Utility
            # eps_probabilities = availability_softmax(all_u[:, :, i], ia_batch, axis=2)
            eps_probabilities = tf.nn.softmax(all_u, axis=2)
            # probabilities.append(eps_probabilities)

            # Average probabilities over heterogeneities
            probabilities = tf.reduce_mean(eps_probabilities, axis=1)

            # It is not in the paper, but let's normalize with availabilities
            probabilities = tf.multiply(probabilities, contexts_items_availabilities)
            probabilities = tf.divide(
                probabilities, tf.reduce_sum(probabilities, axis=1, keepdims=True) + 1e-5
            )

            # Probabilities of selected products
            # chosen_probabilities = tf.gather_nd(indices=choices_nd, params=probabilities)

            # Negative Log-Likelihood
            batch_nll = self.loss(
                y_pred=probabilities,
                y_true=tf.one_hot(choices, depth=probabilities.shape[1]),
                sample_weight=sample_weight,
            )
            # nll = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(
            #     y_pred=probabilities, y_true=c_batch
            # )
            # nll = -tf.reduce_sum(tf.math.log(chosen_probabilities + self.logmin))

        grads = tape.gradient(batch_nll, self.weights)
        self.optimizer.apply_gradients(zip(grads, self.weights))
        return batch_nll

    @tf.function
    def batch_predict(
        self,
        fixed_items_features,
        contexts_features,
        contexts_items_features,
        contexts_items_availabilities,
        choices,
        sample_weight=None,
    ):
        """Function that represents one prediction (Probas + Loss) for one batch of a ChoiceDataset.

        Specific version for RUMnet because it is needed to average probabilities over
        heterogeneities.

        Parameters
        ----------
        fixed_items_features : tuple of np.ndarray (n_items, n_features)
            Items-Features: formatting from ChoiceDataset: a matrix representing the
            products fixed features.
        contexts_features : tuple of np.ndarray (n_contexts, n_features)
            Contexts-Features: features varying with contexts, shared by all products
        contexts_items_features :tuple of np.ndarray (n_contexts, n_items, n_features)
            Features varying with contexts and products
        contexts_items_availabilities : np.ndarray (n_contexts, n_items)
            Availabilities of items
        choices :  np.ndarray (n_contexts, )
            Choices
        sample_weight : np.ndarray, optional
            List samples weights to apply during the gradient descent to the batch elements,
            by default None

        Returns:
        --------
        tf.Tensor (1, )
            Value of NegativeLogLikelihood loss for the batch
        tf.Tensor (batch_size, n_items)
            Probabilities for each product to be chosen for each session
        """
        utilities = self.compute_utility(
            fixed_items_features=fixed_items_features,
            contexts_features=contexts_features,
            contexts_items_features=contexts_items_features,
            contexts_items_availabilities=contexts_items_availabilities,
            choices=choices,
        )
        probabilities = tf.nn.softmax(utilities, axis=2)
        probabilities = tf.reduce_mean(probabilities, axis=1)

        # Normalization with availabilties
        probabilities = tf.multiply(probabilities, contexts_items_availabilities)
        probabilities = tf.divide(
            probabilities, tf.reduce_sum(probabilities, axis=1, keepdims=True) + 1e-5
        )
        batch_nll = self.loss(
            y_pred=probabilities,
            y_true=tf.one_hot(choices, depth=probabilities.shape[1]),
            sample_weight=sample_weight,
        )
        return batch_nll, probabilities


class CPURUMnet(PaperRUMnet):
    """CPU-optimized Re-Implementation of the RUMnet model.

    This implementation handles in parallel the heterogenneities so that the training is faster.
    """

    def compute_batch_utility(
        self,
        fixed_items_features,
        contexts_features,
        contexts_items_features,
        contexts_items_availabilities,
        choices,
    ):
        """Compute utility from a batch of ChoiceDataset.

        Here we asssume that: item features = {fixed item features + session item features}
                              user features = {session features}

        Parameters
        ----------
        fixed_items_features : tuple of np.ndarray (n_items, n_features)
            Items-Features: formatting from ChoiceDataset: a matrix representing the
            products fixed features.
        contexts_features : tuple of np.ndarray (n_contexts, n_features)
            Contexts-Features: features varying with contexts, shared by all products
        contexts_items_features :tuple of np.ndarray (n_contexts, n_items, n_features)
            Features varying with contexts and products
        contexts_items_availabilities : np.ndarray (n_contexts, n_items)
            Availabilities of items
        choices :  np.ndarray (n_contexts, )
            Choices

        Returns:
        --------
        np.ndarray
            Utility of each product for each session.
            Shape must be (n_sessions, n_items)
        """
        (_, _) = contexts_items_availabilities, choices
        ### Restacking of the item features
        stacked_fixed_items_features = tf.concat([*fixed_items_features], axis=-1)
        stacked_contexts_features = tf.concat([*contexts_features], axis=-1)
        stacked_contexts_items_features = tf.concat([*contexts_items_features], axis=-1)

        full_item_features = tf.stack(
            [stacked_fixed_items_features] * stacked_contexts_items_features.shape[0], axis=0
        )
        full_item_features = tf.concat(
            [stacked_contexts_items_features, full_item_features], axis=-1
        )

        ### Computation of utilities
        utilities = []
        batch_size = stacked_contexts_features.shape[0]

        # Computation of the customer features embeddings
        z_embeddings = self.z_model(stacked_contexts_features)

        # Iterate over items in assortment
        for item_i in range(full_item_features.shape[1]):
            # Computation of item features embeddings
            x_embeddings = self.x_model(full_item_features[:, item_i, :])

            stacked_heterogeneities = []
            # Computation of utilites from embeddings, iteration over heterogeneities
            # eps_x * eps_z
            for _x in x_embeddings:
                for _z in z_embeddings:
                    full_embedding = tf.keras.layers.Concatenate()(
                        [full_item_features[:, item_i, :], _x, stacked_contexts_features, _z]
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
        ### Reshape utilities: (batch_size, num_items, heterogeneity)
        return tf.squeeze(tf.stack(utilities, axis=1), -1)


def create_ff_network(
    input_shape, depth, width, activation="elu", add_last=False, l2_regularization_coeff=0.0
):
    """Base function to create a simple fully connected (Dense) network.

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

    Returns:
    --------
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
    """Function to create the three nets used in RUMnet: X_net, Z_net and U_net.

    Parameters
    ----------
    num_products_features : int
        Number of features each product will be described with.
        In terms of ChoiceDataset it is the number of { items_features + sessions_items_features }
        for one product.
    num_customer_features : int
        Number of features each customer will be described with.
        In terms of ChoiceDataset it is the number of sessions_features.
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

    Returns:
    --------
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
