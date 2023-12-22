"""Implementation of RUMnet for easy use."""
import tensorflow as tf

from choice_learn.models.base_model import ChoiceModel


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
        self.loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False, label_smoothing=self.label_smoothing
        )

    def compute_utility(
        self,
        items_features_batch,
        session_features_batch,
        session_items_features_batch,
        availabilities_batch,
        choices_batch,
    ):
        """Compute utility from a batch of ChoiceDataset.

        Here we asssume that: item features = {fixed item features + session item features}
                              user features = {session features}

        Parameters
        ----------
        items_features_batch : tuple of np.ndarray (items_features)
            Items-Features: formatting from ChoiceDataset: a matrix representing the
            products constant features.
        session_features_batch : tuple of np.ndarray (sessions_features)
            Time-Features
        session_items_features_batch :tuple of np.ndarray (sessions_items_features)
            Time-Item-Features
        availabilities_batch : np.ndarray
            Availabilities (sessions_items_availabilities)
        choices_batch :  np.ndarray
            Choices

        Returns:
        --------
        np.ndarray
            Utility of each product for each session.
            Shape must be (n_sessions, n_items)
        """
        del availabilities_batch, choices_batch
        ### Restacking of the item features
        items_features_batch = tf.concat([*items_features_batch], axis=-1)
        session_features_batch = tf.concat([*session_features_batch], axis=-1)
        session_items_features_batch = tf.concat([*session_items_features_batch], axis=-1)

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
        items_batch,
        sessions_batch,
        sessions_items_batch,
        availabilities_batch,
        choices_batch,
        sample_weight=None,
    ):
        """Modified version of train step, as we have to average probabilities over heterogeneities.

        Mayber split into two functions?
        One for computing probabilities, one for gradient descent ?
        Parameters to be renamed !
        Function that represents one training step (= one gradient descent step) of the model.

        Parameters
        ----------
        items_batch : tuple of np.ndarray (items_features)
            Fixed-Item-Features: formatting from ChoiceDataset: a matrix representing
            the products constant features.
        sessions_batch : tuple of np.ndarray (sessions_features)
            Time-Features
        sessions_items_batch : tuple of np.ndarray (sessions_items_features)
            Time-Item-Features
        availabilities_batch : np.ndarray
            Availabilities (sessions_items_availabilities)
        choices_batch : np.ndarray
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
            all_u = self.compute_utility(
                items_batch,
                sessions_batch,
                sessions_items_batch,
                availabilities_batch,
                choices_batch,
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
            """
            # Test with availability normalization
            probabilities = tf.multiply(probabilities, ia_batch)
            probabilities = tf.divide(
                probabilities, tf.reduce_sum(probabilities, axis=1, keepdims=True) + 1e-5
            )
            """
            # Probabilities of selected products
            # chosen_probabilities = tf.gather_nd(indices=choices_nd, params=probabilities)

            # Negative Log-Likelihood
            nll = self.loss(
                y_pred=probabilities,
                y_true=tf.one_hot(choices_batch, depth=probabilities.shape[1]),
                sample_weight=sample_weight,
            )
            # nll = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(
            #     y_pred=probabilities, y_true=c_batch
            # )
            # nll = -tf.reduce_sum(tf.math.log(chosen_probabilities + self.logmin))

        grads = tape.gradient(nll, self.weights)
        self.optimizer.apply_gradients(zip(grads, self.weights))
        return nll

    @tf.function
    def batch_predict(
        self,
        items_batch,
        sessions_batch,
        sessions_items_batch,
        availabilities_batch,
        choices_batch,
        sample_weight=None,
    ):
        """Function that represents one prediction (Probas + Loss) for one batch of a ChoiceDataset.

        Specific version for RUMnet because it is needed to average probabilities over
        heterogeneities.

        Parameters
        ----------
        items_batch : tuple of np.ndarray (items_features)
            Fixed-Item-Features: formatting from ChoiceDataset: a matrix representing the products
            constant features.
        sessions_batch : tuple of np.ndarray (sessions_features)
            Time-Features
        sessions_items_batch : tuple of np.ndarray (sessions_items_features)
            Time-Item-Features
        availabilities_batch : np.ndarray
            Availabilities (sessions_items_availabilities)
        choices_batch : np.ndarray
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
            items_batch, sessions_batch, sessions_items_batch, availabilities_batch, choices_batch
        )
        probabilities = tf.nn.softmax(utilities, axis=2)
        probabilities = tf.reduce_mean(probabilities, axis=1)

        # Test with availability normalization
        """
        probabilities = tf.multiply(probabilities, ia_batch)
        probabilities = tf.divide(
            probabilities, tf.reduce_sum(probabilities, axis=1, keepdims=True) + 1e-5
        )
        """
        batch_loss = self.loss(
            y_pred=probabilities,
            y_true=tf.one_hot(choices_batch, depth=probabilities.shape[1]),
            sample_weight=sample_weight,
        )
        return batch_loss, probabilities


class PaperRUMnet2(PaperRUMnet):
    """Other implementation."""

    def compute_utility(
        self,
        items_features_batch,
        session_features_batch,
        session_items_features_batch,
        availabilities_batch,
        choices_batch,
    ):
        """Compute utility from a batch of ChoiceDataset.

        Here we asssume that: item features = {fixed item features + session item features}
                              user features = {session features}

        Parameters
        ----------
        items_features_batch : tuple of np.ndarray (items_features)
            Items-Features: formatting from ChoiceDataset: a matrix representing
            the products constant features.
        session_features_batch : tuple of np.ndarray (sessions_features)
            Time-Features
        session_items_features_batch :tuple of np.ndarray (sessions_items_features)
            Time-Item-Features
        availabilities_batch : np.ndarray
            Availabilities (sessions_items_availabilities)
        choices_batch :  np.ndarray
            Choices

        Returns:
        --------
        np.ndarray
            Utility of each product for each session.
            Shape must be (n_sessions, n_items)
        """
        del availabilities_batch, choices_batch
        ### Restacking of the item features
        items_features_batch = tf.concat([*items_features_batch], axis=-1)
        session_features_batch = tf.concat([*session_features_batch], axis=-1)
        session_items_features_batch = tf.concat([*session_items_features_batch], axis=-1)

        full_item_features = tf.stack(
            [items_features_batch] * session_items_features_batch.shape[0], axis=0
        )
        full_item_features = tf.concat([session_items_features_batch, full_item_features], axis=-1)

        ### Computation of utilities
        utilities = []

        # Computation of the customer features embeddings
        z_embeddings = self.z_model(session_features_batch)

        # Iterate over items in
        def apply_u(x):
            return self.u_model(x)

        for item_i in range(full_item_features.shape[1]):
            # Computation of item features embeddings
            x_embeddings = self.x_model(full_item_features[:, item_i, :])

            # utilities.append([])

            # Computation of utilites from embeddings, iteration over heterogeneities
            # (eps_x * eps_z)
            _utilities = []
            for _x in x_embeddings:
                for _z in z_embeddings:
                    _u = tf.keras.layers.Concatenate()(
                        [full_item_features[:, item_i, :], _x, session_features_batch, _z]
                    )
                    _utilities.append(_u)
            utilities.append(
                tf.map_fn(
                    fn=apply_u, elems=tf.stack(_utilities, axis=0), fn_output_signature=tf.float32
                )
            )
        ### Reshape utilities: (batch_size, num_items, heterogeneity)
        return tf.transpose(tf.squeeze(tf.stack(utilities, axis=0), -1))


class PaperRUMnet3(PaperRUMnet):
    """Other Implementation."""

    def compute_utility(
        self,
        items_features_batch,
        session_features_batch,
        session_items_features_batch,
        availabilities_batch,
        choices_batch,
    ):
        """Compute utility from a batch of ChoiceDataset.

        Here we asssume that: item features = {fixed item features + session item features}
                              user features = {session features}

        Parameters
        ----------
        items_features_batch : tuple of np.ndarray (items_features)
            Items-Features: formatting from ChoiceDataset: a matrix representing the products
            constant features.
        session_features_batch : tuple of np.ndarray (sessions_features)
            Time-Features
        session_items_features_batch :tuple of np.ndarray (sessions_items_features)
            Time-Item-Features
        availabilities_batch : np.ndarray
            Availabilities (sessions_items_availabilities)
        choices_batch :  np.ndarray
            Choices

        Returns:
        --------
        np.ndarray
            Utility of each product for each session.
            Shape must be (n_sessions, n_items)
        """
        del availabilities_batch, choices_batch
        ### Restacking of the item features
        items_features_batch = tf.concat([*items_features_batch], axis=-1)
        session_features_batch = tf.concat([*session_features_batch], axis=-1)
        session_items_features_batch = tf.concat([*session_items_features_batch], axis=-1)

        full_item_features = tf.stack(
            [items_features_batch] * session_items_features_batch.shape[0], axis=0
        )
        full_item_features = tf.concat([session_items_features_batch, full_item_features], axis=-1)

        ### Computation of utilities
        utilities = []

        # Computation of the customer features embeddings
        z_embeddings = self.z_model(session_features_batch)

        # Iterate over items in assortment
        # for item_i in range(full_item_features.shape[1]):
        def apply_u(x):
            # Computation of item features embeddings
            x_embeddings = self.x_model(x)

            utilities = []

            # Computation of utilites from embeddings, iteration over heterogeneities
            # (eps_x * eps_z)
            for _x in x_embeddings:
                for _z in z_embeddings:
                    _u = tf.keras.layers.Concatenate()([x, _x, session_features_batch, _z])
                    utilities.append(self.u_model(_u))
            return tf.stack(utilities, axis=0)

        utilities = tf.map_fn(fn=apply_u, elems=tf.transpose(full_item_features, perm=[1, 0, 2]))
        ### Reshape utilities: (batch_size, num_items, heterogeneity)
        return tf.transpose(tf.squeeze(tf.stack(utilities, axis=0), -1))


class PaperRUMnet4(PaperRUMnet):
    """Other Implementation."""

    def compute_utility(
        self,
        items_features_batch,
        session_features_batch,
        session_items_features_batch,
        availabilities_batch,
        choices_batch,
    ):
        """Compute utility from a batch of ChoiceDataset.

        Here we asssume that: item features = {fixed item features + session item features}
                              user features = {session features}

        Parameters
        ----------
        items_features_batch : tuple of np.ndarray (items_features)
            Items-Features: formatting from ChoiceDataset: a matrix representing
            the products constant features.
        session_features_batch : tuple of np.ndarray (sessions_features)
            Time-Features
        session_items_features_batch :tuple of np.ndarray (sessions_items_features)
            Time-Item-Features
        availabilities_batch : np.ndarray
            Availabilities (sessions_items_availabilities)
        choices_batch :  np.ndarray
            Choices

        Returns:
        --------
        np.ndarray
            Utility of each product for each session.
            Shape must be (n_sessions, n_items)
        """
        del availabilities_batch, choices_batch
        ### Restacking of the item features
        items_features_batch = tf.concat([*items_features_batch], axis=-1)
        session_features_batch = tf.concat([*session_features_batch], axis=-1)
        session_items_features_batch = tf.concat([*session_items_features_batch], axis=-1)

        full_item_features = tf.stack(
            [items_features_batch] * session_items_features_batch.shape[0], axis=0
        )
        full_item_features = tf.concat([session_items_features_batch, full_item_features], axis=-1)

        ### Computation of utilities
        utilities = []
        batch_size = session_features_batch.shape[0]

        # Computation of the customer features embeddings
        z_embeddings = self.z_model(session_features_batch)

        # Iterate over items in assortment
        for item_i in range(full_item_features.shape[1]):
            # Computation of item features embeddings
            x_embeddings = self.x_model(full_item_features[:, item_i, :])

            # utilities.append([])
            _utilities = []
            # Computation of utilites from embeddings, iteration over heterogeneities
            # (eps_x * eps_z)
            for _x in x_embeddings:
                for _z in z_embeddings:
                    _u = tf.keras.layers.Concatenate()(
                        [full_item_features[:, item_i, :], _x, session_features_batch, _z]
                    )
                    _utilities.append(_u)
            item_utilities = self.u_model(tf.concat(_utilities, axis=0))
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


class PaperRUMnet5(PaperRUMnet):
    """Other Implementation."""

    def compute_utility(
        self,
        items_features_batch,
        session_features_batch,
        session_items_features_batch,
        availabilities_batch,
        choices_batch,
    ):
        """Compute utility from a batch of ChoiceDataset.

        Here we asssume that: item features = {fixed item features + session item features}
                              user features = {session features}

        Parameters
        ----------
        items_features_batch : tuple of np.ndarray (items_features)
            Items-Features: formatting from ChoiceDataset: a matrix representing
            the products constant features.
        session_features_batch : tuple of np.ndarray (sessions_features)
            Time-Features
        session_items_features_batch :tuple of np.ndarray (sessions_items_features)
            Time-Item-Features
        availabilities_batch : np.ndarray
            Availabilities (sessions_items_availabilities)
        choices_batch :  np.ndarray
            Choices

        Returns:
        --------
        np.ndarray
            Utility of each product for each session.
            Shape must be (n_sessions, n_items)
        """
        del availabilities_batch, choices_batch
        ### Restacking of the item features
        items_features_batch = tf.concat([*items_features_batch], axis=-1)
        session_features_batch = tf.concat([*session_features_batch], axis=-1)
        session_items_features_batch = tf.concat([*session_items_features_batch], axis=-1)

        full_item_features = tf.stack(
            [items_features_batch] * session_items_features_batch.shape[0], axis=0
        )
        full_item_features = tf.concat([session_items_features_batch, full_item_features], axis=-1)

        ### Computation of utilities
        utilities = []
        batch_size = session_features_batch.shape[0]
        num_items = full_item_features.shape[1]

        # Computation of the customer features embeddings
        z_embeddings = self.z_model(session_features_batch)

        _utilities = []
        # Iterate over items in assortment
        for item_i in range(num_items):
            # Computation of item features embeddings
            x_embeddings = self.x_model(full_item_features[:, item_i, :])

            # utilities.append([])
            # Computation of utilites from embeddings, iteration over heterogeneities
            # (eps_x * eps_z)
            for _x in x_embeddings:
                for _z in z_embeddings:
                    _u = tf.keras.layers.Concatenate()(
                        [full_item_features[:, item_i, :], _x, session_features_batch, _z]
                    )
                    _utilities.append(_u)
        utilities = self.u_model(tf.concat(_utilities, axis=0))
        length_one_item = len(x_embeddings) * len(z_embeddings) * batch_size
        reshaped_utilities = []
        for item_i in range(num_items):
            item_utilities = tf.stack(
                [
                    utilities[
                        item_i * length_one_item + batch_size * i : item_i * length_one_item
                        + batch_size * (i + 1)
                    ]
                    for i in range(len(x_embeddings) * len(z_embeddings))
                ],
                axis=1,
            )
            print(item_i, "item_u", item_utilities.shape)
            reshaped_utilities.append(item_utilities)
        ### Reshape utilities: (batch_size, num_items, heterogeneity)
        utilities = tf.squeeze(tf.stack(reshaped_utilities, axis=1), -1)
        print("u", utilities.shape)
        # utilities = tf.stack(utilities, axis=0)
        return utilities


def create_ff_network(input_shape, depth, width, add_last=False, l2_regularization_coeff=0.0):
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
            width, activation="elu", kernel_regularizer=regularizer, use_bias=True
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
