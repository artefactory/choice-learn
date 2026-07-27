"""Implementation of an attention-based model for item recommendation."""

from typing import Union

import numpy as np
import tensorflow as tf

from .base_basket_model import BaseBasketModel


class SelfAttentionModelScalar(BaseBasketModel):
    """Class for the self attention model for basket recommendation.

    Basket Choice Modeling
    Inspired by SelfAttentionModel but using dot product instead of distance for embedding

    """

    def __init__(
        self,
        latent_sizes: dict[str, int] = {"short_term": 10, "long_term": 10, "price": 4},
        hinge_margin: float = 0.5,
        short_term_ratio: float = 0.5,
        n_negative_samples: int = 2,
        optimizer: str = "adam",
        callbacks: Union[tf.keras.callbacks.CallbackList, None] = None,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 32,
        grad_clip_value: Union[float, None] = None,
        weight_decay: Union[float, None] = None,
        momentum: float = 0.0,
        l2_regularization: float = 0.0,
        dropout_rate: float = 0.0,
        intercept: bool = True,
        price_effects: bool = False,
        store_effects: bool = False,
        epsilon_price: float = 1e-4,
        value_matrix: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the model with hyperparameters.

        Parameters
        ----------
        latent_size : int
            Size of the item embeddings.
        hinge_margin : float
            Margin parameter for the hinge loss.
        short_term_weight : float
            Weighting factor between long-term and short-term preferences.
        n_negative_samples : int
            Number of negative samples to use in training.
        optimizer : str
            Optimizer to use for training. Default is "Adam".
        callbacks : tf.keras.callbacks.CallbackList or None
            List of callbacks to use during training. Default is None.
        lr : float
            Learning rate for the optimizer.
        epochs : int
            Number of training epochs.
        batch_size : int
            Size of the batches for training. Default is 32.
        grad_clip_value : float or None
            Value for gradient clipping. Default is None (no clipping).
        weight_decay : float or None
            Weight decay (L2 regularization) factor. Default is None (no weight decay).
        momentum : float
            Momentum factor for optimizers that support it. Default is 0.0.
        item_intercept: bool, optional
            Whether to include item intercept in the model, by default True
        price_effects: bool, optional
            Whether to include price effects in the model, by default True
        epsilon_price: float, optional
            Epsilon value to add to prices to avoid NaN values (log(0)), by default 1e-4
        """
        self.instantiated = False

        for val in latent_sizes.keys():
            if val not in ["short_term", "long_term", "price"]:
                raise ValueError(f"Unknown value for latent_sizes dict: {val}.")
        if "short_term" not in latent_sizes:
            latent_sizes["short_term"] = 10
        if "long_term" not in latent_sizes:
            latent_sizes["long_term"] = 10
        if "price" not in latent_sizes:
            latent_sizes["price"] = 4

        self.hinge_margin = hinge_margin
        self.short_term_ratio = short_term_ratio
        self.n_negative_samples = n_negative_samples

        self.latent_sizes = latent_sizes
        self.d = self.latent_sizes["short_term"]
        self.d_long = self.latent_sizes["long_term"]
        self.l2_regularization = l2_regularization
        self.dropout_rate = dropout_rate
        self.item_intercept = intercept
        self.price_effects = price_effects
        self.store_effects = store_effects
        self.epsilon_price = epsilon_price
        self.value_matrix = value_matrix
        super().__init__(
            optimizer=optimizer,
            callbacks=callbacks,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            grad_clip_value=grad_clip_value,
            weight_decay=weight_decay,
            momentum=momentum,
            **kwargs,
        )

    def instantiate(
        self,
        n_items: int,
        n_users: int,
        n_stores: int,
    ) -> None:
        """Initialize the model parameters.

        Parameters
        ----------
        n_items : int
            Number of unique items in the dataset.
        n_users : int
            Number of unique users in the dataset.

        Variables
        ----------
        X : tf.Variable
            Item embedding matrix for short-term preferences, size (n_items, d).
        V : tf.Variable
            Item embedding matrix for long-term preferences, size (n_items, d_long).
        U : tf.Variable
            User embedding matrix for long-term preferences, size (n_users, d_long).
        Wq : tf.Variable
            Weight matrix for query transformation in attention mechanism, size (d, d).
        Wk : tf.Variable
            Weight matrix for key transformation in attention mechanism, size (d, d).
        """
        self.n_items = n_items
        self.n_users = n_users
        self.n_stores = n_stores
        ##############
        if self.store_effects:
            self.theta = tf.Variable(
                tf.random_normal_initializer(mean=0, stddev=0.1, seed=42)(
                    shape=(n_stores, self.d)
                ),  # Dimension for 1 item: latent_sizes["preferences"]
                trainable=True,
                name="theta",
            )
        if self.price_effects:
            # Add price sensitivity
            self.beta = tf.Variable(
                tf.random_normal_initializer(mean=0, stddev=0.1, seed=42)(
                    shape=(n_items, self.latent_sizes["price"])
                ),  # Dimension for 1 item: latent_sizes["price"]
                trainable=True,
                name="beta",
            )
            self.delta = tf.Variable(
                tf.random_normal_initializer(mean=0, stddev=0.1, seed=42)(
                    shape=(n_stores, self.latent_sizes["price"])
                ),  # Dimension for 1 item: latent_sizes["price"]
                trainable=True,
                name="delta",
            )

        if self.item_intercept:
            self.alpha = tf.Variable(
                tf.random_normal_initializer(mean=0, stddev=0.01, seed=42)(
                    shape=(self.n_items,)
                ),  # Dimension for 1 item: latent_sizes["short_term"]
                trainable=True,
                name="alpha",
            )
        self.X = tf.Variable(
            tf.random_normal_initializer(mean=0, stddev=0.01, seed=42)(shape=(n_items, self.d)),
            trainable=True,
            name="X",
        )

        self.V = tf.Variable(
            tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=42)(
                shape=(n_items, self.d_long)
            ),
            trainable=True,
            name="V",
        )

        self.U = tf.Variable(
            tf.random_normal_initializer(mean=0, stddev=0.01, seed=42)(
                shape=(self.n_users, self.d_long)
            ),
            trainable=True,
            name="U",
        )

        self.Wq = tf.Variable(
            tf.random_normal_initializer(mean=0, stddev=0.01, seed=42)(shape=(self.d, self.d)),
            trainable=True,
            name="Wq",
        )

        self.Wk = tf.Variable(
            tf.random_normal_initializer(mean=0, stddev=0.01, seed=42)(shape=(self.d, self.d)),
            trainable=True,
            name="Wk",
        )
        if self.value_matrix:
            self.Wv = tf.Variable(
                tf.random_normal_initializer(mean=0, stddev=0.01, seed=42)(shape=(self.d, self.d)),
                trainable=True,
                name="Wv",
            )

        self.instantiated = True

    @property
    def trainable_weights(self):
        """Return the trainable weights of the model.

        Returns
        -------
            list
                List of trainable weights (X, V, U, Wq, Wk).
        """
        weights = [self.X, self.V, self.U, self.Wq, self.Wk]
        if self.value_matrix:
            weights.extend([self.Wv])
        if self.item_intercept:
            weights.extend([self.alpha])
        if self.price_effects:
            weights.extend([self.beta, self.delta])

        return weights

    @property
    def train_iter_method(self) -> str:
        """Method used to generate sub-baskets from a purchased one.

        Available methods are:
        - 'shopper': randomly orders the purchases and creates the ordered sub-baskets:
                        (1|0); (2|1); (3|1,2); (4|1,2,3); etc...
        - 'aleacarta': creates all the sub-baskets with N-1 items:
                        (4|1,2,3); (3|1,2,4); (2|1,3,4); (1|2,3,4)

        Returns
        -------
        str
            Data generation method.
        """
        return "aleacarta"

    def masked_attention(self, basket_batch, scaled_scores):
        """Compute the masked attention weights.

        Applying a mask to ignore padding items. Also applied a mask on
        the diagonal to avoid attending to the same item, if activated
        """
        # batch_size = tf.shape(basket_batch)[0]
        mask = tf.not_equal(
            basket_batch, self.n_items
        )  # shape: (batch_size, L), True si pas padding

        if tf.shape(basket_batch)[1] == 1:
            attention_weights = tf.ones_like(scaled_scores)  # Shape: (batch_size, L, 1)

        else:
            # Diagonal mask to avoid attending to the same item
            # diag_mask = tf.eye(tf.shape(basket_batch)[1], batch_shape=[batch_size], dtype=tf.bool)
            # scaled_scores = tf.where(
            #    diag_mask,
            #    tf.constant(-np.inf, dtype=scaled_scores.dtype),
            #    scaled_scores,
            # )

            # Masque des padding items
            mask_col = tf.expand_dims(mask, axis=1)  # (batch_size, 1, L)
            scaled_scores = tf.where(
                mask_col, scaled_scores, tf.constant(-np.inf, dtype=scaled_scores.dtype)
            )

            all_inf_row = tf.reduce_all(tf.math.is_inf(scaled_scores), axis=-1)  # (batch_size, L)
            # We set to zero the first value of the rows where all values are -inf to avoid NaNs in
            # softmax
            indices = tf.where(all_inf_row)
            indices_full = tf.concat([indices, tf.zeros_like(indices[:, :1])], axis=1)
            updates = tf.zeros([tf.shape(indices_full)[0]], dtype=scaled_scores.dtype)
            scaled_scores = tf.tensor_scatter_nd_update(scaled_scores, indices_full, updates)

            attention_weights = tf.nn.softmax(scaled_scores, axis=-1)  # Shape: (batch_size, L, L)

        return attention_weights

    def embed_basket(self, basket_batch: tf.Tensor, is_training: bool = False) -> tf.Tensor:
        """Return the context embedding matrix.

        Parameters
        ----------
            basket_batch : tf.Tensor
                [batch_size, L]
                Tensor containing the list of the context items.
            is_training : bool
                Whether the model is in training mode or not, to activate dropout if needed.

        Returns
        -------
            basket_embedding : tf.Tensor
                [batch_size, latent_size] tf.Tensor
                Tensor containing the vector of contexts embeddings.
            attention_weights : tf.Tensor
                [batch_size, L, L] tf.Tensor
                Tensor containing the attention matrix.
        """
        padding_vector = tf.zeros(shape=[1, self.d])  # Shape (1, d)
        padded_items = tf.concat([self.X, padding_vector], axis=0)
        x_basket = tf.gather(padded_items, indices=basket_batch)  # Shape: (batch_size, L, d)

        q_prime = tf.nn.relu(tf.matmul(x_basket, self.Wq))  # Shape: (batch_size, L, d)
        k_prime = tf.nn.relu(tf.matmul(x_basket, self.Wk))

        if is_training:
            q_prime = tf.nn.dropout(q_prime, rate=self.dropout_rate)
            k_prime = tf.nn.dropout(k_prime, rate=self.dropout_rate)

        scores = tf.matmul(q_prime, k_prime, transpose_b=True)
        scaled_scores = scores / tf.sqrt(float(self.d))
        attention_weights = self.masked_attention(
            basket_batch, scaled_scores
        )  # Shape: (batch_size, L, L)

        if self.value_matrix:
            value_matrix = tf.nn.relu(tf.matmul(x_basket, self.Wv))  # Shape: (batch_size, L, d)
            attention_output = tf.matmul(
                attention_weights, value_matrix
            )  # Shape: (batch_size, L, d)
        else:
            attention_output = tf.matmul(attention_weights, x_basket)  # Shape: (batch_size, L, d)

        mask = tf.not_equal(basket_batch, self.n_items)
        mask_float = tf.cast(mask, dtype=tf.float32)
        mask_float = tf.expand_dims(mask_float, axis=-1)
        masked_attention_output = attention_output * mask_float  # (batch_size, L, d)

        # Number of items in each basket (excluding padding)
        num_items_by_basket = tf.reduce_sum(mask_float, axis=1)  # (batch_size, 1)

        basket_embedding = tf.math.divide_no_nan(
            tf.reduce_sum(masked_attention_output, axis=1, keepdims=True),
            num_items_by_basket[:, tf.newaxis, :],
        )
        basket_embedding = tf.squeeze(basket_embedding, axis=1)  # Shape: (batch_size,d)

        return basket_embedding, attention_weights

    def compute_batch_short_utility(
        self,
        item_batch: Union[np.ndarray, tf.Tensor],
        basket_embedding: tf.Tensor,
    ) -> tf.Tensor:
        """Compute the short distance of the items in item_batch given the items in basket_batch.

        Parameters
        ----------
        item_batch: or tf.Tensor
            Batch of the purchased items ID (integers) for which to compute the distance from their
            basket.
            Shape must be (batch_size,None)
            (positive and negative samples concatenated together)
        basket_embedding: tf.Tensor
            Batch of context embeddings for each purchased item
            Shape must be (batch_size, latent_size)

        Returns
        -------
        short_term_distance: tf.Tensor
            Distance of all the items in item_batch from their ground truth embedding (X)
            Shape must be (batch_size,)
        """
        x_item_target = tf.gather(self.X, indices=item_batch)  # Shape: (batch_size, None, d)

        return tf.reduce_sum(
            tf.expand_dims(basket_embedding, axis=1) * x_item_target, axis=-1
        )  # Shape: (batch_size, None)

    def compute_batch_long_utility(
        self,
        item_batch: Union[np.ndarray, tf.Tensor],
        user_batch: np.ndarray,
    ) -> tf.Tensor:
        """Compute the long distance of all the items in item_batch given the user.

        Parameters
        ----------
        item_batch: np.ndarray or tf.Tensor
            Batch of the purchased items ID (integers) for which to compute the distance from their
            user.
            Shape must be (batch_size,)
            (positive and negative samples concatenated together)

        user_batch: np.ndarray
            Batch of user IDs (integers) for each purchased item
            Shape must be (batch_size,)

        Returns
        -------
        long_term_distance: tf.Tensor
            Distance of all the items in item_batch from their ground truth embedding (V)
            Shape must be (batch_size,)
        """
        v_future_batch = tf.gather(self.V, indices=item_batch)  # Shape: (batch_size, d)

        u_user_batch = tf.gather(self.U, indices=user_batch)  # Shape: (batch_size, d)
        return tf.reduce_sum(
            tf.expand_dims(u_user_batch, axis=1) * v_future_batch, axis=-1
        )  # Shape: (batch_size, 1)

    def compute_batch_utility(
        self,
        item_batch: np.ndarray,
        basket_batch: np.ndarray,
        price_batch: np.ndarray,
        week_batch: np.ndarray,
        store_batch: np.ndarray,
        available_item_batch: np.ndarray,
        user_batch: np.ndarray,
        is_training: bool = False,
    ) -> tf.Tensor:
        """Compute the total distance (long + short term) of all the items in item_batch.

        Parameters
        ----------
        item_batch: np.ndarray
            Batch of the purchased items ID (integers) for which to compute the distance from their
            basket.
            Shape must be (batch_size, None)
            (positive and negative samples concatenated together)
        basket_batch: np.ndarray
            Batch of baskets (ID of items already in the baskets) (arrays) for each purchased item
            Shape must be (batch_size, max_basket_size)
        user_batch: np.ndarray
            Batch of user IDs (integers) for each purchased item
            Shape must be (batch_size,)
        is_training : bool
            Whether the model is in training mode or not, to activate dropout if needed.

        Returns
        -------
        total_distance: tf.Tensor
            Total distance of all the items in item_batch from their ground truth embeddings
            Shape must be (batch_size, None)
        """
        _ = week_batch
        _ = available_item_batch
        basket_batch_ragged = tf.cast(
            tf.ragged.boolean_mask(basket_batch, basket_batch != -1),
            dtype=tf.int32,
        )
        basket_batch = basket_batch_ragged.to_tensor(self.n_items)
        basket_embedding, _ = self.embed_basket(basket_batch, is_training)  # Shape: (batch_size, d)

        long_utility = self.compute_batch_long_utility(item_batch, user_batch)

        short_utility = self.compute_batch_short_utility(
            item_batch, basket_embedding
        ) + self.compute_psi(item_batch, price_batch, store_batch)
        return self.short_term_ratio * long_utility + (1 - self.short_term_ratio) * short_utility

    def get_negative_samples(
        self,
        available_items: np.ndarray,
        purchased_items: np.ndarray,
        next_item: int,
        n_samples: int,
    ) -> list[int]:
        """Sample randomly a set of items.

        (set of items not already purchased and *not necessarily* from the basket)

        Parameters
        ----------
        available_items: np.ndarray
            Matrix indicating the availability (1) or not (0) of the products
            Shape must be (n_items,)
        purchased_items: np.ndarray
            List of items already purchased (already in the basket)
        next_item: int
            Next item (to be added in the basket)
        n_samples: int
            Number of samples to draw

        Returns
        -------
        list[int]
            Random sample of items, each of them distinct from
            the next item and from the items already in the basket
        """
        # Convert inputs to tensors
        available_items = tf.cast(tf.convert_to_tensor(available_items), dtype=tf.int32)
        purchased_items = tf.cast(tf.convert_to_tensor(purchased_items), dtype=tf.int32)
        next_item = tf.cast(tf.convert_to_tensor(next_item), dtype=tf.int32)

        # Get the list of available items based on the availability matrix
        item_ids = tf.range(self.n_items)
        available_mask = tf.equal(available_items, 1)
        assortment = tf.boolean_mask(item_ids, available_mask)

        not_to_be_chosen = tf.concat([purchased_items, tf.expand_dims(next_item, axis=0)], axis=0)

        # Sample negative items from the assortment excluding not_to_be_chosen
        negative_samples = tf.boolean_mask(
            tensor=assortment,
            # Reduce the 2nd dimension of the boolean mask to get a 1D mask
            mask=~tf.reduce_any(
                tf.equal(tf.expand_dims(assortment, axis=1), not_to_be_chosen), axis=1
            ),
        )

        error_message = (
            "The number of negative samples to draw must be less than "
            "the number of available items not already purchased and "
            "distinct from the next item."
        )
        # Raise an error if n_samples > tf.size(negative_samples)
        tf.debugging.assert_greater_equal(
            tf.size(negative_samples), n_samples, message=error_message
        )

        # Randomize the sampling
        negative_samples = tf.random.shuffle(negative_samples)

        # Keep only n_samples
        return negative_samples[:n_samples]

    def compute_psi(
        self,
        item_batch: Union[np.ndarray, tf.Tensor],
        price_batch: Union[np.ndarray, tf.Tensor],
        store_batch: Union[np.ndarray, tf.Tensor],
    ) -> tf.Tensor:
        """Compute the psi part of the utility of all the items in item_batch.

        Parameters
        ----------
        item_batch: np.ndarray or tf.Tensor
            Batch of the purchased items ID (integers) for which to compute the utility
            Shape must be (batch_size,None)
            (positive and negative samples concatenated together)
        basket_batch: np.ndarray
            Batch of baskets (ID of items already in the baskets) (arrays) for each purchased item
            Shape must be (batch_size, max_basket_size)
        store_batch: np.ndarray
            Batch of store IDs (integers) for each purchased item
            Shape must be (batch_size,)
        """
        store_batch = tf.cast(store_batch, dtype=tf.int32)
        price_batch = tf.cast(price_batch, dtype=tf.float32)
        x_item = tf.gather(self.X, indices=item_batch)  # Shape: (batch_size, None, d)

        if self.store_effects:
            theta_store = tf.gather(self.theta, indices=store_batch)
            # Compute the dot product along the last dimension
            store_preferences = tf.einsum("kj,klj->kl", theta_store, x_item)
        else:
            store_preferences = tf.zeros_like(
                item_batch, dtype=tf.float32
            )  # Shape: (batch_size,None)

        if self.item_intercept:
            item_intercept = tf.gather(self.alpha, indices=item_batch)  # Shape: (batch_size,None)
        else:
            item_intercept = tf.zeros_like(store_preferences)
        if self.price_effects:
            delta_store = tf.gather(self.delta, indices=store_batch)
            beta_item = tf.gather(self.beta, indices=item_batch)
            # Add epsilon to avoid NaN values (log(0))
            price_effects = (
                -1
                # Compute the dot product along the last dimension
                * tf.einsum("kj,klj->kl", delta_store, beta_item)
                * tf.math.log(price_batch + self.epsilon_price)
            )

        else:
            delta_store = tf.zeros_like(store_batch)
            price_effects = tf.zeros_like(store_preferences)

        # The effects of item intercept, store preferences, price and sensitivity
        # are combined in the per-item per-trip latent variable

        return tf.reduce_sum(
            [
                item_intercept,
                store_preferences,
                price_effects,
            ],
            axis=0,
        )  # Shape: (batch_size,None)

    # @tf.function  # Graph mode
    def compute_batch_loss(
        self,
        item_batch: np.ndarray,
        basket_batch: np.ndarray,
        future_batch: np.ndarray,
        store_batch: np.ndarray,
        week_batch: np.ndarray,
        price_batch: np.ndarray,
        available_item_batch: np.ndarray,
        user_batch: np.ndarray,
        is_training: bool = True,
    ) -> tuple[tf.Variable]:
        """Compute total loss.

        Parameters
        ----------
        item_batch: np.ndarray
            Batch of purchased items ID (integers)
            Shape must be (batch_size,)
        basket_batch: np.ndarray
            Batch of baskets (ID of items already in the baskets) (arrays) for each purchased item
            Shape must be (batch_size, max_basket_size)
        future_batch: np.ndarray
            Batch of items to be purchased in the future (ID of items not yet in the
            basket) (arrays) for each purchased item
            Shape must be (batch_size, max_basket_size)
            Here for signature reasons, unused for this model
        store_batch: np.ndarray
            Batch of store IDs (integers) for each purchased item
            Shape must be (batch_size,)
        week_batch: np.ndarray
            Batch of week numbers (integers) for each purchased item
            Shape must be (batch_size,)
        price_batch: np.ndarray
            Batch of prices (floats) for each purchased item
            Shape must be (batch_size,)
        available_item_batch: np.ndarray
            List of availability matrices (indicating the availability (1) or not (0)
            of the products) (arrays) for each purchased item
            Shape must be (batch_size, n_items)
        user_batch: np.ndarray
            Batch of user IDs (integers) for each purchased item
            Shape must be (batch_size,)
        is_training: bool
            Whether the model is in training mode or not, to activate dropout if needed.
            True by default, cause compute_batch_loss is only used during training.

        Returns
        -------
        tf.Variable
            Value of the loss for the batch (Hinge loss),
            Shape must be (1,)
        _: None
            Placeholder to match the signature of the parent class method
        """
        _ = future_batch  # Unused for this model
        batch_size = len(item_batch)

        negative_samples = tf.stack(
            [
                self.get_negative_samples(
                    available_items=available_item_batch[idx],
                    purchased_items=basket_batch[idx],
                    next_item=item_batch[idx],
                    n_samples=self.n_negative_samples,
                )
                for idx in range(batch_size)
            ],
            axis=0,
        )  # Shape: (batch_size, n_negative_samples)

        item_batch = tf.cast(item_batch, tf.int32)
        negative_samples = tf.cast(negative_samples, tf.int32)

        augmented_item_batch = tf.cast(
            tf.concat([tf.expand_dims(item_batch, axis=-1), negative_samples], axis=1),
            dtype=tf.int32,
        )  # Shape: (batch_size, 1 + n_negative_samples)

        basket_batch_ragged = tf.cast(
            tf.ragged.boolean_mask(basket_batch, basket_batch != -1),
            dtype=tf.int32,
        )
        basket_batch = basket_batch_ragged.to_tensor(self.n_items)
        augmented_price_batch = tf.gather(
            params=price_batch, indices=augmented_item_batch, batch_dims=1
        )  # Shape: (batch_size, 1 + n_negative_samples)
        all_utilities = self.compute_batch_utility(
            item_batch=augmented_item_batch,
            basket_batch=basket_batch,
            store_batch=store_batch,
            week_batch=week_batch,
            price_batch=augmented_price_batch,
            available_item_batch=available_item_batch,
            user_batch=user_batch,
            is_training=is_training,
        )  # Shape: (batch_size, 1 + n_negative_samples)

        positive_samples_utility = tf.gather(params=all_utilities, indices=[0], axis=1)
        negative_samples_utility = tf.gather(
            params=all_utilities, indices=tf.range(1, self.n_negative_samples + 1), axis=1
        )  # (batch_size, n_negative_samples)

        ridge_regularization = self.l2_regularization * tf.add_n(
            [tf.nn.l2_loss(weight) for weight in self.trainable_weights]
        )
        epsilon = 0.0
        loglikelihood = tf.reduce_sum(
            tf.math.log(
                tf.sigmoid(
                    tf.tile(
                        positive_samples_utility,
                        [1, self.n_negative_samples],
                    )
                    - negative_samples_utility
                )
                + epsilon
            ),
        )  # Shape of loglikelihood: (1,))
        bce = tf.keras.backend.binary_crossentropy(
            # Target: 1 for positive samples, 0 for negative samples
            target=tf.concat(
                [
                    tf.ones_like(positive_samples_utility),
                    tf.zeros_like(negative_samples_utility),
                ],
                axis=1,
            ),
            output=tf.nn.sigmoid(all_utilities),
        )  # Shape: (batch_size * (n_negative_samples + 1),)
        return tf.reduce_sum(bce + ridge_regularization) / (
            batch_size * (self.n_negative_samples + 1)
        ), loglikelihood
