"""Implementation of an attention-based model for item recommendation."""

from typing import Union

import numpy as np
import tensorflow as tf

from .base_basket_model import BaseBasketModel
from .data.basket_dataset import TripDataset


class SelfAttentionModel(BaseBasketModel):
    """Class for the self attention model for basket recommendation.

    Basket Choice Modeling
    Inspired by the paper: "Next Item Recommendation with Self-Attention"  Shuai Zhang, Lina Yao,  Yi Tay, and Aixin Sun.
    The algorithm was modified and adapted to the basket recommendation task.
    """

    def __init__(
        self,
        latent_sizes: dict[str, int] = {"short_term": 10, "long_term": 10},
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
        """
        self.instantiated = False

        for val in latent_sizes.keys():
            if val not in ["short_term", "long_term"]:
                raise ValueError(f"Unknown value for latent_sizes dict: {val}.")

        self.hinge_margin = hinge_margin
        self.short_term_ratio = short_term_ratio
        self.n_negative_samples = n_negative_samples

        self.latent_sizes = latent_sizes
        self.d = self.latent_sizes["short_term"]
        self.d_long = self.latent_sizes["long_term"]
        self.l2_regularization = l2_regularization
        self.dropout_rate = dropout_rate

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

        if len(tf.config.get_visible_devices("GPU")):
            # At least one available GPU
            self.on_gpu = True
        else:
            # No available GPU
            self.on_gpu = False

    def instantiate(
        self,
        n_items: int,
        n_users: int,
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
        ##############

        self.X = tf.Variable(
            tf.random_normal_initializer(mean=0, stddev=0.01, seed=42)(
                shape=(n_items, self.d)
            ),
            trainable=True,
            constraint=tf.keras.constraints.MaxNorm(max_value=1.0, axis=1),
            name="X",
        )

        self.V = tf.Variable(
            tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=42)(
                shape=(n_items, self.d_long)
            ),
            trainable=True,
            constraint=tf.keras.constraints.MaxNorm(max_value=1.0, axis=1),
            name="V",
        )

        self.U = tf.Variable(
            tf.random_normal_initializer(mean=0, stddev=0.01, seed=42)(
                shape=(self.n_users, self.d_long)
            ),
            trainable=True,
            constraint=tf.keras.constraints.MaxNorm(max_value=1.0, axis=1),
            name="U",
        )

        self.Wq = tf.Variable(
            tf.random_normal_initializer(mean=0, stddev=0.01, seed=42)(
                shape=(self.d, self.d)
            ),
            trainable=True,
            name="Wq",
        )

        self.Wk = tf.Variable(
            tf.random_normal_initializer(mean=0, stddev=0.01, seed=42)(
                shape=(self.d, self.d)
            ),
            trainable=True,
            name="Wk",
        )

        self.is_trained = False
        self.instantiated = True

    @property
    def trainable_weights(self):
        """Return the trainable weights of the model.

        Returns
        -------
            list
                List of trainable weights (X, V, U, Wq, Wk).
        """
        return [self.X, self.V, self.U, self.Wq, self.Wk]

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

        batch_size = tf.shape(basket_batch)[0]
        mask = tf.not_equal(
            basket_batch, self.n_items
        )  # shape: (batch_size, L), True si pas padding

        if tf.shape(basket_batch)[1] == 1:
            attention_weights = tf.ones_like(scaled_scores)  # Shape: (batch_size, L, 1)

        else:
            # Masque de la diagonale, désactivé pour l'instant
            diag_mask = tf.eye(
                tf.shape(basket_batch)[1], batch_shape=[batch_size], dtype=tf.bool
            )
            scaled_scores = tf.where(
                diag_mask,
                tf.constant(-np.inf, dtype=scaled_scores.dtype),
                scaled_scores,
            )

            # Masque des padding items
            mask_col = tf.expand_dims(mask, axis=1)  # (batch_size, 1, L)
            scaled_scores = tf.where(
                mask_col, scaled_scores, tf.constant(-np.inf, dtype=scaled_scores.dtype)
            )

            all_inf_row = tf.reduce_all(
                tf.math.is_inf(scaled_scores), axis=-1
            )  # (batch_size, L)
            # We set to zero the first value of the rows where all values are -inf to avoid NaNs in softmax
            indices = tf.where(all_inf_row)
            indices_full = tf.concat([indices, tf.zeros_like(indices[:, :1])], axis=1)
            updates = tf.zeros([tf.shape(indices_full)[0]], dtype=scaled_scores.dtype)
            scaled_scores = tf.tensor_scatter_nd_update(
                scaled_scores, indices_full, updates
            )

            attention_weights = tf.nn.softmax(
                scaled_scores, axis=-1
            )  # Shape: (batch_size, L, L)

        return attention_weights

    def embed_context(self, context_items: tf.Tensor, is_training: bool) -> tf.Tensor:
        """Return the context embedding matrix.

        Parameters
        ----------
            context_items : tf.Tensor
                [batch_size, L]
                Tensor containing the list of the context items.

        Returns
        -------
            m_batch : tf.Tensor
                [batch_size, latent_size] tf.Tensor
                Tensor containing the vector of contexts embeddings.
            attention_weights : tf.Tensor
                [batch_size, L, L] tf.Tensor
                Tensor containing the attention matrix.
        """

        # self.X.assign(tf.clip_by_norm(self.X, clip_norm=1.0, axes=1))
        padding_vector = tf.zeros(shape=[1, self.d])  # Forme (1, d)
        padded_X = tf.concat([self.X, padding_vector], axis=0)
        X_future_batch = tf.gather(
            padded_X, indices=context_items
        )  # Shape: (batch_size, L, d)

        Q_prime = tf.nn.relu(
            tf.matmul(X_future_batch, self.Wq)
        )  # Shape: (batch_size, L, d)
        K_prime = tf.nn.relu(tf.matmul(X_future_batch, self.Wk))

        if is_training:
            Q_prime = tf.nn.dropout(Q_prime, rate=self.dropout_rate)
            K_prime = tf.nn.dropout(K_prime, rate=self.dropout_rate)

        scores = tf.matmul(Q_prime, K_prime, transpose_b=True)
        scaled_scores = scores / tf.sqrt(float(self.d))
        attention_weights = self.masked_attention(
            context_items, scaled_scores
        )  # Shape: (batch_size, L, L)

        attention_output = tf.matmul(
            attention_weights, X_future_batch
        )  # Shape: (batch_size, L, d)

        mask = tf.not_equal(context_items, self.n_items)
        mask_float = tf.cast(mask, dtype=tf.float32)
        mask_float = tf.expand_dims(mask_float, axis=-1)
        masked_attention_output = attention_output * mask_float  # (batch_size, L, d)

        # Number of items in each basket (excluding padding)
        num_items_by_basket = tf.reduce_sum(mask_float, axis=1)  # (batch_size, 1)

        m_batch = tf.math.divide_no_nan(
            tf.reduce_sum(masked_attention_output, axis=1, keepdims=True),
            num_items_by_basket[:, tf.newaxis, :],
        )
        m_batch = tf.squeeze(m_batch, axis=1)  # Shape: (batch_size,)
        return m_batch, attention_weights

    def compute_batch_short_distance(
        self,
        item_batch: Union[np.ndarray, tf.Tensor],
        m_batch: tf.Tensor,
    ) -> tf.Tensor:
        """Compute the short distance of all the items in item_batch given the
        items in basket_batch.

        Parameters
        ----------
        item_batch: or tf.Tensor
            Batch of the purchased items ID (integers) for which to compute the utility
            Shape must be (batch_size,)
            (positive and negative samples concatenated together)
        m_batch: tf.Tensor
            Batch of context embeddings for each purchased item
            Shape must be (batch_size, latent_size)

        Returns
        -------
        short_term_distance: tf.Tensor
            Distance of all the items in item_batch from their ground truth embedding (X)
            Shape must be (batch_size,)
        """

        X_item_target = tf.gather(self.X, indices=item_batch)  # Shape: (batch_size, d)

        short_term_distance = tf.reduce_sum(tf.square(m_batch - X_item_target), axis=-1)

        return short_term_distance

    def compute_batch_long_distance(
        self,
        item_batch: Union[np.ndarray, tf.Tensor],
        user_batch: np.ndarray = None,
    ) -> tf.Tensor:
        """Compute the long distance of all the items in item_batch given the
        user.

        Parameters
        ----------
        item_batch: np.ndarray or tf.Tensor
            Batch of the purchased items ID (integers) for which to compute the utility
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

        V_batch = tf.cast(self.V, dtype=tf.float32)
        U_batch = tf.cast(self.U, dtype=tf.float32)

        V_future_batch = tf.gather(
            V_batch, indices=item_batch
        )  # Shape: (batch_size, d)

        U_user_batch = tf.gather(U_batch, indices=user_batch)  # Shape: (batch_size, d)

        long_term_distance = tf.reduce_sum(
            tf.square(U_user_batch - V_future_batch), axis=-1
        )  # Shape: (batch_size, 1)

        return long_term_distance

    def compute_batch_distance(
        self,
        item_batch: np.ndarray,
        m_batch: np.ndarray,
        user_batch: np.ndarray,
    ) -> tf.Tensor:
        """Compute the total distance (long + short term) of all the items in
        item_batch."""

        long_distance = self.compute_batch_long_distance(item_batch, user_batch)

        short_distance = self.compute_batch_short_distance(item_batch, m_batch)

        total_distance = (
            self.short_term_ratio * long_distance
            + (1 - self.short_term_ratio) * short_distance
        )

        return total_distance

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

        not_to_be_chosen = tf.concat(
            [purchased_items, tf.expand_dims(next_item, axis=0)], axis=0
        )

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
            Whether the model is in training mode or not, to activate dropout if needed. True by default, cause compute_batch_loss is only used during training.

        Returns
        -------
        tf.Variable
            Value of the loss for the batch (Hinge loss),
            Shape must be (1,)
        _: None
            Placeholder to match the signature of the parent class method
        """

        _ = future_batch  # Unused for this model
        _ = store_batch  # Unused for this model
        _ = week_batch  # Unused for this model
        _ = price_batch  # Unused for this model

        batch_size = len(item_batch)

        negative_samples = tf.reshape(
            tf.transpose(
                tf.stack(
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
                ),
                # Reshape to have at the beginning of the array all the first negative samples
                # of all positive samples, then all the second negative samples, etc.
                # (same logic as for the calls to np.tile)
            ),
            # Flatten 2D --> 1D
            shape=[-1],
        )
        item_batch = tf.cast(item_batch, tf.int32)
        negative_samples = tf.cast(negative_samples, tf.int32)

        augmented_item_batch = tf.cast(
            tf.concat([item_batch, negative_samples], axis=0), dtype=tf.int32
        )

        basket_batch_ragged = tf.cast(
            tf.ragged.boolean_mask(basket_batch, basket_batch != -1),
            dtype=tf.int32,
        )
        basket_batch = basket_batch_ragged.to_tensor(self.n_items)
        m_batch, _ = self.embed_context(basket_batch, is_training)
        # Compute the utility of all the available items
        all_distance = self.compute_batch_distance(
            item_batch=augmented_item_batch,
            m_batch=tf.tile(m_batch, [self.n_negative_samples + 1, 1]),
            user_batch=tf.tile(user_batch, [self.n_negative_samples + 1]),
        )

        positive_samples_distance = tf.gather(all_distance, tf.range(batch_size))

        neg = tf.gather(
            all_distance, tf.range(batch_size, tf.shape(all_distance)[0])
        )  # Shape: (batch_size * n_negative_samples,)

        pos = tf.tile(positive_samples_distance, [self.n_negative_samples])

        ridge_regularization = self.l2_regularization * (
            tf.nn.l2_loss(self.U)
            + tf.nn.l2_loss(self.V)
            + tf.nn.l2_loss(self.X)
            + tf.nn.l2_loss(self.Wq)
            + tf.nn.l2_loss(self.Wk)
        )

        hinge_loss = (
            tf.maximum(float(0), self.hinge_margin + pos - neg) + ridge_regularization
        )  # (batch_size, n_negative_samples)
        total_loss = tf.reduce_sum(hinge_loss)

        # Normalize by the batch size and the number of negative samples
        if tf.reduce_any(self.hinge_margin > 0):
            return (
                total_loss / (batch_size * self.n_negative_samples * self.hinge_margin),
                _,
            )
        else:
            return total_loss / (batch_size * self.n_negative_samples), _

    def evaluate(
        self,
        trip_dataset: TripDataset,
        batch_size: int = 32,
        hit_k: list = [50],
        metrics: list[callable] = None,
    ):
        """Evaluate the model on the given dataset using the specified metric.

        Parameters
        ----------
        hit_k : list
            List of k values for hit rate calculation.
        metrics : list of callable
            List of metric functions to evaluate. Each function should take
            (all_distances, item_batch, hit_k) as parameters and return a score.

        Returns
        -------
        dict
            Dictionary with metric names as keys and their corresponding scores as values.
        """
        inner_range = trip_dataset.iter_batch(
            shuffle=False, batch_size=batch_size, data_method="aleacarta"
        )

        total = 0
        results = {}
        for (
            item_batch,
            basket_batch,
            _,  # future_batch not used here
            _,  # store_batch not used here
            _,  # week_batch not used here
            _,  # price_batch not used here
            available_item_batch,  # available_item_batch not used here
            user_batch,
        ) in inner_range:
            batch_size = tf.shape(item_batch)[0]
            mask = tf.reduce_max(
                tf.one_hot(basket_batch, depth=self.n_items, dtype=tf.int32), axis=1
            )  # Shape: (batch_size, n_items)
            basket_batch_ragged = tf.cast(
                tf.ragged.boolean_mask(basket_batch, basket_batch != -1),
                dtype=tf.int32,
            )
            basket_batch = basket_batch_ragged.to_tensor(self.n_items)
            m_batch, _ = self.embed_context(basket_batch, is_training=False)
            all_distances = self.compute_batch_distance(
                item_batch=tf.tile(np.arange(self.n_items), [batch_size]),
                m_batch=tf.repeat(m_batch, repeats=self.n_items, axis=0),
                user_batch=tf.repeat(user_batch, repeats=self.n_items),
            )  # Shape: (batch_size * n_items,)
            all_distances = tf.reshape(all_distances, (batch_size, self.n_items))

            ####--------------------------------------------------------------
            # We remove the items in each basket from the recommendations in all_distances
            # 1 if item is in the basket, 0 otherwise
            inf_penalty = 100.0
            mask = tf.cast(mask, dtype=tf.float32)
            available_mask = tf.cast(available_item_batch, dtype=tf.float32)

            inf_mask = (
                mask * inf_penalty + (1 - available_mask) * inf_penalty
            )  # Shape: (batch_size, n_items)
            all_distances = all_distances + inf_mask  # Shape: (batch_size, n_items)
            ####----------------------------------------------------------

            total += batch_size
            for metrique_func in metrics:
                nom_metrique = metrique_func.__name__

                score = metrique_func(all_distances, item_batch, hit_k)

                if nom_metrique not in results:
                    results[nom_metrique] = 0.0

                results[nom_metrique] += score

        for metrique_func in metrics:
            nom_metrique = metrique_func.__name__
            results[nom_metrique] = results[nom_metrique] / float(total)
        return results
