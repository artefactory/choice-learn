"""Implementation of SABAC model."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath("submodules/choice-learn"))
from choice_learn.basket_models.base_basket_model import BaseBasketModel


class SABAC(BaseBasketModel):
    """Class for SABAC model.

    -Self Attention for BAsket Completion- an attention-based model for basket completion
    """

    def __init__(
        self,
        latent_sizes: dict[str, int] = {"short_term": 10, "long_term": 0, "price": 0},
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
        item_intercept: bool = True,
        price_effects: bool = False,
        store_effects: bool = False,
        epsilon_price: float = 1e-4,
        value_matrix: bool = False,
        num_layers: int = 1,
        num_heads: int = 1,
        cls_architecture: bool = False,
        attention_pooling: bool = False,
        rc_ln: bool = True,
        loss: str = "bce",
        **kwargs,
    ) -> None:
        """Initialize the model with hyperparameters.

        Parameters
        ----------
        latent_size : latent_sizes : dict[str, int]
            Dictionary specifying the sizes of different latent dimensions:
            'short_term' for short-term preferences, 'long_term' for long-term preferences,
            and 'price' for price effects.
        short_term_ratio : float
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
        store_effects: bool, optional
            Whether to include store effects in the model, by default True
        epsilon_price: float, optional
            Epsilon value to add to prices to avoid NaN values (log(0)), by default 1e-4
        value_matrix: bool, optional
            Whether to use a value matrix in the attention mechanism, by default False
        num_layers: int, optional
            Number of transformer layers, by default 1
        num_heads: int, optional
            Number of attention heads in the transformer layers, by default 1
        cls_architecture: bool, optional
            Whether to use a CLS token architecture (instead of pooling)
            for the basket representation, by default False
        attention_pooling: bool, optional
            Whether to use attention pooling (instead of mean pooling)
            for the basket representation, by default False
        rc_ln: bool, optional
            Whether to use the Residual_Connexion and Layer_Normalization architecture in the
            transformer blocks, by default True
        loss: str, optional
            Loss function to use, either 'bce' for binary cross-entropy or 'scce' for
            sparse categorical cross-entropy, by default 'bce'

        """
        self.instantiated = False

        for val in latent_sizes.keys():
            if val not in ["short_term", "long_term", "price"]:
                raise ValueError(f"Unknown value for latent_sizes dict: {val}.")
        if "short_term" not in latent_sizes:
            latent_sizes["short_term"] = 10
        if "long_term" not in latent_sizes:
            latent_sizes["long_term"] = 0
        if "price" not in latent_sizes:
            latent_sizes["price"] = 0

        self.short_term_ratio = short_term_ratio
        self.n_negative_samples = n_negative_samples

        self.latent_sizes = latent_sizes
        self.d = self.latent_sizes["short_term"]
        self.d_long = self.latent_sizes["long_term"]
        self.ffn_hidden_dim = self.d * 4
        self.l2_regularization = l2_regularization
        self.dropout_rate = dropout_rate
        self.item_intercept = item_intercept
        self.price_effects = price_effects
        self.store_effects = store_effects
        self.epsilon_price = epsilon_price
        self.value_matrix = value_matrix
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.cls_architecture = cls_architecture
        self.attention_pooling = attention_pooling
        self.rc_ln = rc_ln
        self.loss = loss
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
        theta : tf.Variable, optional
            Store effects embedding matrix, size (n_stores, d).
        beta : tf.Variable, optional
            Item price sensitivity embedding matrix, size (n_items, latent_sizes["price"]).
        delta : tf.Variable, optional
            Store price sensitivity embedding matrix, size (n_stores, latent_sizes["price"]).
        alpha : tf.Variable, optional
            Item intercept vector, size (n_items,).
        CLS_token : tf.Variable, optional
            CLS token embedding, size (1, d).
        W_Q, W_K, W_V, W_O, W1, W2, b1, b2, gamma1, beta1, gamma2, beta2, S : tf.Variable, optional
            Weights and biases for TransformerBlocks and attention pooling.
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
        if self.cls_architecture:
            self.CLS_token = tf.Variable(
                tf.random_normal_initializer(mean=0, stddev=0.01, seed=42)(shape=(1, self.d)),
                trainable=True,
                name="CLS_token",
            )

        self.blocks = []
        for i in range(self.num_layers):
            block = TransformerBlock(
                d_model=self.d,
                d_ffn=self.ffn_hidden_dim,
                num_heads=self.num_heads,
                use_value_matrix=self.value_matrix,
                dropout_rate=self.dropout_rate,
                attention_pooling=False,
                rc_ln=self.rc_ln,
                name=f"block_{i}",
            )
            self.blocks.append(block)
        if not self.attention_pooling:
            print("Using mean pooling for basket representation.")
        else:
            block = TransformerBlock(
                d_model=self.d,
                d_ffn=self.ffn_hidden_dim,
                num_heads=1,
                use_value_matrix=self.value_matrix,
                dropout_rate=self.dropout_rate,
                attention_pooling=True,
                name="attention_pooling_block",
            )
            self.blocks.append(block)
        self.instantiated = True

    @property
    def trainable_weights(self):
        """Return the trainable weights of the model.

        Returns
        -------
            list
                List of trainable weights (X, V, U, alpha, CLS_token, beta, delta,
                and TransformerBlock weights)
        """
        weights = [self.X, self.V, self.U]
        if self.item_intercept:
            weights.extend([self.alpha])
        if self.cls_architecture:
            weights.extend([self.CLS_token])
        if self.price_effects:
            weights.extend([self.beta, self.delta])
        for block in self.blocks:
            weights.extend(block.get_weights())
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

    def embed_basket(
        self, basket_batch: tf.Tensor, is_training: bool = False
    ) -> tuple[tf.Tensor, list[tf.Tensor]]:
        """Compute the embedding of the baskets in basket_batch.

        Parameters
        ----------
        basket_batch: tf.Tensor
            Batch of baskets (integers) for which to compute the embedding.
            Shape must be (batch_size, L) where L is the number of items in the bigger basket.
        is_training: bool
            Whether the model is in training mode.

        Returns
        -------
        tf.Tensor
            Embedding of each basket in basket_batch.
            Shape must be (batch_size, d).
        list[tf.Tensor]
            Attention weights for each block in the model.
            Shape of each tensor in the list must be (batch_size, num_heads, L, L)
            for blocks with attention_pooling=False,
        """
        padding_vector = tf.zeros(shape=[1, self.d])
        padded_items = tf.concat([self.X, padding_vector], axis=0)
        x = tf.gather(padded_items, indices=basket_batch)  # Shape: (batch_size, L, d)

        mask = tf.not_equal(basket_batch, self.n_items)
        mask_float = tf.cast(mask, dtype=tf.float32)

        mask_float = mask_float[:, tf.newaxis, tf.newaxis, :]
        attention_mask = 1.0 - mask_float  # Shape: (batch_size, 1, 1, L)
        num_items_by_basket = tf.reduce_sum(
            tf.expand_dims(mask_float, axis=-1), axis=1
        )  # (batch_size, 1)

        if self.cls_architecture:
            batch_size = tf.shape(x)[0]
            cls_tokens = tf.repeat(
                self.CLS_token, repeats=batch_size, axis=0
            )  # Shape: (batch_size, d)
            x = tf.concat([cls_tokens[:, tf.newaxis, :], x], axis=1)  # Shape: (batch_size, L+1, d)
            attention_mask = tf.concat(
                [tf.zeros((batch_size, 1, 1, 1)), attention_mask], axis=-1
            )  #
            num_items_by_basket += 1

        attention_weights = []
        for block in self.blocks:
            x, attention_weight = block.call(x, mask=attention_mask, training=is_training)
            attention_weights.append(attention_weight)
        if self.cls_architecture:
            basket_embedding = x[:, 0, :]  # Shape: (batch_size, d)
        elif self.attention_pooling:
            basket_embedding = x  # Shape: (batch_size, 1, d)
            basket_embedding = tf.squeeze(basket_embedding, axis=1)  # Shape: (batch_size, d)
        else:
            basket_embedding = tf.math.divide_no_nan(
                tf.reduce_sum(x, axis=1),
                tf.squeeze(tf.reduce_sum(num_items_by_basket, axis=2), axis=-1),
            )  # Shape: (batch_size, d)

        basket_embedding = tf.nn.l2_normalize(basket_embedding, axis=1)
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
            Shape must be (batch_size,)
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
        x_item_target = tf.gather(self.X, indices=item_batch)  # Shape: (batch_size, d)

        return tf.reduce_sum(
            tf.expand_dims(basket_embedding, axis=1) * x_item_target, axis=-1
        )  # Shape: (batch_size,)

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

    def get_negative_samples_vectorized(
        self, item_batch, basket_batch, available_item_batch, n_samples
    ):
        """Vectorized version of get_negative_samples, to speed up the training."""
        batch_size = tf.shape(item_batch)[0]

        # Mask for target items
        target_mask = tf.one_hot(
            item_batch, depth=self.n_items, dtype=tf.float32
        )  # Shape: (batch_size, n_items)

        # Mask for items in the basket
        basket_one_hot = tf.one_hot(
            basket_batch, depth=self.n_items, dtype=tf.float32
        )  # Shape: (batch_size, max_basket_size, n_items)
        history_mask = tf.reduce_max(basket_one_hot, axis=1)  # Shape: (batch_size, n_items)

        # Mask for unavailable items
        unavailable_mask = 1.0 - tf.cast(available_item_batch, tf.float32)

        forbidden_matrix = target_mask + history_mask + unavailable_mask
        forbidden_mask = tf.cast(forbidden_matrix > 0, tf.float32)

        random_scores = tf.random.uniform(shape=[batch_size, self.n_items])

        final_scores = random_scores - (forbidden_mask * 1e9)

        _, negative_samples = tf.math.top_k(final_scores, k=n_samples)

        return tf.cast(negative_samples, tf.int32)

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
            store_preferences = tf.zeros_like(item_batch, dtype=tf.float32)

        if self.item_intercept:
            item_intercept = tf.gather(self.alpha, indices=item_batch)
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
        )

    @tf.function  # Graph mode
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
            Value of the loss for the batch,
            Shape must be (1,)
        _: None
            Placeholder to match the signature of the parent class method
        """
        _ = future_batch  # Unused for this model
        batch_size = len(item_batch)

        # --- NOUVEAU CODE (VECTORISÉ) ---
        negative_samples = self.get_negative_samples_vectorized(
            item_batch=item_batch,
            basket_batch=basket_batch,
            available_item_batch=available_item_batch,
            n_samples=self.n_negative_samples,
        )  # Shape: (batch_size, n_negative_samples)
        # ----------------------------------------------

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
        )
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

        # We don't use popularity bias (alpha) and layer norm parameters in the l2 regularization.
        ridge_regularization = self.l2_regularization * tf.add_n(
            [
                tf.nn.l2_loss(weight)
                for weight in self.trainable_weights
                if "alpha" not in weight.name and "layer_normalization" not in weight.name
            ]
        )

        epsilon = 1e-8
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
        if self.loss == "bce":
            loss = tf.keras.backend.binary_crossentropy(
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
        # --------------------------------------------------------------
        elif self.loss == "scce":
            labels = tf.zeros((batch_size,), dtype=tf.int32)

            loss = tf.keras.losses.sparse_categorical_crossentropy(
                y_true=labels, y_pred=all_utilities, from_logits=True
            )
        # --------------------------------------------------------------
        return tf.reduce_sum(loss + ridge_regularization) / (
            batch_size * (self.n_negative_samples + 1)
        ), loglikelihood

    def save_model(self, path: str) -> None:
        """
        Override of save_model to handle serialization of self.blocks.

        This method saves the weights of the TransformerBlock objects in separate .npy files
        and their configurations in the params.json file, while ensuring that the original
        self.blocks list is not modified.
        """
        if os.path.exists(path):
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"{path}_{current_time}"

        Path(path).mkdir(parents=True, exist_ok=True)

        weights_to_save = self.trainable_weights
        for latent_parameter in weights_to_save:
            parameter_name = latent_parameter.name.split(":")[0]
            np.save(os.path.join(path, parameter_name + ".npy"), latent_parameter.numpy())

        original_blocks = self.blocks
        try:
            self.blocks = [block.to_dict() for block in self.blocks]

            params_to_save = {}
            for key, value in self.__dict__.items():
                try:
                    json.dumps(value)
                    params_to_save[key] = value
                except TypeError:
                    pass

            with open(os.path.join(path, "params.json"), "w") as f:
                json.dump(params_to_save, f, indent=4)

        finally:
            self.blocks = original_blocks

    @classmethod
    def load_model(cls, path: str):
        """
        Override of load_model to handle deserialization of self.blocks.

        This method reconstructs the TransformerBlock objects from their dictionary representations
        and loads their weights from the saved .npy files.
        """
        model = super().load_model(path)

        # Reconstruit les objets TransformerBlock
        reconstructed_blocks = []
        if hasattr(model, "blocks") and model.blocks:
            for block_params in model.blocks:
                # Crée une nouvelle instance de TransformerBlock pour chaque dict de paramètres
                reconstructed_blocks.append(TransformerBlock(**block_params))

        model.blocks = reconstructed_blocks
        for block in model.blocks:
            for var in block.get_weights():
                parameter_name = var.name.split(":")[0]
                weight_path = os.path.join(path, parameter_name + ".npy")
                if os.path.exists(weight_path):
                    loaded_weight = np.load(weight_path)
                    var.assign(loaded_weight)
                else:
                    raise FileNotFoundError(
                        f"Weight file {weight_path} not found during model loading."
                    )
        model.instantiated = True

        return model


class TransformerBlock:
    """Transformer block.

    Implements a single transformer block with multi-head self-attention and
    feed-forward network or attention pooling.
    """

    def __init__(
        self,
        d_model,
        d_ffn,
        num_heads,
        use_value_matrix=True,
        dropout_rate=0.1,
        attention_pooling=False,
        rc_ln=True,
        name="transformer_block",
    ):
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_value_matrix = use_value_matrix
        self.dropout_rate = dropout_rate
        self.attention_pooling = attention_pooling
        self._trainable_weights = []
        self.rc_ln = rc_ln
        self.name = name

        def add_var(shape, var_name, zeros=False):
            full_name = f"{self.name}_{var_name}"
            init = tf.zeros_initializer() if zeros else tf.initializers.GlorotUniform()
            var = tf.Variable(init(shape=shape), name=full_name, trainable=True)
            self._trainable_weights.append(var)
            return var

        if self.attention_pooling:
            # For attention pooling, we use cross-attention with 1 head instead of self-attention
            self.num_heads = 1
            self.head_dim = d_model
            self.S = add_var((1, d_model), "W_Q")
        else:
            self.W_Q = add_var((d_model, d_model), "W_Q")
            self.W1 = add_var((d_model, d_ffn), "ffn_W1")
            self.b1 = add_var((d_ffn,), "ffn_b1", zeros=True)
            self.gamma1 = add_var((d_model,), "ln1_gamma")
            self.beta1 = add_var((d_model,), "ln1_beta", zeros=True)
            self.gamma2 = add_var((d_model,), "ln2_gamma")
            self.beta2 = add_var((d_model,), "ln2_beta", zeros=True)
            self.W2 = add_var((d_ffn, d_model), "ffn_W2")
            self.b2 = add_var((d_model,), "ffn_b2", zeros=True)
            self.W_O = add_var((d_model, d_model), "W_O") if num_heads > 1 else None
        self.W_K = add_var((d_model, d_model), "W_K")
        self.W_V = add_var((d_model, d_model), "W_V") if use_value_matrix else None

    def get_weights(self):
        """Return the trainable weights of the transformer block."""
        return self._trainable_weights

    def call(self, x, mask=None, training=True):
        """Apply the transformer block/attention pooling to the input tensor x."""
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        residual = x

        if self.attention_pooling:
            q = self.S  # Shape: (1, d_model)
            q = tf.tile(q, [batch_size, 1])  # Shape: (batch_size, d_model)
            q = tf.expand_dims(q, axis=1)  # Shape: (batch_size, 1, d_model)
            k = x
        else:
            q = tf.matmul(x, self.W_Q)  # Shape: (batch_size, seq_len, d_model)
            k = tf.matmul(x, self.W_K)  # Shape: (batch_size, seq_len, d_model)
        q_len = tf.shape(q)[1]
        v = tf.matmul(x, self.W_V) if self.use_value_matrix else x

        def split_heads(tensor, seq_len=seq_len):
            tensor = tf.reshape(tensor, (batch_size, seq_len, self.num_heads, self.head_dim))
            return tf.transpose(tensor, perm=[0, 2, 1, 3])

        q_h = split_heads(q, seq_len=q_len)
        k_h = split_heads(k)
        v_h = split_heads(v)

        matmul_qk = tf.matmul(q_h, k_h, transpose_b=True)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(
            tf.cast(self.head_dim, tf.float32)
        )  # Shape: (batch_size, num_heads, seq_len, seq_len)
        if mask is not None:
            scaled_attention_logits += mask * -1e9

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        if training:
            attention_weights = tf.nn.dropout(attention_weights, rate=self.dropout_rate)

        output = tf.matmul(attention_weights, v_h)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, q_len, self.d_model))
        if self.num_heads > 1:
            output = tf.matmul(output, self.W_O)

        if self.attention_pooling:
            x = output
        else:
            x = (
                self._layer_norm(residual + output, self.gamma1, self.beta1)
                if self.rc_ln
                else output
            )
            residual = x
            ffn_out = tf.matmul(x, self.W1) + self.b1
            ffn_out = tf.nn.gelu(ffn_out)
            if training:
                ffn_out = tf.nn.dropout(ffn_out, rate=self.dropout_rate / 4)
            ffn_out = tf.matmul(ffn_out, self.W2) + self.b2

            x = (
                self._layer_norm(residual + ffn_out, self.gamma2, self.beta2)
                if self.rc_ln
                else ffn_out
            )
        return x, attention_weights

    def _layer_norm(self, x, gamma, beta, epsilon=1e-6):
        # x Shape: (batch_size, seq_len, d_model)
        mean, variance = tf.nn.moments(
            x, axes=[-1], keepdims=True
        )  # Shape mean et variance: (batch_size, seq_len, 1)
        return gamma * (x - mean) / tf.sqrt(variance + epsilon) + beta

    def to_dict(self):
        """Convert the TransformerBlock instance to a dictionary for serialization."""
        return {
            "d_model": self.d_model,
            "d_ffn": self.d_ffn,
            "num_heads": self.num_heads,
            "use_value_matrix": self.use_value_matrix,
            "dropout_rate": self.dropout_rate,
            "attention_pooling": self.attention_pooling,
            "rc_ln": self.rc_ln,
            "name": self.name,
        }
