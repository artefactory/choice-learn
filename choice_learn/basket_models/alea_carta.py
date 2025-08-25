"""Implementation of the AleaCarta model."""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import tensorflow as tf
import tqdm

from ..tf_ops import softmax_with_availabilities
from .basket_dataset.dataset import Trip, TripDataset


class AleaCarta:
    """Class for the AleaCarta model.

    Better Capturing Interactions between Products in Retail: Revisited Negative Sampling for
    Basket Choice Modeling,
    Désir, J.; Auriau, V.; Možina, M.; Malherbe, E. (2025), ECML PKDDD
    """

    def __init__(
        self,
        item_intercept: bool = True,
        price_effects: bool = False,
        seasonal_effects: bool = False,
        latent_sizes: dict[str] = {"preferences": 4, "price": 4, "season": 4},
        n_negative_samples: int = 2,
        optimizer: str = "adam",
        callbacks: Union[tf.keras.callbacks.CallbackList, None] = None,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 32,
        grad_clip_value: Union[float, None] = None,
        weight_decay: Union[float, None] = None,
        momentum: float = 0.0,
        epsilon_price: float = 1e-5,
    ) -> None:
        """Initialize the Shopper model.

        Parameters
        ----------
        item_intercept: bool, optional
            Whether to include item intercept in the model, by default True
            Corresponds to the item intercept
        price_effects: bool, optional
            Whether to include price effects in the model, by default True
        seasonal_effects: bool, optional
            Whether to include seasonal effects in the model, by default True
        latent_sizes: dict[str]
            Lengths of the vector representation of the latent parameters
            latent_sizes["preferences"]: length of one vector of theta, alpha
            latent_sizes["price"]: length of one vector of gamma, beta
            latent_sizes["season"]: length of one vector of delta, mu
            by default {"preferences": 4, "price": 4, "season": 4}
        n_negative_samples: int, optional
            Number of negative samples to draw for each positive sample for the training,
            by default 2
            Must be > 0
        optimizer: str, optional
            Optimizer to use for training, by default "adam"
        callbacks: tf.keras.callbacks.Callbacklist, optional
            List of callbacks to add to model.fit, by default None and only add History
        lr: float, optional
            Learning rate, by default 1e-3
        epochs: int, optional
            Number of epochs, by default 100
        batch_size: int, optional
            Batch size, by default 32
        grad_clip_value: float, optional
            Value to clip the gradient, by default None
        weight_decay: float, optional
            Weight decay, by default None
        momentum: float, optional
            Momentum for the optimizer, by default 0. For SGD only
        epsilon_price: float, optional
            Epsilon value to add to prices to avoid NaN values (log(0)), by default 1e-5
        """
        self.item_intercept = item_intercept
        self.price_effects = price_effects
        self.seasonal_effects = seasonal_effects

        if "preferences" not in latent_sizes.keys():
            logging.warning(
                "No latent size value has been specified for preferences, "
                "switching to default value 4."
            )
            latent_sizes["preferences"] = 4
        if "price" not in latent_sizes.keys() and self.price_effects:
            logging.warning(
                "No latent size value has been specified for price_effects, "
                "switching to default value 4."
            )
            latent_sizes["price"] = 4
        if "season" not in latent_sizes.keys() and self.seasonal_effects:
            logging.warning(
                "No latent size value has been specified for seasonal_effects, "
                "switching to default value 4."
            )
            latent_sizes["season"] = 4

        for val in latent_sizes.keys():
            if val not in ["preferences", "price", "season"]:
                raise ValueError(f"Unknown value for latent_sizes dict: {val}.")

        if n_negative_samples <= 0:
            raise ValueError("n_negative_samples must be > 0.")

        self.latent_sizes = latent_sizes

        self.n_negative_samples = n_negative_samples

        self.optimizer_name = optimizer
        if optimizer.lower() == "adam":
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr, clipvalue=grad_clip_value, weight_decay=weight_decay
            )
        elif optimizer.lower() == "amsgrad":
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr,
                amsgrad=True,
                clipvalue=grad_clip_value,
                weight_decay=weight_decay,
            )
        elif optimizer.lower() == "adamax":
            self.optimizer = tf.keras.optimizers.Adamax(
                learning_rate=lr, clipvalue=grad_clip_value, weight_decay=weight_decay
            )
        elif optimizer.lower() == "rmsprop":
            self.optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=lr, clipvalue=grad_clip_value, weight_decay=weight_decay
            )
        elif optimizer.lower() == "sgd":
            self.optimizer = tf.keras.optimizers.SGD(
                learning_rate=lr,
                clipvalue=grad_clip_value,
                weight_decay=weight_decay,
                momentum=momentum,
            )
        else:
            print(f"Optimizer {optimizer} not implemented, switching for default Adam")
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr, clipvalue=grad_clip_value, weight_decay=weight_decay
            )

        self.callbacks = tf.keras.callbacks.CallbackList(callbacks, add_history=True, model=None)
        self.callbacks.set_model(self)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.grad_clip_value = grad_clip_value
        self.weight_decay = weight_decay
        self.momentum = momentum
        # Add epsilon to prices to avoid NaN values (log(0))
        self.epsilon_price = epsilon_price

        if len(tf.config.get_visible_devices("GPU")):
            # At least one available GPU
            self.on_gpu = True
        else:
            # No available GPU
            self.on_gpu = False
        # /!\ If a model trained on GPU is loaded on CPU, self.on_gpu must be set
        # to False manually after loading the model, and vice versa

        self.instantiated = False

    def instantiate(
        self,
        n_items: int,
        n_stores: int = 0,
    ) -> None:
        """Instantiate the Shopper model.

        Parameters
        ----------
        n_items: int
            Number of items to consider, i.e. the number of items in the dataset
        n_stores: int
            Number of stores in the population
        """
        self.n_items = n_items
        if n_stores == 0 and self.price_effects:
            # To take into account the price effects, the number of stores must be > 0
            # to have a gamma embedding
            # (By default, the store id is 0)
            n_stores = 1
        self.n_stores = n_stores

        self.alpha = tf.Variable(
            tf.random_normal_initializer(mean=0, stddev=1.0, seed=42)(
                shape=(n_items, self.latent_sizes["preferences"])
            ),  # Dimension for 1 item: latent_sizes["preferences"]
            trainable=True,
            name="alpha",
        )
        self.theta = tf.Variable(
            tf.random_normal_initializer(mean=0, stddev=1.0, seed=42)(
                shape=(n_stores, self.latent_sizes["preferences"])
            ),  # Dimension for 1 item: latent_sizes["preferences"]
            trainable=True,
            name="theta",
        )

        if self.item_intercept:
            # Add item intercept
            self.lambda_ = tf.Variable(
                tf.random_normal_initializer(mean=0, stddev=1.0, seed=42)(
                    shape=(n_items,)  # Dimension for 1 item: 1
                ),
                trainable=True,
                name="lambda",
            )

        if self.price_effects:
            # Add price sensitivity
            self.beta = tf.Variable(
                tf.random_normal_initializer(mean=0, stddev=1.0, seed=42)(
                    shape=(n_items, self.latent_sizes["price"])
                ),  # Dimension for 1 item: latent_sizes["price"]
                trainable=True,
                name="beta",
            )
            self.gamma = tf.Variable(
                tf.random_normal_initializer(mean=0, stddev=1.0, seed=42)(
                    shape=(n_stores, self.latent_sizes["price"])
                ),  # Dimension for 1 item: latent_sizes["price"]
                trainable=True,
                name="gamma",
            )

        if self.seasonal_effects:
            # Add seasonal effects
            self.mu = tf.Variable(
                tf.random_normal_initializer(mean=0, stddev=0.1, seed=42)(
                    shape=(n_items, self.latent_sizes["season"])
                ),  # Dimension for 1 item: latent_sizes["season"]
                trainable=True,
                name="mu",
            )
            self.delta = tf.Variable(
                tf.random_normal_initializer(mean=0, stddev=0.1, seed=42)(
                    shape=(52, self.latent_sizes["season"])
                ),  # Dimension for 1 item: latent_sizes["season"]
                trainable=True,
                name="delta",
            )

        self.instantiated = True

    @property
    def trainable_weights(self) -> list[tf.Variable]:
        """Latent parameters of the model.

        Returns
        -------
        list[tf.Variable]
            Latent parameters of the model
        """
        weights = [self.alpha, self.theta]

        if self.item_intercept:
            weights.append(self.lambda_)

        if self.price_effects:
            weights.extend([self.beta, self.gamma])

        if self.seasonal_effects:
            weights.extend([self.mu, self.delta])

        return weights

    def compute_batch_utility(
        self,
        item_batch: Union[np.ndarray, tf.Tensor],
        basket_batch: np.ndarray,
        store_batch: np.ndarray,
        week_batch: np.ndarray,
        price_batch: np.ndarray,
    ) -> tf.Tensor:
        """Compute the utility of all the items in item_batch.

        Parameters
        ----------
        item_batch: np.ndarray or tf.Tensor
            Batch of the purchased items ID (integers) for which to compute the utility
            Shape must be (batch_size,)
            (positive and negative samples concatenated together)
        basket_batch: np.ndarray
            Batch of baskets (ID of items already in the baskets) (arrays) for each purchased item
            Shape must be (batch_size, max_basket_size)
        store_batch: np.ndarray
            Batch of store IDs (integers) for each purchased item
            Shape must be (batch_size,)
        week_batch: np.ndarray
            Batch of week numbers (integers) for each purchased item
            Shape must be (batch_size,)
        price_batch: np.ndarray
            Batch of prices (floats) for each purchased item
            Shape must be (batch_size,)

        Returns
        -------
        item_utilities: tf.Tensor
            Utility of all the items in item_batch
            Shape must be (batch_size,)
        """
        item_batch = tf.cast(item_batch, dtype=tf.int32)
        basket_batch = tf.cast(basket_batch, dtype=tf.int32)
        store_batch = tf.cast(store_batch, dtype=tf.int32)
        week_batch = tf.cast(week_batch, dtype=tf.int32)
        price_batch = tf.cast(price_batch, dtype=tf.float32)

        theta_store = tf.gather(self.theta, indices=store_batch)
        alpha_item = tf.gather(self.alpha, indices=item_batch)
        # Compute the dot product along the last dimension
        store_preferences = tf.reduce_sum(theta_store * alpha_item, axis=1)

        if self.item_intercept:
            item_intercept = tf.gather(self.lambda_, indices=item_batch)
        else:
            item_intercept = tf.zeros_like(store_preferences)

        if self.price_effects:
            gamma_store = tf.gather(self.gamma, indices=store_batch)
            beta_item = tf.gather(self.beta, indices=item_batch)
            # Add epsilon to avoid NaN values (log(0))
            price_effects = (
                -1
                # Compute the dot product along the last dimension
                * tf.reduce_sum(gamma_store * beta_item, axis=1)
                * tf.math.log(price_batch + self.epsilon_price)
            )
        else:
            gamma_store = tf.zeros_like(store_batch)
            price_effects = tf.zeros_like(store_preferences)

        if self.seasonal_effects:
            delta_week = tf.gather(self.delta, indices=week_batch)
            mu_item = tf.gather(self.mu, indices=item_batch)
            # Compute the dot product along the last dimension
            seasonal_effects = tf.reduce_sum(delta_week * mu_item, axis=1)
        else:
            delta_week = tf.zeros_like(week_batch)
            seasonal_effects = tf.zeros_like(store_preferences)

        # The effects of item intercept, store preferences, price sensitivity
        # and seasonal effects are combined in the per-item per-trip latent variable
        psi = tf.reduce_sum(
            [
                item_intercept,
                store_preferences,
                price_effects,
                seasonal_effects,
            ],
            axis=0,
        )  # Shape: (batch_size,)

        # Create a RaggedTensor from the indices with padding removed
        item_indices_ragged = tf.cast(
            tf.ragged.boolean_mask(basket_batch, basket_batch != -1),
            dtype=tf.int32,
        )

        if tf.size(item_indices_ragged) == 0:
            # Empty baskets: no alpha embeddings to gather
            # (It must be a ragged tensor here because TF's GraphMode requires the same
            # nested structure to be returned from all branches of a conditional)
            alpha_by_basket = tf.RaggedTensor.from_tensor(
                tf.zeros((len(item_batch), 0, self.alpha.shape[1]))
            )
        else:
            # Gather the embeddings using a ragged tensor of indices
            alpha_by_basket = tf.ragged.map_flat_values(tf.gather, self.alpha, item_indices_ragged)
        # Basket interaction: one vs all
        alpha_i = tf.expand_dims(alpha_item, axis=1)  # Shape: (batch_size, 1, latent_size)
        # Compute the dot product along the last dimension (latent_size)
        basket_interaction_utility = tf.reduce_sum(
            alpha_i * alpha_by_basket, axis=-1
        )  # Shape: (batch_size, None)
        # Sum over the items in the basket
        basket_interaction_utility = tf.reduce_sum(
            basket_interaction_utility, axis=-1
        )  # Shape: (batch_size,)

        return psi + basket_interaction_utility

    def compute_basket_utility(
        self,
        basket: Union[None, np.ndarray] = None,
        store: Union[None, int] = None,
        week: Union[None, int] = None,
        prices: Union[None, np.ndarray] = None,
        trip: Union[None, Trip] = None,
    ) -> float:
        """Compute the utility of an (unordered) basket.

        Take as input directly a Trip object or separately basket, store,
        week and prices.

        Parameters
        ----------
        basket: np.ndarray or None, optional
            ID the of items already in the basket, by default None
        store: int or None, optional
            Store id, by default None
        week: int or None, optional
            Week number, by default None
        prices: np.ndarray or None, optional
            Prices for each purchased item, by default None
            Shape must be (len(basket),)
        trip: Trip or None, optional
            Trip object containing basket, store, week and prices,
            by default None

        Returns
        -------
        float
            Utility of the (unordered) basket
        """
        if trip is None:
            # Trip not provided as an argument
            # Then basket, store, week and prices must be provided
            if basket is None or store is None or week is None or prices is None:
                raise ValueError(
                    "If trip is None, then basket, store, week, and prices "
                    "must be provided as arguments."
                )

        else:
            # Trip directly provided as an argument
            basket = trip.purchases
            store = trip.store
            week = trip.week
            prices = [trip.prices[item_id] for item_id in basket]

        len_basket = len(basket)

        # basket_batch[i] = basket without the i-th item
        basket_batch = np.array(
            [np.delete(basket, i) for i in range(len_basket)]
        )  # Shape: (len_basket, len(basket) - 1)

        # Basket utility = sum of the utilities of the items in the basket
        return tf.reduce_sum(
            self.compute_batch_utility(
                item_batch=basket,
                basket_batch=basket_batch,
                store_batch=np.array([store] * len_basket),
                week_batch=np.array([week] * len_basket),
                price_batch=prices,
            )
        ).numpy()

    def compute_item_likelihood(
        self,
        basket: Union[None, np.ndarray] = None,
        available_items: Union[None, np.ndarray] = None,
        store: Union[None, int] = None,
        week: Union[None, int] = None,
        prices: Union[None, np.ndarray] = None,
        trip: Union[None, Trip] = None,
    ) -> tf.Tensor:
        """Compute the likelihood of all items for a given trip.

        Take as input directly a Trip object or separately basket, available_items,
        store, week and prices.

        Parameters
        ----------
        basket: np.ndarray or None, optional
            ID the of items already in the basket, by default None
        available_items: np.ndarray or None, optional
            Matrix indicating the availability (1) or not (0) of the products,
            by default None
            Shape must be (n_items,)
        store: int or None, optional
            Store id, by default None
        week: int or None, optional
            Week number, by default None
        prices: np.ndarray or None, optional
            Prices of all the items in the dataset, by default None
            Shape must be (n_items,)
        trip: Trip or None, optional
            Trip object containing basket, available_items, store,
            week and prices, by default None

        Returns
        -------
        likelihood: tf.Tensor
            Likelihood of all items for a given trip
            Shape must be (n_items,)
        """
        if trip is None:
            # Trip not provided as an argument
            # Then basket, available_items, store, week and prices must be provided
            if (
                basket is None
                or available_items is None
                or store is None
                or week is None
                or prices is None
            ):
                raise ValueError(
                    "If trip is None, then basket, available_items, store, week, and "
                    "prices must be provided as arguments."
                )

        else:
            # Trip directly provided as an argument
            basket = trip.purchases

            if isinstance(trip.assortment, int):
                # Then it is the assortment ID (ie its index in the attribute
                # available_items of the TripDataset), but we do not have the
                # the TripDataset as input here
                raise ValueError(
                    "The assortment ID is not enough to compute the likelihood. "
                    "Please provide the availability matrix directly (array of shape (n_items,) "
                    "indicating the availability (1) or not (0) of the products)."
                )
            # Else: np.ndarray
            available_items = trip.assortment

            store = trip.store
            week = trip.week
            prices = trip.prices

        # Prevent unintended side effects from in-place modifications
        available_items_copy = available_items.copy()
        for basket_item in basket:
            if basket_item != -1:
                available_items_copy[basket_item] = 0.0

        # Compute the utility of all the items
        all_utilities = self.compute_batch_utility(
            # All items
            item_batch=np.array([item_id for item_id in range(self.n_items)]),
            # For each item: same basket / store / week / prices / available items
            basket_batch=np.array([basket for _ in range(self.n_items)]),
            store_batch=np.array([store for _ in range(self.n_items)]),
            week_batch=np.array([week for _ in range(self.n_items)]),
            price_batch=prices,
        )

        # Softmax on the utilities
        return softmax_with_availabilities(
            items_logit_by_choice=all_utilities,  # Shape: (n_items,)
            available_items_by_choice=available_items_copy,  # Shape: (n_items,)
            axis=-1,
            normalize_exit=False,
            eps=None,
        )

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

    # @tf.function  # Graph mode
    def compute_batch_loss(
        self,
        item_batch: np.ndarray,
        basket_batch: np.ndarray,
        store_batch: np.ndarray,
        week_batch: np.ndarray,
        price_batch: np.ndarray,
        available_item_batch: np.ndarray,
    ) -> tuple[tf.Variable]:
        """Compute log-likelihood and loss for one batch of items.

        Parameters
        ----------
        item_batch: np.ndarray
            Batch of purchased items ID (integers)
            Shape must be (batch_size,)
        basket_batch: np.ndarray
            Batch of baskets (ID of items already in the baskets) (arrays) for each purchased item
            Shape must be (batch_size, max_basket_size)
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

        Returns
        -------
        tf.Variable
            Value of the loss for the batch (binary cross-entropy),
            Shape must be (1,)
        loglikelihood: tf.Variable
            Computed log-likelihood of the batch of items
            Approximated by difference of utilities between positive and negative samples
            Shape must be (1,)
        """
        batch_size = len(item_batch)
        item_batch = tf.cast(item_batch, dtype=tf.int32)

        # Negative sampling
        negative_samples = tf.reshape(
            tf.transpose(
                tf.reshape(
                    tf.concat(
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
                    [batch_size, self.n_negative_samples],
                ),
            ),
            # Flatten 2D --> 1D
            shape=[-1],
        )

        augmented_item_batch = tf.cast(
            tf.concat([item_batch, negative_samples], axis=0), dtype=tf.int32
        )
        prices_tiled = tf.tile(price_batch, [self.n_negative_samples + 1, 1])
        # Each time, pick only the price of the item in augmented_item_batch from the
        # corresponding price array
        augmented_price_batch = tf.gather(
            params=prices_tiled,
            indices=augmented_item_batch,
            # batch_dims=1 is equivalent to having an outer loop over
            # the first axis of params and indices
            batch_dims=1,
        )

        # Compute the utility of all the available items
        all_utilities = self.compute_batch_utility(
            item_batch=augmented_item_batch,
            basket_batch=tf.tile(basket_batch, [self.n_negative_samples + 1, 1]),
            store_batch=tf.tile(store_batch, [self.n_negative_samples + 1]),
            week_batch=tf.tile(week_batch, [self.n_negative_samples + 1]),
            price_batch=augmented_price_batch,
        )

        positive_samples_utilities = tf.gather(all_utilities, tf.range(batch_size))
        negative_samples_utilities = tf.gather(
            all_utilities, tf.range(batch_size, tf.shape(all_utilities)[0])
        )

        # Log-likelihood of a batch = sum of log-likelihoods of its samples
        # Add a small epsilon to gain numerical stability (avoid log(0))
        epsilon = 0.0  # No epsilon added for now
        loglikelihood = tf.reduce_sum(
            tf.math.log(
                tf.sigmoid(
                    tf.tile(
                        positive_samples_utilities,
                        [self.n_negative_samples],
                    )
                    - negative_samples_utilities
                )
                + epsilon
            ),
        )  # Shape of loglikelihood: (1,))
        bce = tf.keras.backend.binary_crossentropy(
            # Target: 1 for positive samples, 0 for negative samples
            target=tf.concat(
                [
                    tf.ones_like(positive_samples_utilities),
                    tf.zeros_like(negative_samples_utilities),
                ],
                axis=0,
            ),
            output=tf.nn.sigmoid(all_utilities),
        )  # Shape: (batch_size * (n_negative_samples + 1),)

        # Normalize by the batch size and the number of negative samples
        return tf.reduce_sum(bce) / (batch_size * self.n_negative_samples), loglikelihood

    @tf.function  # Graph mode
    def train_step(
        self,
        item_batch: np.ndarray,
        basket_batch: np.ndarray,
        store_batch: np.ndarray,
        week_batch: np.ndarray,
        price_batch: np.ndarray,
        available_item_batch: np.ndarray,
    ) -> tf.Variable:
        """Train the model for one step.

        Parameters
        ----------
        item_batch: np.ndarray
            Batch of purchased items ID (integers)
            Shape must be (batch_size,)
        basket_batch: np.ndarray
            Batch of baskets (ID of items already in the baskets) (arrays) for each purchased item
            Shape must be (batch_size, max_basket_size)
        store_batch: np.ndarray
            Batch of store ids (integers) for each purchased item
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

        Returns
        -------
        batch_loss: tf.Tensor
            Value of the loss for the batch
        """
        with tf.GradientTape() as tape:
            batch_loss = self.compute_batch_loss(
                item_batch=item_batch,
                basket_batch=basket_batch,
                store_batch=store_batch,
                week_batch=week_batch,
                price_batch=price_batch,
                available_item_batch=available_item_batch,
            )[0]
        grads = tape.gradient(batch_loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return batch_loss

    def fit(
        self,
        trip_dataset: TripDataset,
        val_dataset: Union[TripDataset, None] = None,
        verbose: int = 0,
    ) -> dict:
        """Fit the model to the data in order to estimate the latent parameters.

        Parameters
        ----------
        trip_dataset: TripDataset
            Dataset on which to fit the model
        val_dataset: TripDataset, optional
            Validation dataset, by default None
        verbose: int, optional
            print level, for debugging, by default 0

        Returns
        -------
        history: dict
            Different metrics values over epochs
        """
        if not self.instantiated:
            # Lazy instantiation
            self.instantiate(n_items=trip_dataset.n_items, n_stores=trip_dataset.n_stores)
        batch_size = self.batch_size

        history = {"train_loss": [], "val_loss": []}
        t_range = tqdm.trange(self.epochs, position=0)

        self.callbacks.on_train_begin()

        # Iterate of epochs
        for epoch_nb in t_range:
            self.callbacks.on_epoch_begin(epoch_nb)
            t_start = time.time()
            train_logs = {"train_loss": []}
            val_logs = {"val_loss": []}
            epoch_losses = []

            if verbose > 1:
                inner_range = tqdm.tqdm(
                    trip_dataset.iter_batch(
                        shuffle=True,
                        batch_size=batch_size,
                        data_method="aleacarta",
                    ),
                    total=int(trip_dataset.n_samples / np.max([batch_size, 1])),
                    position=0,
                    leave=False,
                )
            else:
                inner_range = trip_dataset.iter_batch(
                    shuffle=True,
                    batch_size=batch_size,
                    data_method="aleacarta",
                )

            for batch_nb, (
                item_batch,
                basket_batch,
                _,
                store_batch,
                week_batch,
                price_batch,
                available_item_batch,
            ) in enumerate(inner_range):
                # self.callbacks.on_train_batch_begin(batch_nb)

                batch_loss = self.train_step(
                    item_batch=item_batch,
                    basket_batch=basket_batch,
                    store_batch=store_batch,
                    week_batch=week_batch,
                    price_batch=price_batch,
                    available_item_batch=available_item_batch,
                )
                # train_logs["train_loss"].append(batch_loss)
                # temps_logs = {k: tf.reduce_mean(v) for k, v in train_logs.items()}
                # self.callbacks.on_train_batch_end(batch_nb, logs=temps_logs)

                # Optimization Steps
                epoch_losses.append(batch_loss)

                if verbose > 0:
                    inner_range.set_description(
                        f"Epoch Negative-LogLikeliHood: {np.sum(epoch_losses):.4f}"
                    )

            # Take into account the fact that the last batch may have a
            # different length for the computation of the epoch loss.
            if batch_size != -1:
                """last_batch_size = len(item_batch)
                coefficients = tf.concat(
                    [tf.ones(len(epoch_losses) - 1) * batch_size, [last_batch_size]],
                    axis=0,
                )
                epoch_losses = tf.multiply(epoch_losses, coefficients)
                epoch_loss = tf.reduce_sum(epoch_losses) / trip_dataset.n_samples"""
                epoch_loss = tf.reduce_mean(epoch_losses)
            else:
                epoch_loss = tf.reduce_mean(epoch_losses)

            history["train_loss"].append(epoch_loss)
            print_loss = history["train_loss"][-1].numpy()
            desc = f"Epoch {epoch_nb} Train Loss {print_loss:.4f}"
            if verbose > 1:
                print(
                    f"Loop {epoch_nb} Time:",
                    f"{time.time() - t_start:.4f}",
                    f"Loss: {print_loss:.4f}",
                )

            # Test on val_dataset if provided
            if val_dataset is not None:
                val_losses = []
                for batch_nb, (
                    item_batch,
                    basket_batch,
                    _,
                    store_batch,
                    week_batch,
                    price_batch,
                    available_item_batch,
                ) in enumerate(
                    val_dataset.iter_batch(
                        shuffle=True, batch_size=batch_size, data_method="aleacarta"
                    )
                ):
                    self.callbacks.on_batch_begin(batch_nb)
                    self.callbacks.on_test_batch_begin(batch_nb)

                    val_losses.append(
                        self.compute_batch_loss(
                            item_batch=item_batch,
                            basket_batch=basket_batch,
                            store_batch=store_batch,
                            week_batch=week_batch,
                            price_batch=price_batch,
                            available_item_batch=available_item_batch,
                        )[0]
                    )
                    val_logs["val_loss"].append(val_losses[-1])
                    temps_logs = {k: tf.reduce_mean(v) for k, v in val_logs.items()}
                    self.callbacks.on_test_batch_end(batch_nb, logs=temps_logs)

                val_loss = tf.reduce_mean(val_losses)
                if verbose > 1:
                    print("Test Negative-LogLikelihood:", val_loss.numpy())
                    desc += f", Test Loss {np.round(val_loss.numpy(), 4)}"
                history["val_loss"] = history.get("val_loss", []) + [val_loss.numpy()]
                train_logs = {**train_logs, **val_logs}

            temps_logs = {k: tf.reduce_mean(v) for k, v in train_logs.items()}
            self.callbacks.on_epoch_end(epoch_nb, logs=temps_logs)

            t_range.set_description(desc)
            t_range.refresh()

        temps_logs = {k: tf.reduce_mean(v) for k, v in train_logs.items()}
        self.callbacks.on_train_end(logs=temps_logs)
        return history

    def evaluate(
        self,
        trip_dataset: TripDataset,
        batch_size: int = 32,
        epsilon_eval: float = 1e-6,
    ) -> tf.Tensor:
        """Evaluate the model for each trip (unordered basket) in the dataset.

        Predicts the probabilities according to the model and then computes the
        mean negative log-likelihood (nll) for the dataset

        N.B.: Some randomness is involved in the evaluation through random sampling
        of permutations:
        During batch processing, random permutation of the items in the basket
        when creating augmented data from a trip index

        Parameters
        ----------
        trip_dataset: TripDataset
            Dataset on which to apply to prediction
        batch_size: int, optional
            Batch size, by default 32
        epsilon_eval: float, optional
            Small value to avoid log(0) in the computation of the log-likelihood,
            by default 1e-6

        Returns
        -------
        loss: tf.Tensor
            Value of the mean loss (nll) for the dataset,
            Shape must be (1,)
        """
        sum_loglikelihoods = 0.0

        inner_range = trip_dataset.iter_batch(
            shuffle=False, batch_size=batch_size, data_method="aleacarta"
        )
        n_batches = 0
        for (
            item_batch,
            basket_batch,
            _,
            store_batch,
            week_batch,
            price_batch,
            available_item_batch,
        ) in inner_range:
            # Remove padding (-1) from the baskets
            basket_batch_without_padding = [basket[basket != -1] for basket in basket_batch]
            # Sum of the log-likelihoods of all the (unordered) baskets in the batch
            sum_loglikelihoods += np.sum(
                np.log(
                    [
                        self.compute_item_likelihood(
                            basket=basket,
                            available_items=available_items,
                            store=store,
                            week=week,
                            prices=prices,
                        )
                        + epsilon_eval
                        for basket, available_items, store, week, prices in zip(
                            basket_batch_without_padding,
                            available_item_batch,
                            store_batch,
                            week_batch,
                            price_batch,
                        )
                    ]
                )
            )
            n_batches += 1

        # Obliged to recall iter_batch because a generator is exhausted once iterated over
        # or once transformed into a list
        # n_batches = len(list(trip_dataset.iter_batch(
        #     shuffle=False, batch_size=batch_size, data_method="aleacarta"
        #     )
        # ))
        # Total number of samples processed: sum of the batch sizes
        # (last batch may have a different size if incomplete)
        n_elements = batch_size * (n_batches - 1) + len(basket_batch)

        # Predicted mean negative log-likelihood over all the batches
        return -1 * sum_loglikelihoods / n_elements

    def save_model(self, path: str) -> None:
        """Save the different models on disk.

        Parameters
        ----------
        path: str
            path to the folder where to save the model
        """
        if os.path.exists(path):
            # Add current date and time to the folder name
            # if the folder already exists
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            path += f"_{current_time}/"
        else:
            path += "/"

        if not os.path.exists(path):
            Path(path).mkdir(parents=True, exist_ok=True)

        # Save the parameters in a single pickle file
        params = {}
        for k, v in self.__dict__.items():
            # Save only the JSON-serializable parameters
            if isinstance(v, (int, float, list, str, dict)):
                params[k] = v
        json.dump(params, open(os.path.join(path, "params.json"), "w"))

        # Save the latent parameters in separate numpy files
        for latent_parameter in self.trainable_weights:
            parameter_name = latent_parameter.name.split(":")[0]
            np.save(os.path.join(path, parameter_name + ".npy"), latent_parameter)

    @classmethod
    def load_model(cls, path: str) -> "AleaCarta":
        """Load a model previously saved with save_model().

        Parameters
        ----------
        path: str
            path to the folder where the saved model files are

        Returns
        -------
        ChoiceModel
            Loaded ChoiceModel
        """
        # Load parameters
        params = json.load(open(os.path.join(path, "params.json")))

        # Initialize model
        model = cls(
            item_intercept=params["item_intercept"],
            price_effects=params["price_effects"],
            seasonal_effects=params["seasonal_effects"],
            latent_sizes=params["latent_sizes"],
            n_negative_samples=params["n_negative_samples"],
            optimizer=params["optimizer_name"],
            callbacks=params.get("callbacks", None),  # To avoid KeyError if None
            lr=params["lr"],
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            grad_clip_value=params.get("grad_clip_value", None),
            weight_decay=params.get("weight_decay", None),
            momentum=params["momentum"],
            epsilon_price=params["epsilon_price"],
        )

        # Instantiate manually the model
        model.n_items = params["n_items"]
        model.n_stores = params["n_stores"]

        # Fix manually trainable weights values
        model.alpha = tf.Variable(
            np.load(os.path.join(path, "alpha.npy")), trainable=True, name="alpha"
        )
        model.theta = tf.Variable(
            np.load(os.path.join(path, "theta.npy")), trainable=True, name="theta"
        )

        lambda_path = os.path.join(path, "lambda.npy")
        if os.path.exists(lambda_path):
            model.lambda_ = tf.Variable(np.load(lambda_path), trainable=True, name="lambda")

        beta_path = os.path.join(path, "beta.npy")
        if os.path.exists(beta_path):
            # Then the paths to the saved gamma should also exist (price effects)
            model.beta = tf.Variable(np.load(beta_path), trainable=True, name="beta")
            model.gamma = tf.Variable(
                np.load(os.path.join(path, "gamma.npy")), trainable=True, name="gamma"
            )

        mu_path = os.path.join(path, "mu.npy")
        if os.path.exists(mu_path):
            # Then the paths to the saved delta should also exist (price effects)
            model.mu = tf.Variable(np.load(mu_path), trainable=True, name="mu")
            model.delta = tf.Variable(
                np.load(os.path.join(path, "delta.npy")), trainable=True, name="delta"
            )

        model.instantiated = params["instantiated"]

        return model
