"""Implementation of the AleaCarta model."""

import logging
from typing import Union

import numpy as np
import tensorflow as tf

from .base_basket_model import BaseBasketModel
from .data.basket_dataset import Trip, TripDataset


class AleaCarta(BaseBasketModel):
    """Class for the AleaCarta model.

    Better Capturing Interactions between Products in Retail: Revisited Negative Sampling for
    Basket Choice Modeling,
    Désir, J.; Auriau, V.; Možina, M.; Malherbe, E. (2025), ECML PKDDD
    """

    def __init__(
        self,
        item_intercept: bool = True,
        price_effects: bool = True,
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
        l2_regularization: float = 0.0,
        **kwargs,
    ) -> None:
        """Initialize the AleaCarta model.

        Parameters
        ----------
        item_intercept: bool, optional
            Whether to include item intercept in the model, by default True
            Corresponds to the item intercept
        price_effects: bool, optional
            Whether to include price effects in the model, by default True
        seasonal_effects: bool, optional
            Whether to include seasonal effects in the model, by default False
        latent_sizes: dict[str]
            Lengths of the vector representation of the latent parameters
            latent_sizes["preferences"]: length of one vector of theta, gamma
            latent_sizes["price"]: length of one vector of delta, beta
            latent_sizes["season"]: length of one vector of nu, mu
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
        self.l2_regularization = l2_regularization

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

        # Add epsilon to prices to avoid NaN values (log(0))
        self.epsilon_price = epsilon_price

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
            # to have a delta embedding
            # (By default, the store id is 0)
            n_stores = 1
        self.n_stores = n_stores

        self.gamma = tf.Variable(
            tf.random_normal_initializer(mean=0, stddev=0.1, seed=42)(
                shape=(n_items, self.latent_sizes["preferences"])
            ),  # Dimension for 1 item: latent_sizes["preferences"]
            trainable=True,
            name="gamma",
        )
        self.theta = tf.Variable(
            tf.random_normal_initializer(mean=0, stddev=0.1, seed=42)(
                shape=(n_stores, self.latent_sizes["preferences"])
            ),  # Dimension for 1 item: latent_sizes["preferences"]
            trainable=True,
            name="theta",
        )

        if self.item_intercept:
            # Add item intercept
            self.alpha = tf.Variable(
                tf.random_normal_initializer(mean=0, stddev=0.1, seed=42)(
                    shape=(n_items,)  # Dimension for 1 item: 1
                ),
                trainable=True,
                name="alpha",
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

        if self.seasonal_effects:
            # Add seasonal effects
            self.mu = tf.Variable(
                tf.random_normal_initializer(mean=0, stddev=0.1, seed=42)(
                    shape=(n_items, self.latent_sizes["season"])
                ),  # Dimension for 1 item: latent_sizes["season"]
                trainable=True,
                name="mu",
            )
            self.nu = tf.Variable(
                tf.random_normal_initializer(mean=0, stddev=0.1, seed=42)(
                    shape=(52, self.latent_sizes["season"])
                ),  # Dimension for 1 item: latent_sizes["season"]
                trainable=True,
                name="nu",
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
        weights = [self.gamma, self.theta]

        if self.item_intercept:
            weights.append(self.alpha)

        if self.price_effects:
            weights.extend([self.beta, self.delta])

        if self.seasonal_effects:
            weights.extend([self.mu, self.nu])

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

    # @tf.function  # Graph mode
    def compute_preference_utility(
        self,
        item_batch: Union[np.ndarray, tf.Tensor],
        store_batch: np.ndarray,
        week_batch: np.ndarray,
        price_batch: np.ndarray,
    ) -> tf.Tensor:
        """Compute the psi part of the utility of all the items in item_batch.

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
        """
        store_batch = tf.cast(store_batch, dtype=tf.int32)
        week_batch = tf.cast(week_batch, dtype=tf.int32)
        price_batch = tf.cast(price_batch, dtype=tf.float32)

        theta_store = tf.gather(self.theta, indices=store_batch)
        gamma_item = tf.gather(self.gamma, indices=item_batch)
        # Compute the dot product along the last dimension
        store_preferences = tf.reduce_sum(theta_store * gamma_item, axis=1)

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
                * tf.reduce_sum(delta_store * beta_item, axis=1)
                * tf.math.log(price_batch + self.epsilon_price)
            )
        else:
            delta_store = tf.zeros_like(store_batch)
            price_effects = tf.zeros_like(store_preferences)

        if self.seasonal_effects:
            nu_week = tf.gather(self.nu, indices=week_batch)
            mu_item = tf.gather(self.mu, indices=item_batch)
            # Compute the dot product along the last dimension
            seasonal_effects = tf.reduce_sum(nu_week * mu_item, axis=1)
        else:
            nu_week = tf.zeros_like(week_batch)
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
        return psi

    def embed_basket(
        self,
        basket_batch: np.ndarray,
    ) -> tf.Tensor:
        """Compute the embedding of all the baskets in basket_batch.

        Parameters
        ----------
        basket_batch: np.ndarray
            Batch of baskets (ID of items already in the baskets) (arrays) for each purchased item
            Shape must be (batch_size, max_basket_size)

        Returns
        -------
        tf.Tensor
            Embedding of all the baskets in basket_batch
            Shape must be (batch_size, latent_size)
        """
        basket_batch = tf.cast(basket_batch, dtype=tf.int32)
        # Create a RaggedTensor from the indices with padding removed
        item_indices_ragged = tf.cast(
            tf.ragged.boolean_mask(basket_batch, basket_batch != -1),
            dtype=tf.int32,
        )
        if tf.size(item_indices_ragged) == 0:
            # Empty baskets: no gamma embeddings to gather
            # (It must be a ragged tensor here because TF's GraphMode requires the same
            # nested structure to be returned from all branches of a conditional)
            gamma_by_basket = tf.RaggedTensor.from_tensor(
                tf.zeros((len(basket_batch), 0, self.gamma.shape[1]))
            )
        else:
            # Gather the embeddings using a ragged tensor of indices and then sum them in each basket
            gamma_by_basket = tf.ragged.map_flat_values(tf.gather, self.gamma, item_indices_ragged)

        gamma_by_basket = tf.reduce_mean(gamma_by_basket, axis=1)

        has_nan_row = tf.reduce_any(tf.math.is_nan(gamma_by_basket), axis=1)
        condition_mask = tf.expand_dims(has_nan_row, axis=1)
        zeros = tf.zeros_like(gamma_by_basket)
        gamma_by_basket = tf.where(
            condition_mask, zeros, gamma_by_basket
        )  # Shape: (batch_size, latent_size)

        return gamma_by_basket

    # @tf.function  # Graph mode
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
        """


        preference_utility = self.compute_preference_utility(
            item_batch=item_batch,
            store_batch=store_batch,
            week_batch=week_batch,
            price_batch=price_batch,
        )  # Shape: (batch_size,)

        # Basket utility = sum of the utilities of the items in the basket
        gamma_by_basket = self.embed_basket(basket_batch=basket_batch)

        interaction_utility = self.compute_interaction_utility(
            item_batch=item_batch,
            gamma_by_basket=gamma_by_basket,
        )  # Shape: (batch_size,)

        return interaction_utility + preference_utility

    def compute_interaction_utility(
        self,
        item_batch: Union[np.ndarray, tf.Tensor],
        gamma_by_basket: np.ndarray,
    ) -> tf.Tensor:
        """Compute the utility of all the items in item_batch given the items in basket_batch.

        Parameters
        ----------
        item_batch: np.ndarray or tf.Tensor
            Batch of the purchased items ID (integers) for which to compute the utility
            Shape must be (batch_size,)
            (positive and negative samples concatenated together)
        gamma_by_basket: np.ndarray
            Embedding of all the baskets in basket_batch
        Returns
        -------
        basket_interaction_utility: tf.Tensor
            Interaction utility of all the items in item_batch
            Shape must be (batch_size,)
        """

        item_batch = tf.cast(item_batch, dtype=tf.int32)
        gamma_item = tf.gather(self.gamma, indices=item_batch)  # Shape: (batch_size, latent_size)

        # Basket interaction: one vs all
        # Compute the dot product along the last dimension (latent_size)
        basket_interaction_utility = tf.reduce_sum(
            gamma_by_basket * gamma_item, axis=1
        )  # Shape: (batch_size,)

        return basket_interaction_utility

    def compute_basket_utility(
        self,
        basket: Union[None, np.ndarray] = None,
        store: Union[None, int] = None,
        week: Union[None, int] = None,
        prices: Union[None, np.ndarray] = None,
        available_item_batch: Union[None, np.ndarray] = None,
        trip: Union[None, Trip] = None,
    ) -> float:
        r"""Compute the utility of an (unordered) basket.

        Corresponds to the sum of all the conditional utilities:
                \sum_{i \in basket} U(i | basket \ {i})
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
            available_item_batch = trip.assortment
            prices = [trip.prices[item_id] for item_id in basket]

        len_basket = len(basket)

        # basket_batch[i] = basket without the i-th item
        basket_batch = np.array(
            [np.delete(basket, i) for i in range(len_basket)]
        )  # Shape: (len_basket, len(basket) - 1)

        return tf.reduce_sum(
            self.compute_batch_utility(
                item_batch=basket,
                basket_batch=basket_batch,
                store_batch=np.array([store] * len_basket),
                week_batch=np.array([week] * len_basket),
                price_batch=prices,
            )
        ).numpy()

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
        future_batch: np.ndarray,
        store_batch: np.ndarray,
        week_batch: np.ndarray,
        price_batch: np.ndarray,
        available_item_batch: np.ndarray,
        user_batch: np.ndarray,
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
        _ = future_batch
        batch_size = len(item_batch)
        item_batch = tf.cast(item_batch, dtype=tf.int32)

        # Negative sampling
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
        )  # Shape: (batch_size * (n_negative_samples + 1),)

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
        ridge_regularization = self.l2_regularization * (
            tf.nn.l2_loss(self.gamma) + tf.nn.l2_loss(self.theta)
        )
        # Normalize by the batch size and the number of negative samples
        return tf.reduce_sum(bce + ridge_regularization) / (batch_size), loglikelihood

    # @tf.function  # Graph mode
    def evaluate(
        self,
        trip_dataset: TripDataset,
        batch_size: int = 32,
        hit_k: list = [50],
        metrics: list[callable] = None,  # Change *metrics to a named parameter
    ):
        """Evaluate the model on the given dataset using the specified metric."""
        inner_range = trip_dataset.iter_batch(
            shuffle=False, batch_size=batch_size, data_method="aleacarta"
        )

        total = 0
        results = {}

        intercept = self.compute_preference_utility(
            item_batch=tf.tile(np.arange(self.n_items), [batch_size]),
            store_batch=np.tile([0] * batch_size, [self.n_items]),
            week_batch=np.tile([0] * batch_size, [self.n_items]),
            price_batch=np.tile([1] * batch_size, [self.n_items]),
        )
        for (
            item_batch,
            basket_batch,
            _,  # future_batch not used here
            store_batch,  # store_batch not used here
            week_batch,  # week_batch not used here
            price_batch,  # price_batch not used here
            available_item_batch,  # available_item_batch not used here
            _,
        ) in inner_range:
            batch_size = tf.shape(item_batch)[0]
            mask = tf.reduce_max(
                tf.one_hot(basket_batch, depth=self.n_items, dtype=tf.int32), axis=1
            )  # Shape: (batch_size, n_items)
            gamma_by_basket = self.embed_basket(basket_batch=basket_batch)

            all_distances = (
                self.compute_interaction_utility(
                    item_batch=tf.tile(np.arange(self.n_items), [batch_size]),
                    gamma_by_basket=tf.repeat(gamma_by_basket, repeats=self.n_items, axis=0),
                )
                + intercept[: batch_size * self.n_items]
            )
            # Shape: (batch_size * n_items,)

            all_distances = tf.reshape(all_distances, (batch_size, self.n_items))

            ####--------------------------------------------------------------
            # We remove the items in each basket from the recommendations in all_distances
            # 1 if item is in the basket, 0 otherwise
            inf_penalty = 100.0
            mask = tf.cast(mask, dtype=tf.float32)

            inf_mask = mask * inf_penalty  # Shape: (batch_size, n_items)
            all_distances = all_distances - inf_mask  # Shape: (batch_size, n_items)
            ####----------------------------------------------------------
            total += batch_size
            for metrique_func in metrics:
                nom_metrique = metrique_func.__name__

                score = metrique_func(-all_distances, item_batch, hit_k)

                if nom_metrique not in results:
                    results[nom_metrique] = 0.0

                results[nom_metrique] += score
        for metrique_func in metrics:
            nom_metrique = metrique_func.__name__
            results[nom_metrique] = results[nom_metrique] / float(total)
        return results
