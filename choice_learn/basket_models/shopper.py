"""Implementation of the Shopper model."""

import logging
from typing import Union

import numpy as np
import tensorflow as tf

from .base_basket_model import BaseBasketModel


class Shopper(BaseBasketModel):
    """Class for the Shopper model.

    SHOPPER: A Probabilistic Model of Consumer Choice with Substitutes and Complements,
    Ruiz, F. J. R.; Athey, S.; Blei, D. M. (2019)
    """

    def __init__(
        self,
        item_intercept: bool = True,
        price_effects: bool = False,
        seasonal_effects: bool = False,
        think_ahead: bool = False,
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
        **kwargs,
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
        think_ahead: bool, optional
            Whether to include "thinking ahead" in the model, by default False
        latent_sizes: dict[str]
            Lengths of the vector representation of the latent parameters
            latent_sizes["preferences"]: length of one vector of theta, alpha, rho
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
        self.think_ahead = think_ahead

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
            (includes the checkout item)
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

        self.rho = tf.Variable(
            tf.random_normal_initializer(mean=0, stddev=1.0, seed=42)(
                shape=(n_items, self.latent_sizes["preferences"])
            ),  # Dimension for 1 item: latent_sizes["preferences"]
            trainable=True,
            name="rho",
        )
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
                    # No lambda for the checkout item (set to 0 later)
                    shape=(n_items - 1,)  # Dimension for 1 item: 1
                ),
                trainable=True,
                name="lambda_",
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
        weights = [self.rho, self.alpha, self.theta]

        if self.item_intercept:
            weights.append(self.lambda_)

        if self.price_effects:
            weights.extend([self.beta, self.gamma])

        if self.seasonal_effects:
            weights.extend([self.mu, self.delta])

        return weights

    @property
    def train_iter_method(self):
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
        return "shopper"

    def thinking_ahead(
        self,
        item_batch: Union[np.ndarray, tf.Tensor],
        ragged_basket_batch: tf.RaggedTensor,
        price_batch: np.ndarray,
        available_item_batch: np.ndarray,
        theta_store: tf.Tensor,
        gamma_store: tf.Tensor,
        delta_week: tf.Tensor,
    ) -> tf.Tensor:
        """Compute the utility of all the items in item_batch.

        Parameters
        ----------
        item_batch: np.ndarray or tf.Tensor
            Batch of the purchased items ID (integers) for which to compute the utility
            Shape must be (batch_size,)
            (positive and negative samples concatenated together)
        ragged_basket_batch: tf.RaggedTensor
            Batch of baskets (ID of items already in the baskets) (arrays) without padding
            for each purchased item
            Shape must be (batch_size, None)
        price_batch: np.ndarray
            Batch of prices (integers) for each purchased item
            Shape must be (batch_size,)
        available_item_batch: np.ndarray
            Batch of availability matrices (indicating the availability (1) or not (0)
            of the products) (arrays) for each purchased item
            Shape must be (batch_size, n_items)
        theta_store: tf.Tensor
            Slices from theta embedding gathered according to the indices that correspond
            to the store of each purchased item in the batch
            Shape must be (batch_size, latent_sizes["preferences"])
        gamma_store: tf.Tensor
            Slices from gamma embedding gathered according to the indices that correspond
            to the store of each purchased item in the batch
            Shape must be (batch_size, latent_sizes["price"])
        delta_week: tf.Tensor
            Slices from delta embedding gathered according to the indices that correspond
            to the week of each purchased item in the batch
            Shape must be (batch_size, latent_sizes["season"])

        Returns
        -------
        total_next_step_utilities: tf.Tensor
            Nex step utility of all the items in item_batch
            Shape must be (batch_size,)
        """
        total_next_step_utilities = tf.zeros_like(item_batch, dtype=tf.float32)
        # Compute the next step item utility for each element of the batch, one by one
        # TODO: avoid a for loop on ragged_basket_batch at a later stage
        for idx in tf.range(ragged_basket_batch.shape[0]):
            basket = tf.gather(ragged_basket_batch, idx)
            if len(basket) != 0 and basket[-1] == 0:
                # No thinking ahead when the basket ends already with the checkout item 0
                total_next_step_utilities = tf.tensor_scatter_nd_update(
                    tensor=total_next_step_utilities, indices=[[idx]], updates=[0]
                )

            else:
                # Basket with the hypothetical current item
                next_basket = tf.concat([basket, [item_batch[idx]]], axis=0)
                # Get the list of available items based on the availability matrix
                item_ids = tf.range(self.n_items)
                available_mask = tf.equal(available_item_batch[idx], 1)
                assortment = tf.boolean_mask(item_ids, available_mask)
                hypothetical_next_purchases = tf.boolean_mask(
                    assortment,
                    ~tf.reduce_any(
                        tf.equal(tf.expand_dims(assortment, axis=1), next_basket), axis=1
                    ),
                )
                # Check if there are still items to purchase during the next step
                if len(hypothetical_next_purchases) == 0:
                    # No more items to purchase: next step impossible
                    total_next_step_utilities = tf.tensor_scatter_nd_update(
                        tensor=total_next_step_utilities, indices=[[idx]], updates=[0]
                    )
                else:
                    # Compute the dot product along the last dimension between the embeddings
                    # of the given store's theta and alpha of all the items
                    hypothetical_store_preferences = tf.reduce_sum(
                        theta_store[idx] * self.alpha, axis=1
                    )

                    if self.item_intercept:
                        # Manually enforce the lambda of the checkout item to be 0
                        # (equivalent to translating the lambda values)
                        hypothetical_item_intercept = tf.concat([[0.0], self.lambda_], axis=0)
                    else:
                        hypothetical_item_intercept = tf.zeros_like(hypothetical_store_preferences)

                    if self.price_effects:
                        hypothetical_price_effects = (
                            -1
                            # Compute the dot product along the last dimension between
                            # the embeddings of the given store's gamma and beta
                            # of all the items
                            * tf.reduce_sum(gamma_store[idx] * self.beta, axis=1)
                            * tf.math.log(price_batch[idx] + self.epsilon_price)
                        )
                    else:
                        hypothetical_price_effects = tf.zeros_like(hypothetical_store_preferences)

                    if self.seasonal_effects:
                        # Compute the dot product along the last dimension between the embeddings
                        # of delta of the given week and mu of all the items
                        hypothetical_seasonal_effects = tf.reduce_sum(
                            delta_week[idx] * self.mu, axis=1
                        )
                    else:
                        hypothetical_seasonal_effects = tf.zeros_like(
                            hypothetical_store_preferences
                        )

                    # The effects of item intercept, store preferences, price sensitivity
                    # and seasonal effects are combined in the per-item per-trip latent variable
                    hypothetical_psi = tf.reduce_sum(
                        [
                            hypothetical_item_intercept,  # 0 if self.item_intercept is False
                            hypothetical_store_preferences,
                            hypothetical_price_effects,  # 0 if self.price_effects is False
                            hypothetical_seasonal_effects,  # 0 if self.seasonal_effects is False
                        ],
                        axis=0,
                    )  # Shape: (n_items,)

                    # Shape: (len(hypothetical_next_purchases),)
                    next_psi = tf.gather(hypothetical_psi, indices=hypothetical_next_purchases)

                    # Consider hypothetical "next" item one by one
                    next_step_basket_interaction_utilities = tf.zeros(
                        (len(hypothetical_next_purchases),), dtype=tf.float32
                    )
                    for inner_idx in tf.range(len(hypothetical_next_purchases)):
                        next_item_id = tf.gather(hypothetical_next_purchases, inner_idx)
                        rho_next_item = tf.gather(
                            self.rho, indices=next_item_id
                        )  # Shape: (latent_size,)
                        # Gather the embeddings using a tensor of indices
                        # (before ensure that indices are integers)
                        next_alpha_by_basket = tf.gather(
                            self.alpha, indices=tf.cast(next_basket, dtype=tf.int32)
                        )  # Shape: (len(next_basket), latent_size)
                        # Divide the sum of alpha embeddings by the number of items
                        # in the basket of the next step (always > 0)
                        next_alpha_average = tf.reduce_sum(next_alpha_by_basket, axis=0) / tf.cast(
                            len(next_basket), dtype=tf.float32
                        )  # Shape: (latent_size,)
                        next_step_basket_interaction_utilities = tf.tensor_scatter_nd_update(
                            tensor=next_step_basket_interaction_utilities,
                            indices=[[inner_idx]],
                            # Compute the dot product along the last dimension, shape: (1,)
                            updates=[tf.reduce_sum(rho_next_item * next_alpha_average)],
                        )

                    # Optimal next step: take the maximum utility among all possible next purchases
                    next_step_utility = tf.reduce_max(
                        next_psi + next_step_basket_interaction_utilities, axis=0
                    )  # Shape: (1,)
                    total_next_step_utilities = tf.tensor_scatter_nd_update(
                        tensor=total_next_step_utilities,
                        indices=[[idx]],
                        updates=[next_step_utility],
                    )

        return total_next_step_utilities  # Shape: (batch_size,)

    def compute_batch_utility(
        self,
        item_batch: Union[np.ndarray, tf.Tensor],
        basket_batch: np.ndarray,
        store_batch: np.ndarray,
        week_batch: np.ndarray,
        price_batch: np.ndarray,
        user_batch: np.ndarray,
        available_item_batch: np.ndarray,
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
            Batch of prices (integers) for each purchased item
            Shape must be (batch_size,)
        available_item_batch: np.ndarray
            Batch of availability matrices (indicating the availability (1) or not (0)
            of the products) (arrays) for each purchased item
            Shape must be (batch_size, n_items)

        Returns
        -------
        item_utilities: tf.Tensor
            Utility of all the items in item_batch
            Shape must be (batch_size,)
        """
        _ = user_batch
        item_batch = tf.cast(item_batch, dtype=tf.int32)
        basket_batch = tf.cast(basket_batch, dtype=tf.int32)
        store_batch = tf.cast(store_batch, dtype=tf.int32)
        week_batch = tf.cast(week_batch, dtype=tf.int32)
        price_batch = tf.cast(price_batch, dtype=tf.float32)
        available_item_batch = tf.cast(available_item_batch, dtype=tf.int32)

        if len(tf.shape(item_batch)) == 1:
            if len(tf.shape(price_batch)) != 1:
                raise ValueError(f"""Arguments price_batch and item_batch should have same shape
                and are:{item_batch.shape} and {price_batch.shape}""")
            item_batch = tf.expand_dims(item_batch, axis=1)
            price_batch = tf.expand_dims(price_batch, axis=1)
            squeeze = True
        else:
            squeeze = False

        theta_store = tf.gather(self.theta, indices=store_batch)
        alpha_item = tf.gather(self.alpha, indices=item_batch)
        # Compute the dot product along the last dimension
        store_preferences = tf.einsum("kj,klj->kl", theta_store, alpha_item)

        if self.item_intercept:
            # Manually enforce the lambda of the checkout item to be 0
            # (equivalent to translating the lambda values)
            item_intercept = tf.gather(tf.concat([[0.0], self.lambda_], axis=0), indices=item_batch)
        else:
            item_intercept = tf.zeros_like(store_preferences)

        if self.price_effects:
            gamma_store = tf.gather(self.gamma, indices=store_batch)
            beta_item = tf.gather(self.beta, indices=item_batch)
            # Add epsilon to avoid NaN values (log(0))
            price_effects = (
                -1
                # Compute the dot product along the last dimension
                * tf.einsum("kj,klj->kl", gamma_store, beta_item)
                * tf.math.log(price_batch + self.epsilon_price)
            )
        else:
            gamma_store = tf.zeros_like(store_batch)
            price_effects = tf.zeros_like(store_preferences)

        if self.seasonal_effects:
            delta_week = tf.gather(self.delta, indices=week_batch)
            mu_item = tf.gather(self.mu, indices=item_batch)
            # Compute the dot product along the last dimension
            seasonal_effects = tf.einsum("kj,klj->kl", delta_week, mu_item)
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

        # Apply boolean mask to mask out the padding value -1
        masked_baskets = tf.where(
            condition=basket_batch > -1,  # If False: padding value -1
            x=1,  # Output where condition is True
            y=0,  # Output where condition is False
        )
        # Number of items in each basket
        count_items_in_basket = tf.reduce_sum(masked_baskets, axis=1)

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

        # Compute the sum of the alpha embeddings for each basket
        alpha_sum = tf.reduce_sum(alpha_by_basket, axis=1)

        rho_item = tf.gather(self.rho, indices=item_batch)

        # Divide each sum of alpha embeddings by the number of items in the corresponding basket
        # Avoid NaN values (division by 0)
        count_items_in_basket_expanded = tf.expand_dims(
            tf.cast(count_items_in_basket, dtype=tf.float32), -1
        )

        # Apply boolean mask for case distinction
        alpha_average = tf.where(
            condition=count_items_in_basket_expanded != 0,  # If True: count_items_in_basket > 0
            x=alpha_sum / count_items_in_basket_expanded,  # Output if condition is True
            y=tf.zeros_like(alpha_sum),  # Output if condition is False
        )

        # Compute the dot product along the last dimension
        basket_interaction_utility = tf.einsum("kj,klj->kl", alpha_average, rho_item)

        item_utilities = psi + basket_interaction_utility

        # No thinking ahead
        if not self.think_ahead:
            if squeeze:
                return tf.gather(item_utilities, 0, axis=1)
            return item_utilities

        # Thinking ahead
        next_step_utilities = tf.stack(
            [
                self.thinking_ahead(
                    item_batch=tf.gather(item_batch, i, axis=1),
                    ragged_basket_batch=item_indices_ragged,
                    price_batch=price_batch,
                    available_item_batch=available_item_batch,
                    theta_store=theta_store,
                    gamma_store=gamma_store,  # 0 if self.price_effects is False
                    delta_week=delta_week,  # 0 if self.seasonal_effects is False
                )
                for i in range(tf.shape(item_batch)[1])
            ],
            axis=-1,
        )

        if squeeze:
            return tf.gather(item_utilities + next_step_utilities, 0, axis=1)
        return item_utilities + next_step_utilities

    def get_negative_samples(
        self,
        available_items: np.ndarray,
        purchased_items: np.ndarray,
        future_purchases: np.ndarray,
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
        future_purchases: np.ndarray
            List of items to be purchased in the future (not yet in the basket)
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
        future_purchases = tf.cast(tf.convert_to_tensor(future_purchases), dtype=tf.int32)
        next_item = tf.cast(tf.convert_to_tensor(next_item), dtype=tf.int32)

        # Get the list of available items based on the availability matrix
        item_ids = tf.range(self.n_items)
        available_mask = tf.equal(available_items, 1)
        assortment = tf.boolean_mask(item_ids, available_mask)

        not_to_be_chosen = tf.concat(
            [purchased_items, future_purchases, tf.expand_dims(next_item, axis=0)], axis=0
        )

        # Ensure that the checkout item 0 can be picked as a negative sample
        # if it is not the next item
        # (otherwise 0 is always in not_to_be_chosen because it's in future_purchases)
        if not tf.equal(next_item, 0):
            not_to_be_chosen = tf.boolean_mask(not_to_be_chosen, not_to_be_chosen != 0)

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
        store_batch: np.ndarray
            Batch of store IDs (integers) for each purchased item
            Shape must be (batch_size,)
        week_batch: np.ndarray
            Batch of week numbers (integers) for each purchased item
            Shape must be (batch_size,)
        price_batch: np.ndarray
            Batch of prices (integers) for each purchased item
            Shape must be (batch_size,)
        available_item_batch: np.ndarray
            List of availability matrices (indicating the availability (1) or not (0)
            of the products) (arrays) for each purchased item
            Shape must be (batch_size, n_items)

        Returns
        -------
        batch_loss: tf.Variable
            Value of the loss for the batch (normalized negative log-likelihood),
            Shape must be (1,)
        loglikelihood: tf.Variable
            Computed log-likelihood of the batch of items
            Approximated by difference of utilities between positive and negative samples
            Shape must be (1,)
        """
        _ = user_batch
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
                                future_purchases=future_batch[idx],
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
            available_item_batch=tf.tile(available_item_batch, [self.n_negative_samples + 1, 1]),
            user_batch=tf.tile(user_batch, [self.n_negative_samples + 1]),
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
        )  # Shape of loglikelihood: (1,)

        # Maximize the predicted log-likelihood (ie minimize the negative log-likelihood)
        # normalized by the batch size and the number of negative samples
        batch_loss = -1 * loglikelihood / (batch_size * self.n_negative_samples)

        return batch_loss, loglikelihood
