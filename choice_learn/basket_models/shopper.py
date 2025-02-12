"""Implementation of the Shopper model."""

import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import tensorflow as tf
import tqdm

from ..tf_ops import softmax_with_availabilities
from .trip_dataset import TripDataset
from .utils.permutation import permutations


class Shopper:
    """Class for the Shopper model."""

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
                "No latent size value has been specified for preferences,\
                switching to default value, 4."
            )
        if "price" not in latent_sizes.keys() and self.price_effects:
            logging.warning(
                "No latent size value has been specified for price_effects,\
                    switching to default value, 4."
            )
        if "seasons" not in latent_sizes.keys() and self.seasonal_effects:
            logging.warning(
                "No latent size value has been specified for seasonal_effects,\
                    switching to default value, 4."
            )

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
        n_customers: int = 0,
    ) -> None:
        """Instantiate the Shopper model.

        Parameters
        ----------
        n_items: int
            Number of items to consider, i.e. the number of items in the dataset
            (includes the checkout item)
        n_customers: int
            Number of customers in the population
        """
        self.n_items = n_items
        if n_customers == 0 and self.price_effects:
            # To take into account the price effects, the number of customers must be > 0
            # to have a gamma embedding
            # (By default, the customer id is 0)
            n_customers = 1
        self.n_customers = n_customers

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
                shape=(n_customers, self.latent_sizes["preferences"])
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
            # Manually enforce the lambda of the checkout item to be 0
            # (equivalent to translating the lambda values)
            self.lambda_.assign(
                tf.tensor_scatter_nd_update(tensor=self.lambda_, indices=[[0]], updates=[0])
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
                    shape=(n_customers, self.latent_sizes["price"])
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

    def thinking_ahead(
        self,
        item_batch: Union[np.ndarray, tf.Tensor],
        basket_batch_without_padding: list,
        price_batch: np.ndarray,
        available_item_batch: np.ndarray,
        theta_customer: tf.Tensor,
        gamma_customer: tf.Tensor,
        delta_week: tf.Tensor,
    ) -> tf.Tensor:
        """Compute the utility of all the items in item_batch.

        Parameters
        ----------
        item_batch: np.ndarray or tf.Tensor
            Batch of the purchased items ID (integers) for which to compute the utility
            Shape must be (batch_size,)
            (positive and negative samples concatenated together)
        basket_batch_without_padding: list
            Batch of baskets (ID of items already in the baskets) (arrays) without padding
            for each purchased item
            Length must be batch_size
        price_batch: np.ndarray
            Batch of prices (integers) for each purchased item
            Shape must be (batch_size,)
        available_item_batch: np.ndarray
            Batch of availability matrices (indicating the availability (1) or not (0)
            of the products) (arrays) for each purchased item
            Shape must be (batch_size, n_items)
        theta_customer: tf.Tensor
            Slices from theta embedding gathered according to the indices that correspond
            to the customer of each purchased item in the batch
            Shape must be (batch_size, latent_sizes["preferences"])
        gamma_customer: tf.Tensor
            Slices from gamma embedding gathered according to the indices that correspond
            to the customer of each purchased item in the batch
            Shape must be (batch_size, latent_sizes["price"])
        delta_week: tf.Tensor
            Slices from delta embedding gathered according to the indices that correspond
            to the week of each purchased item in the batch
            Shape must be (batch_size, latent_sizes["season"])

        Returns
        -------
        tf.Tensor
            Nex step utility of all the items in item_batch
            Shape must be (batch_size,)
        """
        total_next_step_utilities = []
        # Compute the next step item utility for each element of the batch, one by one
        # TODO: avoid a for loop on basket_batch_without_padding at a later stage
        for idx, basket in enumerate(basket_batch_without_padding):
            if len(basket) and basket[-1] == 0:
                # No thinking ahead when the basket ends already with the checkout item 0
                total_next_step_utilities.append(0)

            else:
                # Basket with the hypothetical current item
                next_basket = np.append(basket, item_batch[idx])
                assortment = np.array(
                    [
                        item_id
                        for item_id in range(self.n_items)
                        if available_item_batch[idx][item_id] == 1
                    ]
                )
                hypothetical_next_purchases = np.array(
                    [item_id for item_id in assortment if item_id not in next_basket]
                )
                # Check if there are still items to purchase during the next step
                if len(hypothetical_next_purchases) == 0:
                    # No more items to purchase: next step impossible
                    total_next_step_utilities.append(0)
                else:
                    # Compute the dot product along the last dimension between the embeddings
                    # of the given customer's theta and alpha of all the items
                    hypothetical_customer_preferences = tf.reduce_sum(
                        theta_customer[idx] * self.alpha, axis=1
                    )

                    if self.item_intercept:
                        hypothetical_item_intercept = self.lambda_
                    else:
                        hypothetical_item_intercept = tf.zeros_like(
                            hypothetical_customer_preferences
                        )

                    if self.price_effects:
                        hypothetical_price_effects = (
                            -1
                            # Compute the dot product along the last dimension between
                            # the embeddings of the given customer's gamma and beta
                            # of all the items
                            * tf.reduce_sum(gamma_customer[idx] * self.beta, axis=1)
                            * tf.cast(
                                tf.math.log(price_batch[idx] + self.epsilon_price),
                                dtype=tf.float32,
                            )
                        )
                    else:
                        hypothetical_price_effects = tf.zeros_like(
                            hypothetical_customer_preferences
                        )

                    if self.seasonal_effects:
                        # Compute the dot product along the last dimension between the embeddings
                        # of delta of the given week and mu of all the items
                        hypothetical_seasonal_effects = tf.reduce_sum(
                            delta_week[idx] * self.mu, axis=1
                        )
                    else:
                        hypothetical_seasonal_effects = tf.zeros_like(
                            hypothetical_customer_preferences
                        )

                    # The effects of item intercept, customer preferences, price sensitivity
                    # and seasonal effects are combined in the per-item per-trip latent variable
                    hypothetical_psi = tf.reduce_sum(
                        [
                            hypothetical_item_intercept,  # 0 if self.item_intercept is False
                            hypothetical_customer_preferences,
                            hypothetical_price_effects,  # 0 if self.price_effects is False
                            hypothetical_seasonal_effects,  # 0 if self.seasonal_effects is False
                        ],
                        axis=0,
                    )  # Shape: (n_items,)

                    # Shape: (len(hypothetical_next_purchases),)
                    next_psi = tf.gather(hypothetical_psi, indices=hypothetical_next_purchases)

                    # Consider hypothetical "next" item one by one
                    next_step_basket_interaction_utilities = []
                    for next_item_id in hypothetical_next_purchases:
                        rho_next_item = tf.gather(
                            self.rho, indices=next_item_id
                        )  # Shape: (latent_size,)
                        # Gather the embeddings using a tensor of indices
                        # (before ensure that indices are integers)
                        next_alpha_by_basket = tf.gather(
                            self.alpha, indices=tf.cast(next_basket, dtype=tf.int32)
                        )  # Shape: (len(next_basket), latent_size)
                        # Compute the sum of the alpha embeddings
                        next_alpha_sum = tf.reduce_sum(
                            next_alpha_by_basket, axis=0
                        )  # Shape: (latent_size,)
                        # Divide the sum of alpha embeddings by the number of items
                        # in the basket of the next step (always > 0)
                        next_alpha_average = next_alpha_sum / len(
                            next_basket
                        )  # Shape: (latent_size,)
                        next_step_basket_interaction_utilities.append(
                            tf.reduce_sum(rho_next_item * next_alpha_average).numpy()
                        )  # Shape: (1,)
                    # Shape: (len(hypothetical_next_purchases),)
                    next_step_basket_interaction_utilities = tf.constant(
                        next_step_basket_interaction_utilities
                    )

                    # Optimal next step: take the maximum utility among all possible next purchases
                    next_step_utility = tf.reduce_max(
                        next_psi + next_step_basket_interaction_utilities, axis=0
                    ).numpy()  # Shape: (1,)

                    total_next_step_utilities.append(next_step_utility)

        return tf.constant(total_next_step_utilities)  # Shape: (batch_size,)

    def compute_batch_utility(
        self,
        item_batch: Union[np.ndarray, tf.Tensor],
        basket_batch: np.ndarray,
        customer_batch: np.ndarray,
        week_batch: np.ndarray,
        price_batch: np.ndarray,
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
        customer_batch: np.ndarray
            Batch of customer IDs (integers) for each purchased item
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
        # Ensure that item ids are integers
        item_batch = tf.cast(item_batch, dtype=tf.int32)

        theta_customer = tf.gather(self.theta, indices=customer_batch)
        alpha_item = tf.gather(self.alpha, indices=item_batch)
        # Compute the dot product along the last dimension
        customer_preferences = tf.reduce_sum(theta_customer * alpha_item, axis=1)

        if self.item_intercept:
            item_intercept = tf.gather(self.lambda_, indices=item_batch)
        else:
            item_intercept = tf.zeros_like(customer_preferences)

        if self.price_effects:
            gamma_customer = tf.gather(self.gamma, indices=customer_batch)
            beta_item = tf.gather(self.beta, indices=item_batch)
            # Add epsilon to avoid NaN values (log(0))
            price_effects = (
                -1
                # Compute the dot product along the last dimension
                * tf.reduce_sum(gamma_customer * beta_item, axis=1)
                * tf.cast(
                    tf.math.log(np.array(price_batch) + self.epsilon_price),
                    dtype=tf.float32,
                )
            )
        else:
            gamma_customer = tf.zeros_like(customer_batch)
            price_effects = tf.zeros_like(customer_preferences)

        if self.seasonal_effects:
            delta_week = tf.gather(self.delta, indices=week_batch)
            mu_item = tf.gather(self.mu, indices=item_batch)
            # Compute the dot product along the last dimension
            seasonal_effects = tf.reduce_sum(delta_week * mu_item, axis=1)
        else:
            delta_week = tf.zeros_like(week_batch)
            seasonal_effects = tf.zeros_like(customer_preferences)

        # The effects of item intercept, customer preferences, price sensitivity
        # and seasonal effects are combined in the per-item per-trip latent variable
        psi = tf.reduce_sum(
            [
                item_intercept,
                customer_preferences,
                price_effects,
                seasonal_effects,
            ],
            axis=0,
        )  # Shape: (batch_size,)

        # Apply boolean mask to mask out the padding value -1
        masked_baskets = tf.where(
            condition=tf.constant(basket_batch) > -1,  # If False: padding value -1
            x=1,  # Output where condition is True
            y=0,  # Output where condition is False
        )
        # Number of items in each basket
        count_items_in_basket = tf.reduce_sum(masked_baskets, axis=1)

        # Create a RaggedTensor from the indices
        basket_batch_without_padding = [basket[basket != -1] for basket in basket_batch]
        item_indices_ragged = tf.ragged.constant(basket_batch_without_padding)

        if tf.size(item_indices_ragged) == 0:
            # Empty baskets: no alpha embeddings to gather
            alpha_by_basket = tf.zeros((len(item_batch), 0, self.alpha.shape[1]))
        else:
            # Using GPU: gather the embeddings using a tensor of indices
            if self.on_gpu:
                # When using GPU, tf.nn.embedding_lookup returns 0 for ids out of bounds
                # (negative indices or indices >= len(params))
                # Cf https://github.com/tensorflow/tensorflow/issues/59724
                # https://github.com/tensorflow/tensorflow/issues/62628
                alpha_by_basket = tf.nn.embedding_lookup(params=self.alpha, ids=basket_batch)

            # Using CPU: gather the embeddings using a RaggedTensor of indices
            else:
                alpha_by_basket = tf.ragged.map_flat_values(
                    tf.gather, self.alpha, item_indices_ragged
                )

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
        basket_interaction_utility = tf.reduce_sum(rho_item * alpha_average, axis=1)

        item_utilities = psi + basket_interaction_utility

        # No thinking ahead
        if not self.think_ahead:
            return item_utilities

        # Thinking ahead
        next_step_utilities = self.thinking_ahead(
            item_batch=item_batch,
            basket_batch_without_padding=basket_batch_without_padding,
            price_batch=price_batch,
            available_item_batch=available_item_batch,
            theta_customer=theta_customer,
            gamma_customer=gamma_customer,  # 0 if self.price_effects is False
            delta_week=delta_week,  # 0 if self.seasonal_effects is False
        )

        return item_utilities + next_step_utilities

    def compute_item_likelihood(
        self,
        basket: np.ndarray,
        available_items: np.ndarray,
        customer: int,
        week: int,
        prices: np.ndarray,
    ) -> tf.Tensor:
        """Compute the likelihood of all items for a given trip.

        Parameters
        ----------
        basket: np.ndarray
            ID the of items already in the basket
        available_items: np.ndarray
            Matrix indicating the availability (1) or not (0) of the products
            Shape must be (n_items,)
        customer: int
            Customer id
        week: int
            Week number
        prices: np.ndarray
            Prices of all the items in the dataset
            Shape must be (n_items,)

        Returns
        -------
        likelihood: tf.Tensor
            Likelihood of all items for a given trip
            Shape must be (n_items,)
        """
        # Prevent unintended side effects from in-place modifications
        available_items_copy = available_items.copy()

        # Compute the utility of all the items
        all_utilities = self.compute_batch_utility(
            # All items
            item_batch=np.array([item_id for item_id in range(self.n_items)]),
            # For each item: same basket / customer / week / prices / available items
            basket_batch=np.array([basket for _ in range(self.n_items)]),
            customer_batch=np.array([customer for _ in range(self.n_items)]),
            week_batch=np.array([week for _ in range(self.n_items)]),
            price_batch=prices,
            available_item_batch=np.array([available_items_copy for _ in range(self.n_items)]),
        )

        # Softmax on the utilities
        return softmax_with_availabilities(
            items_logit_by_choice=all_utilities,  # Shape: (n_items,)
            available_items_by_choice=available_items_copy,  # Shape: (n_items,)
            axis=-1,
            normalize_exit=False,
            eps=None,
        )

    def compute_ordered_basket_likelihood(
        self,
        basket: np.ndarray,
        available_items: np.ndarray,
        customer: int,
        week: int,
        prices: np.ndarray,
    ) -> float:
        """Compute the utility of an ordered basket.

        Parameters
        ----------
        basket: np.ndarray
            ID the of items already in the basket
        available_items: np.ndarray
            Matrix indicating the availability (1) or not (0) of the products
            Shape must be (n_items,)
        customer: int
            Customer id
        week: int
            Week number
        prices: np.ndarray
            Prices of all the items in the dataset

        Returns
        -------
        likelihood: float
            Likelihood of the ordered basket
        """
        # Prevent unintended side effects from in-place modifications
        available_items_copy = available_items.copy()

        ordered_basket_likelihood = 1.0
        for j in range(0, len(basket)):
            next_item_id = basket[j]

            # Compute the likelihood of the j-th item of the basket
            ordered_basket_likelihood *= self.compute_item_likelihood(
                basket=basket[:j],
                available_items=available_items_copy,
                customer=customer,
                week=week,
                prices=prices,
            )[next_item_id].numpy()

            # This item is not available anymore
            available_items_copy[next_item_id] = 0

        return ordered_basket_likelihood

    def compute_basket_likelihood(
        self,
        basket: np.ndarray,
        available_items: np.ndarray,
        customer: int,
        week: int,
        prices: np.ndarray,
        n_permutations: int = 1,
        verbose: int = 0,
    ) -> float:
        """Compute the utility of an (unordered) basket.

        Parameters
        ----------
        basket: np.ndarray
            ID the of items already in the basket
        available_items: np.ndarray
            Matrix indicating the availability (1) or not (0) of the products
            Shape must be (n_items,)
        customer: int
            Customer id
        week: int
            Week number
        prices: np.ndarray
            Prices of all the items in the dataset
        n_permutations: int, optional
            Number of permutations to average over, by default 1
        verbose: int, optional
            print level, for debugging, by default 0
            (0: no print, 1: print)

        Returns
        -------
        likelihood: float
            Likelihood of the (unordered) basket
        """
        if verbose > 0:
            print(
                f"Nb of items to be permuted = basket size - 1 = {len(basket) - 1}",
                f"Nb of permutations = n! = {np.math.factorial(len(basket) - 1)}",
            )

        # Permute all the items in the basket except the last one (the checkout item)
        permutation_list = list(permutations(range(len(basket) - 1)))
        total_n_permutations = len(permutation_list)  # = n!

        # Limit the number of permutations to n!
        if n_permutations <= total_n_permutations:
            permutation_list = random.sample(permutation_list, n_permutations)
        else:
            print(
                "Warning: n_permutations > n! (all permutations). ",
                "Taking all permutations instead of n_permutations",
            )

        return (
            np.mean(
                [
                    self.compute_ordered_basket_likelihood(
                        # The last item should always be the checkout item 0
                        basket=[basket[i] for i in permutation] + [0],
                        available_items=available_items,
                        customer=customer,
                        week=week,
                        prices=prices,
                    )
                    for permutation in permutation_list
                ]
            )
            * total_n_permutations
        )  # Rescale the mean to the total number of permutations

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
        # Get the list of available items based on the availability matrix
        assortment = [item_id for item_id in range(self.n_items) if available_items[item_id] == 1]

        not_to_be_chosen = np.unique(
            np.concatenate([purchased_items, future_purchases, [next_item]])
        )

        # Ensure that the checkout item 0 can be picked as a negative sample
        # if it is not the next item
        # (otherwise 0 is always in not_to_be_chosen because it's in future_purchases)
        if next_item:
            not_to_be_chosen = np.setdiff1d(not_to_be_chosen, [0])

        # Items that can be picked as negative samples
        possible_items = np.setdiff1d(assortment, not_to_be_chosen)

        # Ensure that the while loop will not run indefinitely
        if n_samples > len(possible_items):
            raise ValueError(
                "The number of samples to draw must be less than the "
                "number of available items not already purchased and "
                "distinct from the next item."
            )

        return random.sample(list(possible_items), n_samples)

    def compute_batch_loss(
        self,
        item_batch: np.ndarray,
        basket_batch: np.ndarray,
        future_batch: np.ndarray,
        customer_batch: np.ndarray,
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
        future_batch: np.ndarray
            Batch of items to be purchased in the future (ID of items not yet in the
            basket) (arrays) for each purchased item
            Shape must be (batch_size, max_basket_size)
        customer_batch: np.ndarray
            Batch of customer IDs (integers) for each purchased item
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
        batch_size = len(item_batch)

        # Negative sampling
        negative_samples = (
            np.concatenate(
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
                # Reshape to have at the beginning of the array all the first negative samples
                # of all positive samples, then all the second negative samples, etc.
                # (same logic as for the calls to np.tile)
            )
            .reshape(batch_size, self.n_negative_samples)
            .T.flatten()
        )

        augmented_item_batch = np.concatenate((item_batch, negative_samples)).astype(int)
        prices_tiled = np.tile(price_batch, (self.n_negative_samples + 1, 1))
        # Each time, pick only the price of the item in augmented_item_batch from the
        # corresponding price array
        augmented_price_batch = np.array(
            [
                prices_tiled[idx][augmented_item_batch[idx]]
                for idx in range(len(augmented_item_batch))
            ]
        )

        # Compute the utility of all the available items
        all_utilities = self.compute_batch_utility(
            item_batch=augmented_item_batch,
            basket_batch=np.tile(basket_batch, (self.n_negative_samples + 1, 1)),
            customer_batch=np.tile(customer_batch, self.n_negative_samples + 1),
            week_batch=np.tile(week_batch, self.n_negative_samples + 1),
            price_batch=augmented_price_batch,
            available_item_batch=np.tile(available_item_batch, (self.n_negative_samples + 1, 1)),
        )

        positive_samples_utilities = all_utilities[:batch_size]
        negative_samples_utilities = all_utilities[batch_size:]

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

    # @tf.function # TODO: not working for now
    def train_step(
        self,
        item_batch: np.ndarray,
        basket_batch: np.ndarray,
        future_batch: np.ndarray,
        customer_batch: np.ndarray,
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
        future_batch: np.ndarray
            Batch of items to be purchased in the future (ID of items not yet in the
            basket) (arrays) for each purchased item
            Shape must be (batch_size, max_basket_size)
        customer_batch: np.ndarray
            Batch of customer ids (integers) for each purchased item
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
        batch_loss: tf.Tensor
            Value of the loss for the batch
        """
        with tf.GradientTape() as tape:
            batch_loss = self.compute_batch_loss(
                item_batch=item_batch,
                basket_batch=basket_batch,
                future_batch=future_batch,
                customer_batch=customer_batch,
                week_batch=week_batch,
                price_batch=price_batch,
                available_item_batch=available_item_batch,
            )[0]
        grads = tape.gradient(batch_loss, self.trainable_weights)

        # Set the gradient of self.lambda_[0] to 0 to prevent updates
        # so that the lambda of the checkout item remains 0
        # (equivalent to translating the lambda values)
        if self.item_intercept:
            # Find the index of the lambda_ variable in the trainable weights
            # Cannot use list.index() method on a GPU, use next() instead
            # (ie compare object references instead of tensor values)
            lambda_grads = grads[
                next(i for i, v in enumerate(self.trainable_weights) if v is self.lambda_)
            ]
            lambda_grads = tf.tensor_scatter_nd_update(lambda_grads, indices=[[0]], updates=[0])
            grads[next(i for i, v in enumerate(self.trainable_weights) if v is self.lambda_)] = (
                lambda_grads
            )

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
            self.instantiate(n_items=trip_dataset.n_items, n_customers=trip_dataset.n_customers)

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

            if verbose > 0:
                inner_range = tqdm.tqdm(
                    trip_dataset.iter_batch(
                        shuffle=True,
                        batch_size=batch_size,
                    ),
                    total=int(trip_dataset.n_samples / np.max([batch_size, 1])),
                    position=1,
                    leave=False,
                )
            else:
                inner_range = trip_dataset.iter_batch(shuffle=True, batch_size=batch_size)

            for batch_nb, (
                item_batch,
                basket_batch,
                future_batch,
                customer_batch,
                week_batch,
                price_batch,
                available_item_batch,
            ) in enumerate(inner_range):
                self.callbacks.on_train_batch_begin(batch_nb)

                batch_loss = self.train_step(
                    item_batch=item_batch,
                    basket_batch=basket_batch,
                    future_batch=future_batch,
                    customer_batch=customer_batch,
                    week_batch=week_batch,
                    price_batch=price_batch,
                    available_item_batch=available_item_batch,
                )
                train_logs["train_loss"].append(batch_loss)
                temps_logs = {k: tf.reduce_mean(v) for k, v in train_logs.items()}
                self.callbacks.on_train_batch_end(batch_nb, logs=temps_logs)

                # Optimization Steps
                epoch_losses.append(batch_loss)

                if verbose > 0:
                    inner_range.set_description(
                        f"Epoch Negative-LogLikeliHood: {np.sum(epoch_losses):.4f}"
                    )

            # Take into account the fact that the last batch may have a
            # different length for the computation of the epoch loss.
            if batch_size != -1:
                last_batch_size = len(item_batch)
                coefficients = tf.concat(
                    [tf.ones(len(epoch_losses) - 1) * batch_size, [last_batch_size]],
                    axis=0,
                )
                epoch_losses = tf.multiply(epoch_losses, coefficients)
                epoch_loss = tf.reduce_sum(epoch_losses) / trip_dataset.n_samples
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
                    future_batch,
                    customer_batch,
                    week_batch,
                    price_batch,
                    available_item_batch,
                ) in enumerate(val_dataset.iter_batch(shuffle=True, batch_size=batch_size)):
                    self.callbacks.on_batch_begin(batch_nb)
                    self.callbacks.on_test_batch_begin(batch_nb)

                    val_losses.append(
                        self.compute_batch_loss(
                            item_batch=item_batch,
                            basket_batch=basket_batch,
                            future_batch=future_batch,
                            customer_batch=customer_batch,
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
        n_permutations: int = 1,
        epsilon_eval: float = 1e-6,
    ) -> tf.Tensor:
        """Evaluate the model for each trip (unordered basket) in the dataset.

        Predicts the probabilities according to the model and then computes the
        mean negative log-likelihood (nll) for the dataset

        N.B.: Some randomness is involved in the evaluation through random sampling
        of permutations at 2 levels:
        - During batch processing: random permutation of the items in the basket
        when creating augmented data from a trip index
        - During the computation of the likelihood of an (unordered) basket: approximation
        by the average of the likelihoods of several permutations of the basket

        Parameters
        ----------
        trip_dataset: TripDataset‡‡
            Dataset on which to apply to prediction
        n_permutations: int, optional
            Number of permutations to average over, by default 1
        epsilon_eval: float, optional
            Small value to avoid log(0) in the computation of the log-likelihood,
            by default 1e-6

        Returns
        -------
        loss: tf.Tensor
            Value of the mean loss (nll) for the dataset,
            Shape must be (1,)
        """
        (
            _,
            basket_batch,
            _,
            customer_batch,
            week_batch,
            price_batch,
            available_item_batch,
        ) = list(  # Convert the generator to a list (of 1 element here)
            trip_dataset.iter_batch(shuffle=True, batch_size=-1)
        )[0]

        batch_size = len(basket_batch)  # Here: batch = whole TripDataset

        # Sum of the log-likelihoods of all the (unordered) baskets
        sum_loglikelihoods = np.sum(
            np.log(
                [
                    self.compute_basket_likelihood(
                        basket=basket,
                        available_items=available_items,
                        customer=customer,
                        week=week,
                        prices=prices,
                        n_permutations=n_permutations,
                    )
                    + epsilon_eval
                    for basket, available_items, customer, week, prices in zip(
                        basket_batch, available_item_batch, customer_batch, week_batch, price_batch
                    )
                ]
            )
        )

        # Predicted mean negative log-likelihood
        return -1 * sum_loglikelihoods / batch_size

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
    def load_model(cls, path: str) -> object:
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
            think_ahead=params["think_ahead"],
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
        model.n_customers = params["n_customers"]

        # Fix manually trainable weights values
        model.rho = tf.Variable(np.load(os.path.join(path, "rho.npy")), trainable=True, name="rho")
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
