"""Implementation of the Shopper model."""

import json
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
        stage: int = 1,
        latent_sizes: dict[str] = {"preferences": 1, "price": 1, "season": 1},
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
        stage: int, optional
            Modeling stage of the model (1, 2 or 3), by default 1
        latent_sizes: dict[str]
            Lengths of the vector representation of the latent parameters
            latent_sizes["preferences"]: length of one vector of theta, alpha, rho
            latent_sizes["price"]: length of one vector of gamma, beta
            latent_sizes["season"]: length of one vector of delta, mu
        n_negative_samples: int, optional
            Number of negative samples to draw for each positive sample for the training,
            by default 2
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
        if stage not in range(1, 4):
            raise ValueError("Stage number must be between 1 and 3, inclusive.")
        self.stage = stage

        if latent_sizes.keys() != {"preferences", "price", "season"}:
            raise ValueError(
                "The latent_sizes dictionary must contain the keys 'preferences', 'price' and "
                "'season'."
            )
        self.latent_sizes = latent_sizes

        self.n_negative_samples = n_negative_samples

        self.optimizer_name = optimizer
        if optimizer.lower() == "adam":
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr, clipvalue=grad_clip_value, weight_decay=weight_decay
            )
        elif optimizer.lower() == "amsgrad":
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr, amsgrad=True, clipvalue=grad_clip_value, weight_decay=weight_decay
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

        self.instantiated = False

    def instantiate(
        self,
        n_items: int,
        n_customers: int,
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

        if self.stage > 1:
            # In addition to customer preferences: item popularity, price sensitivity
            # and seasonal effects
            self.lambda_ = tf.Variable(
                tf.random_normal_initializer(mean=0, stddev=1.0, seed=42)(
                    shape=(n_items,)  # Dimension for 1 item: 1
                ),
                trainable=True,
                name="lambda",
            )
            self.beta = tf.Variable(
                tf.random_normal_initializer(mean=0, stddev=1.0, seed=42)(
                    shape=(n_items, self.latent_sizes["price"])
                ),  # Dimension for 1 item: latent_sizes["price"]
                trainable=True,
                name="beta",
            )
            self.mu = tf.Variable(
                tf.random_normal_initializer(mean=0, stddev=0.1, seed=42)(
                    shape=(n_items, self.latent_sizes["season"])
                ),  # Dimension for 1 item: latent_sizes["season"]
                trainable=True,
                name="mu",
            )
            self.gamma = tf.Variable(
                tf.random_normal_initializer(mean=0, stddev=1.0, seed=42)(
                    shape=(n_customers, self.latent_sizes["price"])
                ),  # Dimension for 1 item: latent_sizes["price"]
                trainable=True,
                name="gamma",
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
        if self.stage == 1:
            return [self.rho, self.alpha, self.theta]

        # Stage 2 or 3
        return [
            self.rho,
            self.alpha,
            self.theta,
            self.lambda_,
            self.beta,
            self.mu,
            self.gamma,
            self.delta,
        ]

    def compute_batch_utility(
        self,
        item_list: Union[np.ndarray, tf.Tensor],
        basket_list: np.ndarray,
        customer_list: np.ndarray,
        week_list: np.ndarray,
        price_list: np.ndarray,
        av_matrix_list: np.ndarray,
    ) -> tf.Tensor:
        """Compute the utility of all the items in item_list.

        Parameters
        ----------
        item_list: np.ndarray or tf.Tensor
            List of the purchased items ID (integers) for which to compute the utility
            Shape must be (batch_size * (n_negative_samples + 1),)
            (positive and negative samples concatenated together)
        basket_list: np.ndarray
            List of baskets (ID of items already in the baskets) (arrays) for each purchased item
            Shape must be (batch_size * (n_negative_samples + 1), max_basket_size)
        customer_list: np.ndarray
            List of customer IDs (integers) for each purchased item
            Shape must be (batch_size * (n_negative_samples + 1),)
        week_list: np.ndarray
            List of week numbers (integers) for each purchased item
            Shape must be (batch_size * (n_negative_samples + 1),)
        price_list: np.ndarray
            List of prices (integers) for each purchased item
            Shape must be (batch_size * (n_negative_samples + 1),)
        av_matrix_list: np.ndarray
            List of availability matrices (indicating the availability (1) or not (0)
            of the products) (arrays) for each purchased item
            Shape must be (batch_size * (n_negative_samples + 1), n_items)

        Returns
        -------
        item_utilities: tf.Tensor
            Utility of all the items in item_list
            Shape must be (batch_size * (n_negative_samples + 1),)
        """
        # Ensure that item ids are integers
        item_list = tf.cast(item_list, dtype=tf.int32)

        # Psi values
        if self.stage == 1:
            theta_customer = tf.gather(self.theta, indices=customer_list)
            alpha_item = tf.gather(self.alpha, indices=item_list)
            # Compute the dot product along the last dimension
            psi = tf.reduce_sum(theta_customer * alpha_item, axis=1)

        else:
            # The effects of item popularity, customer preferences, price sensitivity
            # and seasonal effects are combined in the per-item per-trip latent variable
            item_popularity = tf.gather(self.lambda_, indices=item_list)

            theta_customer = tf.gather(self.theta, indices=customer_list)
            alpha_item = tf.gather(self.alpha, indices=item_list)
            # Compute the dot product along the last dimension
            customer_preferences = tf.reduce_sum(theta_customer * alpha_item, axis=1)

            gamma_customer = tf.gather(self.gamma, indices=customer_list)
            beta_item = tf.gather(self.beta, indices=item_list)
            price_effects = (
                -1
                # Compute the dot product along the last dimension
                * tf.reduce_sum(gamma_customer * beta_item, axis=1)
                * tf.cast(tf.math.log(np.array(price_list) + self.epsilon_price), dtype=tf.float32)
            )

            delta_week = tf.gather(self.delta, indices=week_list)
            mu_item = tf.gather(self.mu, indices=item_list)
            # Compute the dot product along the last dimension
            seasonal_effects = tf.reduce_sum(delta_week * mu_item, axis=1)

            psi = tf.reduce_sum(
                [
                    item_popularity,
                    customer_preferences,
                    price_effects,
                    seasonal_effects,
                ],
                axis=0,
            )

        # Apply boolean mask to mask out the padding value -1
        masked_tensors = tf.where(
            condition=tf.constant(basket_list) > -1,  # If False: padding value -1
            x=1,  # Output where condition is True
            y=0,  # Output where condition is False
        )
        # Number of items in each basket
        count_items_in_basket = tf.reduce_sum(masked_tensors, axis=1)

        # Create a RaggedTensor from the indices
        basket_list_without_padding = [basket[basket != -1] for basket in basket_list]
        indices_ragged = tf.ragged.constant(basket_list_without_padding)

        if tf.size(indices_ragged) == 0:
            # Empty baskets: no alpha embeddings to gather
            alpha_by_basket = tf.zeros((len(item_list), 0, self.alpha.shape[1]))
        else:
            # Using GPU: gather the embeddings using a tensor of indices
            if len(tf.config.get_visible_devices("GPU")):
                # When using GPU, tf.nn.embedding_lookup returns 0 for ids out of bounds
                # (negative indices or indices >= len(params))
                # Cf https://github.com/tensorflow/tensorflow/issues/59724
                # https://github.com/tensorflow/tensorflow/issues/62628
                alpha_by_basket = tf.nn.embedding_lookup(params=self.alpha, ids=basket_list)

            # Using CPU: gather the embeddings using a RaggedTensor of indices
            else:
                alpha_by_basket = tf.ragged.map_flat_values(tf.gather, self.alpha, indices_ragged)

        # Compute the sum of the alpha embeddings for each basket
        alpha_sum = tf.reduce_sum(alpha_by_basket, axis=1)

        rho_item = tf.gather(self.rho, indices=item_list)

        # Divide each sum of alpha embeddings by the number of items in the corresponding basket
        # Avoid NaN values (division by 0)
        count_items_in_basket_expanded = tf.expand_dims(
            tf.cast(count_items_in_basket, dtype=tf.float32), -1
        )
        alpha_average = tf.where(
            condition=count_items_in_basket_expanded != 0,  # If True: count_items_in_basket > 0
            x=alpha_sum / count_items_in_basket_expanded,  # Output if condition is True
            y=tf.zeros_like(alpha_sum),  # Output if condition is False
        )

        # Compute the dot product along the last dimension
        product_to_add_to_psi = tf.reduce_sum(rho_item * alpha_average, axis=1)

        false_output = psi
        true_output = psi + product_to_add_to_psi

        # Apply boolean mask for case distinction
        item_utilities = tf.where(
            condition=count_items_in_basket > 0,  # If False: empty basket
            x=true_output,  # Output if condition is True
            y=false_output,  # Output if condition is False
        )

        ##### No thinking ahead #####
        if self.stage < 3:
            return item_utilities

        ##### Thinking ahead #####
        total_next_step_utilities = []
        # Compute the next step item utility for each element of the batch, one by one
        # TODO: avoid a for loop on basket_list_without_padding at a later stage
        for idx, basket in enumerate(basket_list_without_padding):
            if len(basket) and basket[-1] == 0:
                # No thinking ahead when the basket ends already with the checkout item 0
                total_next_step_utilities.append(0)
            else:
                # Basket with the hypothetical current item
                next_basket = np.append(basket, item_list[idx])

                assortment = np.array(
                    [
                        item_id
                        for item_id in range(self.n_items)
                        if av_matrix_list[idx][item_id] == 1
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
                    # The effects of item popularity, customer preferences, price sensitivity
                    # and seasonal effects are combined in the per-item per-trip latent variable
                    hypothetical_item_popularity = self.lambda_

                    # Compute the dot product along the last dimension between the embeddings
                    # of the given customer's theta and alpha of all the items
                    hypothetical_customer_preferences = tf.reduce_sum(
                        theta_customer[idx] * self.alpha, axis=1
                    )

                    hypothetical_price_effects = (
                        -1
                        # Compute the dot product along the last dimension between the embeddings
                        # of the given customer's gamma and beta of all the items
                        * tf.reduce_sum(gamma_customer[idx] * self.beta, axis=1)
                        * tf.cast(
                            tf.math.log(price_list[idx] + self.epsilon_price), dtype=tf.float32
                        )
                    )

                    # Compute the dot product along the last dimension between the embeddings
                    # of delta of the given week and mu of all the items
                    hypothetical_seasonal_effects = tf.reduce_sum(delta_week[idx] * self.mu, axis=1)

                    # Shape: (n_items,)
                    hypothetical_psi = tf.reduce_sum(
                        [
                            hypothetical_item_popularity,
                            hypothetical_customer_preferences,
                            hypothetical_price_effects,
                            hypothetical_seasonal_effects,
                        ],
                        axis=0,
                    )
                    # Shape: (len(hypothetical_next_purchases),)
                    next_psi = tf.gather(hypothetical_psi, indices=hypothetical_next_purchases)

                    # Consider hypothetical "next" item one by one
                    next_step_products_to_add_to_psi = []
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

                        next_step_products_to_add_to_psi.append(
                            tf.reduce_sum(rho_next_item * next_alpha_average).numpy()
                        )  # Shape: (1,)

                    # Shape: (len(hypothetical_next_purchases),)
                    next_step_products_to_add_to_psi = tf.constant(next_step_products_to_add_to_psi)

                    # Optimal next step: take the maximum utility among all possible next purchases
                    next_step_utility = tf.reduce_max(
                        next_psi + next_step_products_to_add_to_psi, axis=0
                    ).numpy()  # Shape: (1,)

                    total_next_step_utilities.append(next_step_utility)

        total_next_step_utilities = tf.constant(
            total_next_step_utilities
        )  # Shape: (batch_size * (n_negative_samples + 1),)

        return item_utilities + total_next_step_utilities

    def compute_item_likelihood(
        self,
        basket: np.ndarray,
        availability_matrix: np.ndarray,
        customer: int,
        week: int,
        prices: np.ndarray,
    ) -> tf.Tensor:
        """Compute the likelihood of all items for a given trip.

        Parameters
        ----------
        basket: np.ndarray
            ID the of items already in the basket
        availability_matrix: np.ndarray
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
        availability_matrix_copy = availability_matrix.copy()

        # Compute the utility of all the items
        all_utilities = self.compute_batch_utility(
            # All items
            item_list=np.array([item_id for item_id in range(self.n_items)]),
            # For each item: same basket / customer / week / prices / availability matrix
            basket_list=np.array([basket for _ in range(self.n_items)]),
            customer_list=np.array([customer for _ in range(self.n_items)]),
            week_list=np.array([week for _ in range(self.n_items)]),
            price_list=prices,
            av_matrix_list=np.array([availability_matrix_copy for _ in range(self.n_items)]),
        )

        # Equation (3) of the paper Shopper, ie softmax on the utilities
        return softmax_with_availabilities(
            items_logit_by_choice=all_utilities,  # Shape: (n_items,)
            available_items_by_choice=availability_matrix_copy,  # Shape: (n_items,)
            axis=-1,
            normalize_exit=False,
            eps=None,
        )

    def compute_ordered_basket_likelihood(
        self,
        basket: np.ndarray,
        availability_matrix: np.ndarray,
        customer: int,
        week: int,
        prices: np.ndarray,
    ) -> float:
        """Compute the utility of an ordered basket.

        Parameters
        ----------
        basket: np.ndarray
            ID the of items already in the basket
        availability_matrix: np.ndarray
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
        availability_matrix_copy = availability_matrix.copy()

        # Equation (5) of the paper Shopper
        ordered_basket_likelihood = 1.0
        for j in range(0, len(basket)):
            next_item_id = basket[j]

            # Compute the likelihood of the j-th item of the basket
            ordered_basket_likelihood *= self.compute_item_likelihood(
                basket=basket[:j],
                availability_matrix=availability_matrix_copy,
                customer=customer,
                week=week,
                prices=prices,
            )[next_item_id].numpy()

            # This item is not available anymore
            availability_matrix_copy[next_item_id] = 0

        return ordered_basket_likelihood

    def compute_unordered_basket_likelihood(
        self,
        basket: np.ndarray,
        availability_matrix: np.ndarray,
        customer: int,
        week: int,
        prices: np.ndarray,
        verbose: int = 0,
    ) -> float:
        """Compute the utility of an unordered basket.

        Parameters
        ----------
        basket: np.ndarray
            ID the of items already in the basket
        availability_matrix: np.ndarray
            Matrix indicating the availability (1) or not (0) of the products
            Shape must be (n_items,)
        customer: int
            Customer id
        week: int
            Week number
        prices: np.ndarray
            Prices of all the items in the dataset
        verbose: int, optional
            print level, for debugging, by default 0
            (0: no print, 1: print)

        Returns
        -------
        likelihood: float
            Likelihood of the unordered basket
        """
        if verbose > 0:
            print(
                f"Nb of items to be permuted = basket size - 1 = {len(basket) - 1}",
                f"Nb of permutations = n! = {np.math.factorial(len(basket) - 1)}",
            )

        # Permute all the items in the basket except the last one (the checkout item)
        permutation_list = list(permutations(range(len(basket) - 1)))

        # Equation (6) of the paper Shopper
        return sum(
            [
                self.compute_ordered_basket_likelihood(
                    # The last item should always be the checkout item 0
                    basket=[basket[i] for i in permutation] + [0],
                    availability_matrix=availability_matrix,
                    customer=customer,
                    week=week,
                    prices=prices,
                )
                for permutation in permutation_list
            ]
        )

    def get_negative_samples(
        self,
        availability_matrix: np.ndarray,
        purchased_items: np.ndarray,
        next_item: int,
        n_samples: int,
    ) -> list[int]:
        """Sample randomly a set of items.

        (set of items not already purchased and *not necessarily* from the basket)

        Parameters
        ----------
        availability_matrix: np.ndarray
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
        # Build the assortment based on the availability matrix
        assortment = [
            item_id for item_id in range(self.n_items) if availability_matrix[item_id] == 1
        ]

        # Ensure that the while loop will not run indefinitely
        n_available_items = len(
            [item for item in assortment if item not in purchased_items and item != next_item]
        )
        if n_samples > n_available_items:
            raise ValueError(
                "The number of samples to draw must be less than the "
                "number of available items not already purchased and "
                "distinct from the next item."
            )

        negative_samples = []
        while len(negative_samples) < n_samples:
            # Sample a random item
            item = random.sample(assortment, 1)[0]
            if item not in negative_samples:
                # Check that the sample is distinct from the next item
                # and from the items already in the basket
                if item not in purchased_items and item != next_item:
                    negative_samples.append(item)

        return negative_samples

    def batch_predict(
        self,
        item_batch: np.ndarray,
        basket_batch: np.ndarray,
        customer_batch: np.ndarray,
        week_batch: np.ndarray,
        price_batch: np.ndarray,
        av_matrix_batch: np.ndarray,
    ) -> tuple[tf.Variable]:
        """Prediction (log-likelihood and loss) for one batch of items.

        Parameters
        ----------
        item_batch: np.ndarray
            Batch of purchased item IDs (integers)
            Shape must be (batch_size,)
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
        av_matrix_batch: np.ndarray
            List of availability matrices (indicating the availability (1) or not (0)
            of the products) (arrays) for each purchased item
            Shape must be (batch_size, n_items)

        Returns
        -------
        batch_loss: tf.Variable
            Value of the loss for the batch,
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
                        availability_matrix=av_matrix_batch[idx],
                        purchased_items=basket_batch[idx],
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

        item_list = np.concatenate((item_batch, negative_samples)).astype(int)
        prices_tiled = np.tile(price_batch, (self.n_negative_samples + 1, 1))
        # Each time, pick only the price of the item in item_list from the
        # corresponding price array
        price_list = np.array([prices_tiled[idx][item_list[idx]] for idx in range(len(item_list))])

        # Compute the utility of all the available items
        all_utilities = self.compute_batch_utility(
            item_list=item_list,
            basket_list=np.tile(basket_batch, (self.n_negative_samples + 1, 1)),
            customer_list=np.tile(customer_batch, self.n_negative_samples + 1),
            week_list=np.tile(week_batch, self.n_negative_samples + 1),
            price_list=price_list,
            av_matrix_list=np.tile(av_matrix_batch, (self.n_negative_samples + 1, 1)),
        )

        positive_samples_utilities = all_utilities[:batch_size]
        negative_samples_utilities = all_utilities[batch_size:]

        # Equation (29) of the paper Shopper
        # Loglikelihood of a batch = sum of loglikelihoods of its samples
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
        customer_batch: np.ndarray,
        week_batch: np.ndarray,
        price_batch: np.ndarray,
        av_matrix_batch: np.ndarray,
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
        customer_batch: np.ndarray
            Batch of customer ids (integers) for each purchased item
            Shape must be (batch_size,)
        week_batch: np.ndarray
            Batch of week numbers (integers) for each purchased item
            Shape must be (batch_size,)
        price_batch: np.ndarray
            Batch of prices (integers) for each purchased item
            Shape must be (batch_size,)
        av_matrix_batch: np.ndarray
            List of availability matrices (indicating the availability (1) or not (0)
            of the products) (arrays) for each purchased item
            Shape must be (batch_size, n_items)

        Returns
        -------
        batch_loss: tf.Tensor
            Value of the loss for the batch
        """
        with tf.GradientTape() as tape:
            batch_loss = self.batch_predict(
                item_batch=item_batch,
                basket_batch=basket_batch,
                customer_batch=customer_batch,
                week_batch=week_batch,
                price_batch=price_batch,
                av_matrix_batch=av_matrix_batch,
            )[0]
        grads = tape.gradient(batch_loss, self.trainable_weights)

        # Filter out None gradients
        non_none_grads = [grad for grad in grads if grad is not None]

        # Compute the norm of the gradients
        self.norm_grads = [float(tf.norm(grad).numpy()) for grad in non_none_grads]

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
            self.n_items = TripDataset.n_items()
            self.n_customers = TripDataset.n_customers()

            self.rho = tf.Variable(
                tf.random_normal_initializer(mean=0, stddev=1.0, seed=42)(
                    shape=(self.n_items, self.latent_sizes["preferences"])
                ),  # Dimension for 1 item: latent_sizes["preferences"]
                trainable=True,
                name="rho",
            )
            self.alpha = tf.Variable(
                tf.random_normal_initializer(mean=0, stddev=1.0, seed=42)(
                    shape=(self.n_items, self.latent_sizes["preferences"])
                ),  # Dimension for 1 item: latent_sizes["preferences"]
                trainable=True,
                name="alpha",
            )
            self.theta = tf.Variable(
                tf.random_normal_initializer(mean=0, stddev=1.0, seed=42)(
                    shape=(self.n_customers, self.latent_sizes["preferences"])
                ),  # Dimension for 1 item: latent_sizes["preferences"]
                trainable=True,
                name="theta",
            )

            if self.stage > 1:
                # In addition to customer preferences: item popularity, price sensitivity
                # and seasonal effects
                self.lambda_ = tf.Variable(
                    tf.random_normal_initializer(mean=0, stddev=1.0, seed=42)(
                        shape=(self.n_items,)  # Dimension for 1 item: 1
                    ),
                    trainable=True,
                    name="lambda",
                )
                self.beta = tf.Variable(
                    tf.random_normal_initializer(mean=0, stddev=1.0, seed=42)(
                        shape=(self.n_items, self.latent_sizes["price"])
                    ),  # Dimension for 1 item: latent_sizes["price"]
                    trainable=True,
                    name="beta",
                )
                self.mu = tf.Variable(
                    tf.random_normal_initializer(mean=0, stddev=0.1, seed=42)(
                        shape=(self.n_items, self.latent_sizes["season"])
                    ),  # Dimension for 1 item: latent_sizes["season"]
                    trainable=True,
                    name="mu",
                )
                self.gamma = tf.Variable(
                    tf.random_normal_initializer(mean=0, stddev=1.0, seed=42)(
                        shape=(self.n_customers, self.latent_sizes["price"])
                    ),  # Dimension for 1 item: latent_sizes["price"]
                    trainable=True,
                    name="gamma",
                )
                self.delta = tf.Variable(
                    tf.random_normal_initializer(mean=0, stddev=0.1, seed=42)(
                        shape=(52, self.latent_sizes["season"])
                    ),  # Dimension for 1 item: latent_sizes["season"]
                    trainable=True,
                    name="delta",
                )

            self.instantiated = True

        batch_size = self.batch_size

        history = {"train_loss": [], "norm_grads": []}
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
                        # TODO: shuffle or not?
                        shuffle=False,
                        batch_size=batch_size,
                    ),
                    total=int(trip_dataset.n_samples / np.max([batch_size, 1])),
                    position=1,
                    leave=False,
                )
            else:
                inner_range = trip_dataset.iter_batch(shuffle=False, batch_size=batch_size)

            for batch_nb, (
                item_batch,
                basket_batch,
                customer_batch,
                week_batch,
                price_batch,
                av_matrix_batch,
            ) in enumerate(inner_range):
                self.callbacks.on_train_batch_begin(batch_nb)

                neg_loglikelihood = self.train_step(
                    item_batch=item_batch,
                    basket_batch=basket_batch,
                    customer_batch=customer_batch,
                    week_batch=week_batch,
                    price_batch=price_batch,
                    av_matrix_batch=av_matrix_batch,
                )
                train_logs["train_loss"].append(neg_loglikelihood)
                temps_logs = {k: tf.reduce_mean(v) for k, v in train_logs.items()}
                self.callbacks.on_train_batch_end(batch_nb, logs=temps_logs)

                # Optimization Steps
                epoch_losses.append(neg_loglikelihood)

                if verbose > 0:
                    inner_range.set_description(
                        f"Epoch Negative-LogLikeliHood: {np.sum(epoch_losses):.4f}"
                    )

            # Take into account the fact that the last batch may have a
            # different length for the computation of the epoch loss.
            if batch_size != -1:
                last_batch_size = len(item_batch)
                coefficients = tf.concat(
                    [tf.ones(len(epoch_losses) - 1) * batch_size, [last_batch_size]], axis=0
                )
                epoch_losses = tf.multiply(epoch_losses, coefficients)
                epoch_loss = tf.reduce_sum(epoch_losses) / trip_dataset.n_samples
            else:
                epoch_loss = tf.reduce_mean(epoch_losses)

            history["train_loss"].append(epoch_loss)
            history["norm_grads"].append(self.norm_grads)
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
                test_losses = []
                for batch_nb, (
                    item_batch,
                    basket_batch,
                    customer_batch,
                    week_batch,
                    price_batch,
                    av_matrix_batch,
                ) in enumerate(val_dataset.iter_batch(shuffle=False, batch_size=batch_size)):
                    self.callbacks.on_batch_begin(batch_nb)
                    self.callbacks.on_test_batch_begin(batch_nb)

                    test_losses.append(
                        self.batch_predict(
                            item_batch=item_batch,
                            basket_batch=basket_batch,
                            customer_batch=customer_batch,
                            week_batch=week_batch,
                            price_batch=price_batch,
                            av_matrix_batch=av_matrix_batch,
                        )[0]
                    )
                    val_logs["val_loss"].append(test_losses[-1])
                    temps_logs = {k: tf.reduce_mean(v) for k, v in val_logs.items()}
                    self.callbacks.on_test_batch_end(batch_nb, logs=temps_logs)

                test_loss = tf.reduce_mean(test_losses)
                if verbose > 1:
                    print("Test Negative-LogLikelihood:", test_loss.numpy())
                    desc += f", Test Loss {np.round(test_loss.numpy(), 4)}"
                history["test_loss"] = history.get("test_loss", []) + [test_loss.numpy()]
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
        batch_size: int = -1,
    ) -> tf.Tensor:
        """Evaluate the model for each trip (unordered basket) in the dataset.

        Predicts the probabilities according to the model and computes the loss
        from the actual unordered baskets distribution.

        Parameters
        ----------
        trip_dataset: TripDataset
            Dataset on which to apply to prediction
        batch_size: int, optional
            Batch size to set, by default -1 (the whole dataset)

        Returns
        -------
        batch_loss: tf.Tensor
            Value of the loss for the batch,
            Shape must be (1,)
        """
        batch_losses = []
        for (
            item_batch,
            basket_batch,
            customer_batch,
            week_batch,
            price_batch,
            av_matrix_batch,
        ) in trip_dataset.iter_batch(shuffle=False, batch_size=batch_size):
            loss = self.batch_predict(
                item_batch=item_batch,
                basket_batch=basket_batch,
                customer_batch=customer_batch,
                week_batch=week_batch,
                price_batch=price_batch,
                av_matrix_batch=av_matrix_batch,
            )[0]
            batch_losses.append(loss)

        # Take into account the fact that the last batch may have a
        # different length for the computation of the epoch loss.
        if batch_size != -1:
            last_batch_size = len(item_batch)
            coefficients = tf.concat(
                [tf.ones(len(batch_losses) - 1) * batch_size, [last_batch_size]], axis=0
            )
            batch_losses = tf.multiply(batch_losses, coefficients)
            return tf.reduce_sum(batch_losses) / trip_dataset.n_samples

        # If batch_size == -1 (the whole dataset)
        return tf.reduce_mean(batch_losses)

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
            stage=params["stage"],
            optimizer=params["optimizer_name"],
            lr=params["lr"],
            epochs=params["epochs"],
            batch_size=params["batch_size"],
        )

        # Instantiate manually the model
        model.n_items = params["n_items"]
        model.n_customers = params["n_customers"]
        model.latent_sizes = params["latent_sizes"]
        model.n_negative_samples = params["n_negative_samples"]
        model.instantiated = params["instantiated"]

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
            # Then the paths to the saved beta, mu, gamma and delta should also exist
            # (stage 2 or 3)
            model.lambda_ = tf.Variable(np.load(lambda_path), trainable=True, name="lambda")
            model.beta = tf.Variable(
                np.load(os.path.join(path, "beta.npy")), trainable=True, name="beta"
            )
            model.mu = tf.Variable(np.load(os.path.join(path, "mu.npy")), trainable=True, name="mu")
            model.gamma = tf.Variable(
                np.load(os.path.join(path, "gamma.npy")), trainable=True, name="gamma"
            )
            model.delta = tf.Variable(
                np.load(os.path.join(path, "delta.npy")), trainable=True, name="delta"
            )

        return model
