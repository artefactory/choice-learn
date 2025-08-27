"""Base Class for basket choice modeling."""

import json
import logging
import os
import random
import time
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import tensorflow as tf
import tqdm

from ..tf_ops import softmax_with_availabilities
from .data.basket_dataset import Trip, TripDataset
from .utils.permutation import permutations


class BaseBasketModel:
    """Base class for basket choice models."""

    def __init__(
        self,
        optimizer: str = "adam",
        callbacks: Union[tf.keras.callbacks.CallbackList, None] = None,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 32,
        momentum: float = 0.0,
        grad_clip_value: Union[float, None] = None,
        weight_decay: Union[float, None] = None,
    ) -> None:
        """Initialize the model.

        Parameters
        ----------
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
        momentum: float, optional
            Momentum for the optimizer, by default 0. For SGD only
        grad_clip_value: float, optional
            Value to clip the gradient, by default None
        weight_decay: float, optional
            Weight decay, by default None
        """
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
            logging.warning(f"Optimizer {optimizer} not implemented, switching for default Adam")
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

        self._trainable_weights = []
        self.instantiated = False

    @abstractmethod
    def instantiate(
        self,
        n_items,
        n_stores,
    ) -> None:
        """Instantiate the model."""
        # Add the weights to be learned in _trainable_weights
        self._trainable_weights = []
        self.instantiated = True

    @property
    def trainable_weights(self) -> list[tf.Variable]:
        """Latent parameters of the model.

        Returns
        -------
        list[tf.Variable]
            Latent parameters of the model
        """
        return self._trainable_weights

    @property
    @abstractmethod
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
        raise ValueError("Argument 'train_iter_method' should be defined in child class.")

    @abstractmethod
    def compute_batch_utility(
        self,
        item_batch: Union[np.ndarray, tf.Tensor],
        basket_batch: np.ndarray,
        store_batch: np.ndarray,
        week_batch: np.ndarray,
        price_batch: np.ndarray,
        available_item_batch: np.ndarray,
    ) -> tf.Tensor:
        """Compute the utility of all the items in item_batch given the 5 other data.

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
        return

    # Not clear
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
            if isinstance(trip.assortment, int):
                # Then it is the assortment ID (ie its index in the attribute
                # available_items of the TripDataset), but we do not have the
                # the TripDataset as input here
                raise ValueError(
                    "The assortment ID is not enough to compute the likelihood. "
                    "Please provide the availability matrix directly (array of shape (n_items,) "
                    "indicating the availability (1) or not (0) of the products)."
                )

            return self.compute_item_likelihood(
                basket=trip.purchases,
                available_items=trip.assortment,
                store=trip.store,
                week=trip.week,
                prices=trip.prices,
                trip=None,
            )

        # Prevent unintended side effects from in-place modifications
        available_items_copy = available_items.copy()
        for basket_item in basket:
            if basket_item != -1:
                available_items_copy[basket_item] = 0.0

        # Compute the utility of all the items
        all_utilities = self.compute_batch_utility(
            # All items
            item_batch=np.arange(self.n_items),
            # For each item: same basket / store / week / prices / available items
            basket_batch=np.array([basket for _ in range(self.n_items)]),
            store_batch=np.array([store for _ in range(self.n_items)]),
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
        basket: Union[None, np.ndarray] = None,
        available_items: Union[None, np.ndarray] = None,
        store: Union[None, int] = None,
        week: Union[None, int] = None,
        prices: Union[None, np.ndarray] = None,
        trip: Union[None, Trip] = None,
    ) -> float:
        r"""Compute the utility of an ordered basket.

        o-basket-LL = \sum_{i \in \mathcal{B}} \mathbb{P}(i | \mathcal{B}[:i-1])
        Take as input directly a Trip object or separately basket, available_items,
        store, week and prices.

        Parameters
        ----------
        basket: np.ndarray or None, optional
            ordered IDs the of items already in the basket, by default None
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
        likelihood: float
            Likelihood of the ordered basket
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
                    "prices must be providedas arguments."
                )

        else:
            # Trip directly provided as an argument
            if isinstance(trip.assortment, int):
                # Then it is the assortment ID (ie its index in the attribute
                # available_items of the TripDataset), but we do not have the
                # the TripDataset as input here
                raise ValueError(
                    "The assortment ID is not enough to compute the likelihood. "
                    "Please provide the availability matrix directly (array of shape (n_items,) "
                    "indicating the availability (1) or not (0) of the products)."
                )
            return self.compute_ordered_basket_likelihood(
                basket=trip.purchases,
                available_items=trip.assortment,
                store=trip.store,
                week=trip.week,
                prices=trip.prices,
                trip=None,
            )

        # Prevent unintended side effects from in-place modifications
        available_items_copy = available_items.copy()

        ordered_basket_likelihood = 1.0
        for j in range(0, len(basket)):
            next_item_id = basket[j]

            # Compute the likelihood of the j-th item of the basket
            ordered_basket_likelihood *= self.compute_item_likelihood(
                basket=basket[:j],
                available_items=available_items_copy,
                store=store,
                week=week,
                prices=prices,
            )[next_item_id].numpy()

            # This item is not available anymore
            available_items_copy[next_item_id] = 0

        return ordered_basket_likelihood

    # Check the 0-exit-item functionment
    # @tf.function  # TODO: make it work with tf.function
    def compute_basket_likelihood(
        self,
        basket: Union[None, np.ndarray] = None,
        available_items: Union[None, np.ndarray] = None,
        store: Union[None, int] = None,
        week: Union[None, int] = None,
        prices: Union[None, np.ndarray] = None,
        trip: Union[None, Trip] = None,
        n_permutations: int = 1,
        verbose: int = 0,
    ) -> float:
        r"""Compute the utility of an (unordered) basket.

        basket-LL = \sum_{perm} \sum_{i \in \mathcal{B}_{perm}}
                                \mathbb{P}(i|\mathcal{B}_{perm}[:i-1])
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
            if isinstance(trip.assortment, int):
                # Then it is the assortment ID (ie its index in the attribute
                # available_items of the TripDataset), but we do not have the
                # the TripDataset as input here
                raise ValueError(
                    "The assortment ID is not enough to compute the likelihood. "
                    "Please provide the availability matrix directly (array of shape (n_items,) "
                    "indicating the availability (1) or not (0) of the products)."
                )
            return self.compute_basket_likelihood(
                basket=trip.purchases,
                available_items=trip.assortment,
                store=trip.store,
                week=trip.week,
                prices=trip.prices,
                trip=None,
                n_permutations=n_permutations,
                verbose=verbose,
            )

        if verbose > 0:
            print(
                f"Nb of items to be permuted = basket size - 1 = {len(basket) - 1}",
                f"Nb of permutations = {len(basket) - 1}!",
            )

        # Permute all the items in the basket except the last one (the checkout item)
        permutation_list = list(permutations(range(len(basket) - 1)))
        total_n_permutations = len(permutation_list)  # = n!

        # Limit the number of permutations to n!
        if n_permutations <= total_n_permutations:
            permutation_list = random.sample(permutation_list, n_permutations)
        else:
            logging.warning(
                "Warning: n_permutations > n! (all permutations). \
                Taking all permutations instead of n_permutations"
            )

        return (
            np.mean(
                [
                    self.compute_ordered_basket_likelihood(
                        # The last item should always be the checkout item 0
                        basket=[basket[i] for i in permutation] + [0],
                        available_items=available_items,
                        store=store,
                        week=week,
                        prices=prices,
                    )
                    for permutation in permutation_list
                ]
            )
            * total_n_permutations
        )  # Rescale the mean to the total number of permutations

    @abstractmethod
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
        batch_loss = 0.0
        loglikelihood = 0.0
        return batch_loss, loglikelihood

    @tf.function  # Graph mode
    def train_step(
        self,
        item_batch: np.ndarray,
        basket_batch: np.ndarray,
        future_batch: np.ndarray,
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
        future_batch: np.ndarray
            Batch of items to be purchased in the future (ID of items not yet in the
            basket) (arrays) for each purchased item
            Shape must be (batch_size, max_basket_size)
        store_batch: np.ndarray
            Batch of store ids (integers) for each purchased item
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
            (0: no print, 1: print)

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

            if verbose > 0:
                inner_range = tqdm.tqdm(
                    trip_dataset.iter_batch(
                        shuffle=True, batch_size=batch_size, data_method=self.train_iter_method
                    ),
                    total=int(trip_dataset.n_samples / np.max([batch_size, 1])),
                    position=1,
                    leave=False,
                )
            else:
                inner_range = trip_dataset.iter_batch(
                    shuffle=True, batch_size=batch_size, data_method=self.train_iter_method
                )

            for batch_nb, (
                item_batch,
                basket_batch,
                future_batch,
                store_batch,
                week_batch,
                price_batch,
                available_item_batch,
            ) in enumerate(inner_range):
                self.callbacks.on_train_batch_begin(batch_nb)

                batch_loss = self.train_step(
                    item_batch=item_batch,
                    basket_batch=basket_batch,
                    future_batch=future_batch,
                    store_batch=store_batch,
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
                    store_batch,
                    week_batch,
                    price_batch,
                    available_item_batch,
                ) in enumerate(
                    val_dataset.iter_batch(
                        shuffle=True, batch_size=batch_size, data_method=self.train_iter_method
                    )
                ):
                    self.callbacks.on_batch_begin(batch_nb)
                    self.callbacks.on_test_batch_begin(batch_nb)

                    val_losses.append(
                        self.compute_batch_loss(
                            item_batch=item_batch,
                            basket_batch=basket_batch,
                            future_batch=future_batch,
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
        epsilon_eval: float = 1e-9,
    ) -> tf.Tensor:
        r"""Evaluate the model for each trip (unordered basket) in the dataset.

        Predicts the probabilities according to the model and then computes the
        mean negative log-likelihood (nll) for the dataset
        NLL = \sum_{\mathcal{B} \in \mathcal{D}}
                    \sum_{i \in \mathcal{B}} \mathbb{P}(i | \mathcal{B} \setminus i)

        Parameters
        ----------
        trip_dataset: TripDataset
            Dataset on which to apply to prediction
        n_permutations: int, optional
            Number of permutations to average over, by default 1
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
        n_evals = 0

        inner_range = trip_dataset.iter_batch(
            shuffle=False, batch_size=batch_size, data_method="aleacarta"
        )
        for (
            item_batch,
            basket_batch,
            _,
            store_batch,
            week_batch,
            price_batch,
            available_item_batch,
        ) in inner_range:
            # Sum of the log-likelihoods of all the baskets in the batch
            basket_batch = [basket[basket != -1] for basket in basket_batch]
            sum_loglikelihoods += np.sum(
                np.log(
                    [
                        self.compute_item_likelihood(
                            basket=basket,
                            available_items=available_items,
                            store=store,
                            week=week,
                            prices=prices,
                        )[item]
                        + epsilon_eval
                        for basket, item, available_items, store, week, prices in zip(
                            basket_batch,
                            item_batch,
                            available_item_batch,
                            store_batch,
                            week_batch,
                            price_batch,
                        )
                    ]
                )
            )
            n_evals += len(item_batch)

        # Predicted mean negative log-likelihood over all the batches
        return -1 * sum_loglikelihoods / n_evals

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

    # Fine how-to link to _trainable_weights
    def _load_weights(self, directory):
        """Load all the .npy weights within a directory.

        Parameters
        ----------
        directory: path
            path of the directory where the weights to be loaded are.
        """
        for file in os.listdir(directory):
            if file.endswith(".npy"):
                weight_name = file.split(".")[0]
                setattr(
                    self,
                    weight_name,
                    tf.Variable(
                        np.load(os.path.join(directory, file)), trainable=True, name=weight_name
                    ),
                )

    @classmethod
    def load_model(cls, path: str) -> object:
        """Load a model previously saved with save_model().

        Parameters
        ----------
        path: str
            path to the folder where the saved model files are

        Returns
        -------
        BasketModel
            Loaded BasketModel
        """
        import inspect

        # Load parameters
        params = json.load(open(os.path.join(path, "params.json")))

        init_params = {}
        non_init_params = {}
        for key, val in params.items():
            if key in inspect.signature(cls.__init__).parameters.keys():
                init_params[key] = val
            else:
                non_init_params[key] = val

        # Initialize model
        model = cls(**init_params)

        # Set non-init parameters
        for key, val in non_init_params.items():
            setattr(model, key, val)

        # Load weights
        model._load_weights(path)

        return model
