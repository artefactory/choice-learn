"""Implementation of an attention-based model for item recommendation."""

from typing import Union

import numpy as np
import tensorflow as tf

from ..tf_ops import NoiseConstrastiveEstimation
from .base_basket_model import BaseBasketModel
from .data.basket_dataset import TripDataset


class AttentionBasedContextEmbedding(BaseBasketModel):
    """Class for the attention-based model.

    Wang, Shoujin, Liang Hu, Longbing Cao, Xiaoshui Huang, Defu Lian,
    and Wei Liu. "Attention-based transactional context embedding for
    next-item recommendation." In Proceedings of the AAAI conference on
    artificial intelligence, vol. 32, no. 1. 2018.
    """

    def __init__(
        self,
        latent_size: int = 4,
        n_negative_samples: int = 2,
        nce_distribution="natural",
        optimizer: str = "adam",
        callbacks: Union[tf.keras.callbacks.CallbackList, None] = None,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 32,
        grad_clip_value: Union[float, None] = None,
        weight_decay: Union[float, None] = None,
        momentum: float = 0.0,
        **kwargs,
    ) -> None:
        """Initialize the model with hyperparameters.

        Parameters
        ----------
        epochs : int
            Number of training epochs.
        lr : float
            Learning rate for the optimizer.
        latent_size : int
            Size of the item embeddings.
        n_negative_samples : int
            Number of negative samples to use in training.
        batch_size : int
            Size of the batches for training. Default is 50.
        optimizer : str
            Optimizer to use for training. Default is "Adam".
        nce_distribution: str
            Items distribution to be used to compute the NCE Loss
            Currentlry available: 'natural' to estimate the distribution
            from the train dataset and 'uniform' where all items have the
            same disitrbution, 1/n_items. Default is 'natural'.
        """
        self.instantiated = False

        self.latent_size = latent_size
        self.n_negative_samples = n_negative_samples
        self.nce_distribution = nce_distribution

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
    ) -> None:
        """Initialize the model parameters.

        Parameters
        ----------
        n_items : int
            Number of unique items in the dataset.
        """
        self.n_items = n_items

        self.Wi = tf.Variable(
            tf.random.normal((self.n_items, self.latent_size), stddev=0.1, seed=42),
            name="Wi",
        )
        self.Wo = tf.Variable(
            tf.random.normal((self.n_items, self.latent_size), stddev=0.1, seed=42),
            name="Wo",
        )
        self.wa = tf.Variable(
            tf.random.normal((self.latent_size,), stddev=0.1, seed=42), name="wa"
        )

        self.empty_context_embedding = tf.Variable(
            tf.random.normal((self.latent_size,), stddev=0.1, seed=42),
            name="empty_context_embedding",
        )

        self.loss = NoiseConstrastiveEstimation()
        self.is_trained = False
        self.instantiated = True

    @property
    def trainable_weights(self):
        """Return the trainable weights of the model.

        Returns
        -------
            list
                List of trainable weights (Wi, wa, Wo).
        """
        return [self.Wi, self.wa, self.Wo, self.empty_context_embedding]

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

    def embed_context(self, context_items: tf.Tensor) -> tf.Tensor:
        """Return the context embedding matrix.

        Parameters
        ----------
            context_items : tf.Tensor
                [batch_size, variable_length] tf.RaggedTensor
                Tensor containing the list of the context items.

        Returns
        -------
            tf.Tensor
                [batch_size, latent_size] tf.Tensor
                Tensor containing the matrix of contexts embeddings.
        """
        context_emb = tf.gather(self.Wi, tf.cast(context_items, tf.int32), axis=0)
        return tf.map_fn(
            lambda x: tf.cond(
                tf.equal(tf.shape(x)[0], 0),
                lambda: self.empty_context_embedding,
                lambda: tf.reduce_sum(
                    tf.transpose(x) * tf.nn.softmax(tf.tensordot(x, self.wa, axes=1)),
                    axis=1,
                ),
            ),
            context_emb,
            fn_output_signature=tf.float32,
        )

    def compute_batch_utility(
        self,
        item_batch: Union[np.ndarray, tf.Tensor],
        basket_batch: np.ndarray,
        store_batch: np.ndarray,
        week_batch: np.ndarray,
        price_batch: np.ndarray,
        available_item_batch: np.ndarray,
    ) -> tf.Tensor:
        """Compute the utility of all the items in item_batch given the items
        in basket_batch.

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
        _ = store_batch
        _ = price_batch
        _ = week_batch
        _ = available_item_batch
        basket_batch_ragged = tf.cast(
            tf.ragged.boolean_mask(basket_batch, basket_batch != -1),
            dtype=tf.int32,
        )
        context_embedding = self.embed_context(basket_batch_ragged)
        return tf.reduce_sum(
            tf.multiply(
                tf.gather(self.Wo, tf.cast(item_batch, tf.int32)), context_embedding
            ),
            axis=1,
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

    def _get_items_frequencies(self, dataset: TripDataset) -> tf.Tensor:
        """Count the occurrences of each item in the dataset.

        Parameters
        ----------
            dataset : TripDataset
                Dataset containing the baskets.

        Returns
        -------
            tf.Tensor
                Tensor containing the count of each item.
        """
        item_counts = np.zeros(self.n_items, dtype=np.int32)
        for trip in dataset.trips:
            for item in trip.purchases:
                item_counts[item] += 1
        items_distribution = item_counts / item_counts.sum()
        return tf.constant(items_distribution, dtype=tf.float32)

    def compute_batch_loss(
        self,
        item_batch: np.ndarray,
        basket_batch: np.ndarray,
        future_batch: np.ndarray,
        store_batch: np.ndarray,
        week_batch: np.ndarray,
        price_batch: np.ndarray,
        available_item_batch: np.ndarray,
        user_batch=None,
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
        negative_samples = tf.transpose(
            tf.stack(
                [
                    self.get_negative_samples(
                        available_items=available_item_batch[idx],
                        purchased_items=basket_batch[idx],
                        next_item=item_batch[idx],
                        n_samples=self.n_negative_samples,
                    )
                    for idx in range(len(item_batch))
                ],
                axis=0,
            ),
        )
        pos_score = self.compute_batch_utility(
            item_batch,
            basket_batch,
            store_batch,
            week_batch,
            price_batch,
            available_item_batch,
        )
        neg_scores = tf.map_fn(
            lambda neg_items: self.compute_batch_utility(
                item_batch=neg_items,
                basket_batch=basket_batch,
                store_batch=store_batch,
                week_batch=week_batch,
                price_batch=price_batch,
                available_item_batch=available_item_batch,
            ),
            negative_samples,
            fn_output_signature=tf.float32,
        )
        # neg_scores = tf.reshape(neg_scores, (-1, self.n_negative_samples))
        return (
            self.loss(
                logit_true=pos_score,
                logit_negative=tf.transpose(neg_scores),
                freq_true=tf.gather(
                    self.negative_samples_distribution, tf.cast(item_batch, tf.int32)
                ),
                freq_negative=tf.gather(
                    self.negative_samples_distribution,
                    tf.cast(tf.transpose(negative_samples), tf.int32),
                ),
            ),
            1e-10,
        )

    def fit(
        self,
        trip_dataset: TripDataset,
        val_dataset: Union[TripDataset, None] = None,
        verbose: int = 0,
    ) -> None:
        """Trains the model for a specified number of epochs.

        Parameters
        ----------
            dataset : TripDataset
                Dataset of baskets to train the model on.
        """
        if not self.instantiated:
            self.instantiate(n_items=trip_dataset.n_items)

        if not isinstance(trip_dataset, TripDataset):
            raise TypeError("Dataset must be a TripDataset.")

        if (
            max([len(trip.purchases) for trip in trip_dataset.trips])
            + self.n_negative_samples
            > self.n_items
        ):
            raise ValueError(
                "The number of items in the dataset is less than the number of negative samples."
            )

        if self.nce_distribution == "natural":
            self.negative_samples_distribution = self._get_items_frequencies(
                trip_dataset
            )
        else:
            self.negative_samples_distribution = (1 / trip_dataset.n_items) * np.ones(
                (trip_dataset.n_items,)
            )

        history = super().fit(
            trip_dataset=trip_dataset, val_dataset=val_dataset, verbose=verbose
        )

        self.is_trained = True

        return history
