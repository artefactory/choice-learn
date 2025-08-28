"""Implementation of an attention-based model for item recommendation."""

from typing import Union

import numpy as np
import tensorflow as tf
import tqdm

from ..tf_ops import softmax_with_availabilities
from .base_basket_model import BaseBasketModel
from .data.basket_dataset import TripDataset


class AttentionBasedContextEmbedding(BaseBasketModel):
    """
    Class for the attention-based model.

    Wang, Shoujin, Liang Hu, Longbing Cao, Xiaoshui Huang, Defu Lian, and Wei Liu.
    "Attention-based transactional context embedding for next-item recommendation."
    In Proceedings of the AAAI conference on artificial intelligence, vol. 32, no. 1. 2018.
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
        self.n_items = tf.constant(n_items, dtype=tf.int32)

        self.Wi = tf.Variable(
            tf.random.normal((self.n_items, self.latent_size), stddev=0.1), name="Wi"
        )
        self.Wo = tf.Variable(
            tf.random.normal((self.n_items, self.latent_size), stddev=0.1), name="Wo"
        )
        self.wa = tf.Variable(tf.random.normal((self.latent_size,), stddev=0.1), name="wa")

        self.empty_context_embedding = tf.Variable(
            tf.random.normal((self.latent_size,), stddev=0.1),
            name="empty_context_embedding",
        )

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
        context_emb = tf.gather(self.Wi, context_items, axis=0)
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
        """Compute the utility of all the items in item_batch given the items in basket_batch.

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
        context_embedding = self.embed_context(basket_batch)
        return tf.reduce_sum(tf.multiply(tf.gather(self.Wo, item_batch), context_embedding), axis=1)

    def positive_samples_probability(
        self, pos_score: tf.Tensor, target_items: tf.Tensor
    ) -> tf.Tensor:
        """Calculate the probability of positive samples.

        Parameters
        ----------
            pos_score : tf.Tensor
                [batch_size,] tf.Tensor
                Tensor containing the scores of positive samples.
            target_items : tf.Tensor
                [batch_size,] tf.Tensor
                Tensor containing the target items.

        Returns
        -------
            tf.Tensor
                [batch_size,] tf.Tensor
                Tensor containing the probabilities of positive samples.
        """
        q_dist = tf.gather(self.negative_samples_distribution, target_items)
        return 1 / (1 + self.n_negative_samples * q_dist * tf.exp(-pos_score))

    def negative_samples_probability(
        self, neg_score: tf.Tensor, target_items: tf.Tensor
    ) -> tf.Tensor:
        """Calculate the probability of negative samples.

        Parameters
        ----------
            neg_score : tf.Tensor
                [batch_size, n_negative_samples] tf.Tensor
                Tensor containing the scores of negative samples.
            target_items : tf.Tensor
                [batch_size, n_negative_samples] tf.Tensor
                Tensor containing the target items.

        Returns
        -------
            tf.Tensor
                [batch_size, n_negative_samples] tf.Tensor
                Tensor containing the probabilities of negative samples.
        """
        q_dist = tf.gather(self.negative_samples_distribution, target_items)
        return 1 - (1 / (1 + self.n_negative_samples * q_dist * tf.exp(-neg_score)))

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

    '''def get_negative_samples(
        self, context_items: tf.Tensor, target_items: tf.Tensor, available_items: tf.Tensor
    ) -> tf.Tensor:
        """
        Generate negative samples for the given context and target items.

        Parameters
        ----------
            context_items : tf.Tensor
                [batch_size, variable_length]
                Tensor contenant les items de contexte (batch_size, variable_length).
            target_items : tf.Tensor
                [batch_size,] tf.Tensor
                Tensor contenant les items cibles (batch_size,).
            available_items : tf.Tensor
                [batch_size, n_items] tf.
                Tensor binaire (batch_size, n_items) indiquant les items disponibles.

        Returns
        -------
            tf.Tensor
                [batch_size, n_negative_samples] tf.Tensor
                Tensor contenant les échantillons négatifs pour chaque contexte.
        """
        target_items_exp = tf.expand_dims(target_items, axis=1)
        forbidden_items = tf.concat([context_items, target_items_exp], axis=1)
        if isinstance(forbidden_items, tf.RaggedTensor):
            forbidden_items = forbidden_items.to_tensor(default_value=-1)
        item_range = tf.range(self.n_items)
        item_range = tf.reshape(item_range, (1, 1, self.n_items))
        forbidden_items_exp = tf.expand_dims(forbidden_items, axis=-1)
        mask = tf.reduce_any(tf.equal(forbidden_items_exp, item_range), axis=1)
        candidates_mask = tf.logical_and(~mask, available_items > 0)

        def sample_negatives(candidates):
            indices = tf.where(candidates)[:, 0]
            n_candidates = tf.shape(indices)[0]
            shuffled = tf.random.shuffle(indices)
            pad_value = tf.cond(
                n_candidates > 0,
                lambda: tf.cast(shuffled[0], tf.int64),
                lambda: tf.constant(0, dtype=tf.int64),
            )
            return tf.cond(
                n_candidates >= self.n_negative_samples,
                lambda: shuffled[: self.n_negative_samples],
                lambda: tf.pad(
                    shuffled,
                    [[0, self.n_negative_samples - n_candidates]],
                    constant_values=pad_value,
                ),
            )

        neg_samples = tf.map_fn(
            sample_negatives,
            candidates_mask,
            fn_output_signature=tf.TensorSpec([self.n_negative_samples], dtype=tf.int64),
        )
        return tf.cast(neg_samples, tf.int32)'''

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

    def nce_loss(
        self,
        pos_score: tf.Tensor,
        target_items: tf.Tensor,
        list_neg_items: tf.Tensor,
        neg_scores: tf.Tensor,
    ) -> tf.Tensor:
        """Calculate the loss using Noise Contrastive Estimation (NCE).

        Parameters
        ----------
            pos_score : tf.Tensor
                [batch_size,] tf.Tensor
                Tensor containing the scores of positive samples.
            target_items : tf.Tensor
                [batch_size,] tf.Tensor
                Tensor containing the target items.
            list_neg_items : tf.Tensor
                [batch_size, n_negative_samples] tf.Tensor
                Tensor containing the negative samples.
            neg_scores : tf.Tensor
                [batch_size, n_negative_samples] tf.Tensor
                Tensor containing the scores of negative samples.

        Returns
        -------
            tf.Tensor
                [batch_size,] tf.Tensor
                Tensor containing the NCE loss.
        """
        loss = -tf.math.log(self.positive_samples_probability(pos_score, target_items))
        for i in range(len(neg_scores)):
            loss -= tf.math.log(
                self.negative_samples_probability(
                    neg_scores[i], tf.gather(list_neg_items, i, axis=1)
                )
            )
        return loss

    def negative_log_likelihood_loss(
        self, context_vec: tf.Tensor, target_items: tf.Tensor
    ) -> tf.Tensor:
        """Calculate the loss using tf.keras.losses.sparse_categorical_crossentropy.

        Parameters
        ----------
            context_vec : tf.Tensor
                [batch_size, latent_size] tf.Tensor
                Tensor containing the context vector for the batch.
            target_items : tf.Tensor
                [batch_size,] tf.Tensor
                Tensor containing the target items to predict.

        Returns
        -------
            tf.Tensor
                [batch_size,] tf.Tensor
                Tensor containing the negative log likelihood loss.
        """
        logits = tf.matmul(context_vec, self.Wo, transpose_b=True)
        return tf.keras.losses.sparse_categorical_crossentropy(
            target_items, logits, from_logits=True
        )

    def compute_batch_loss(
        self,
        context_batch: tf.Tensor,
        items_batch: tf.Tensor,
        available_items: tf.Tensor,
    ) -> tf.Tensor:
        """Calculate the loss for a batch of baskets.

        Parameters
        ----------
            context_batch : tf.Tensor
                [batch_size, variable_length] tf.Tensor
                Tensor containing the context items for the batch.
            items_batch : tf.Tensor
                [batch_size,] tf.Tensor
                Tensor containing the target items for the batch.

        Returns
        -------
            tf.Tensor
                Tensor containing the total loss for the batch.
        """
        context_vec = self.embed_context(context_batch)
        pos_score = self.compute_batch_utility(context_vec, items_batch)
        list_neg_items = self.get_negative_samples(context_batch, items_batch, available_items)
        negative_samples = self.get_negative_samples(
            available_items=available_items,
            purchased_items=context_batch,
            next_item=items_batch,
            n_samples=self.n_negative_samples,
        )
        neg_scores = tf.map_fn(
            lambda neg_items: self.compute_batch_utility(context_vec, neg_items),
            negative_samples,
            fn_output_signature=tf.float32,
        )
        return tf.reduce_sum(
            self.nce_loss(
                pos_score=pos_score,
                target_items=items_batch,
                list_neg_items=list_neg_items,
                neg_scores=neg_scores,
            )
        )

    @tf.function
    def train_step(self, context_batch, items_batch, available_items) -> tf.Tensor:
        """Perform a single training step on the batch of baskets.

        Parameters
        ----------
            context_batch : tf.Tensor
                [batch_size, variable_length] tf.Tensor
                Tensor containing the batch of baskets.
            items_batch : tf.Tensor
                [batch_size,] tf.Tensor
                Tensor containing the target items for the batch.
            available_items : tf.Tensor
                [batch_size, n_items] tf.Tensor
                Tensor indicating the available items for the batch.

        Returns
        -------
            tf.Tensor
                Tensor containing the total loss for the batch.
        """
        with tf.GradientTape() as tape:
            loss = self.compute_batch_loss(context_batch, items_batch, available_items)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return loss

    def predict(self, context_items: tf.Tensor, available_items: np.ndarray) -> np.ndarray:
        """
        Predicts the item probabilities given the context items.

        Parameters
        ----------
            context_items : tf.Tensor
                [batch_size, variable_length] tf.Tensor or tf.RaggedTensor
                Tensor containing the context items for prediction.
            available_items : np.ndarray
                [bacth_size,] np.ndarray
                Numpy array indicating the available items for prediction.

        Returns
        -------
            np.ndarray
                [batch_size, n_items] np.ndarray
                Numpy array containing the predicted probabilities for each item.
        """
        if not self.instantiated:
            self.instantiate(n_items=len(available_items))

        context_vec = self.embed_context(context_items)
        scores = tf.matmul(context_vec, self.Wo, transpose_b=True)
        avail_mask = tf.convert_to_tensor(available_items, dtype=scores.dtype)
        probs = softmax_with_availabilities(
            items_logit_by_choice=scores,
            available_items_by_choice=avail_mask,
            axis=-1,
            normalize_exit=False,
            eps=1e-5,
        )

        return probs.numpy()

    def fit(self, dataset: TripDataset) -> None:
        """Trains the model for a specified number of epochs.

        Parameters
        ----------
            dataset : TripDataset
                Dataset of baskets to train the model on.
        """
        if not self.instantiated:
            self.instantiate(n_items=len(dataset.trips[0].purchases))

        if not isinstance(dataset, TripDataset):
            raise TypeError("Dataset must be a TripDataset.")

        if (
            max([len(trip.purchases) for trip in dataset.trips]) + self.n_negative_samples
            > self.n_items
        ):
            raise ValueError(
                "The number of items in the dataset is less than the number of negative samples."
            )

        history = {"train_loss": []}

        if self.use_true_nce_distribution:
            self.negative_samples_distribution = self._get_items_frequencies(dataset)

        epochs_range = tqdm.trange(self.epochs, desc="Training Epochs")
        for _ in epochs_range:
            epoch_loss = 0

            for batch in dataset.iter_batch(
                batch_size=self.batch_size, shuffle=True, data_method="aleacarta"
            ):
                ragged_batch = tf.ragged.constant(
                    [row[row != -1] for row in batch[1]], dtype=tf.int32
                )
                target_items = tf.constant(batch[0], dtype=tf.int32)
                available_items = tf.constant(batch[6], dtype=tf.int32)
                loss = self.train_step(ragged_batch, target_items, available_items)
                epoch_loss += loss

            float_loss = float(epoch_loss.numpy())
            history["train_loss"].append(float_loss)
            epochs_range.set_postfix({"epoch_loss": float_loss})

        self.is_trained = True

        return history

    '''def evaluate(self, dataset: tf.Tensor) -> float:
        """Evaluate the model on the given dataset and returns the average loss.

        Parameters
        ----------
            dataset : TripDataset
                Tensor containing the dataset of baskets to evaluate.

        Returns
        -------
            float
                Average loss over the dataset.
        """
        total_loss = 0.0
        total_samples = 0

        for batch in dataset.iter_batch(
            batch_size=self.batch_size, shuffle=False, data_method="aleacarta"
        ):
            ragged_batch = tf.ragged.constant([row[row != -1] for row in batch[1]], dtype=tf.int32)
            target_items = tf.constant(batch[0], dtype=tf.int32)
            context_vec = self.embed_context(ragged_batch)
            batch_loss = self.negative_log_likelihood_loss(context_vec, target_items)
            total_loss += tf.reduce_sum(batch_loss).numpy()
            total_samples += batch_loss.shape[0]

        return total_loss / total_samples'''

    '''def save_model(self, filepath: str, overwrite: bool = True) -> None:
        """Save the model parameters to a file.

        Parameters
        ----------
            filepath : str
                Path to the file where the model parameters will be saved.
            overwrite : bool
                If True, overwrites the file if it already exists.
        """
        if os.path.exists(filepath):
            if overwrite:
                Path(filepath).unlink()
            else:
                raise FileExistsError(f"Model file {filepath} already exists.")

        base = Path(filepath)
        base_no_suffix = base.with_suffix("")  # Removes the extension

        wi_file = str(base_no_suffix) + "_Wi.npy"
        wo_file = str(base_no_suffix) + "_Wo.npy"
        wa_file = str(base_no_suffix) + "_wa.npy"
        empty_context_file = str(base_no_suffix) + "_empty_context_embedding.npy"

        # Save weights as .npy files
        np.save(wi_file, self.Wi.numpy())
        np.save(wo_file, self.Wo.numpy())
        np.save(wa_file, self.wa.numpy())
        np.save(empty_context_file, self.empty_context_embedding.numpy())

        data = {
            "n_items": int(self.n_items),
            "latent_size": int(self.latent_size),
            "n_negative_samples": int(self.n_negative_samples),
            "lr": float(self.lr),
            "epochs": int(self.epochs),
            "optimizer": self.optimizer.get_config(),
            "batch_size": int(self.batch_size),
            "negative_samples_distribution": self.negative_samples_distribution.numpy().tolist()
            if hasattr(self.negative_samples_distribution, "numpy")
            else list(self.negative_samples_distribution),
            "Wi_file": wi_file,
            "Wo_file": wo_file,
            "wa_file": wa_file,
            "empty_context_embedding_file": empty_context_file,
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

    def load_model(self, filepath: str) -> None:
        """Load the model parameters from a file.

        Parameters
        ----------
            filepath : str
                Path to the file from which the model parameters will be loaded.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} does not exist.")

        with open(filepath) as f:
            data = json.load(f)

        # Set hyperparameters and attributes
        self.n_items = int(data["n_items"])
        self.latent_size = int(data["latent_size"])
        self.n_negative_samples = int(data["n_negative_samples"])
        self.lr = float(data["lr"])
        self.epochs = int(data["epochs"])
        self.batch_size = int(data["batch_size"])
        self.negative_samples_distribution = tf.constant(
            data["negative_samples_distribution"], dtype=tf.float32
        )

        # Load weights from .npy files
        self.Wi = tf.Variable(np.load(data["Wi_file"]), name="Wi")
        self.Wo = tf.Variable(np.load(data["Wo_file"]), name="Wo")
        self.wa = tf.Variable(np.load(data["wa_file"]), name="wa")
        self.empty_context_embedding = tf.Variable(
            np.load(data["empty_context_embedding_file"]), name="empty_context_embedding"
        )

        # Re-instantiate optimizer
        if isinstance(data["optimizer"], dict) and "name" in data["optimizer"]:
            if data["optimizer"]["name"].lower() == "adam":
                self.optimizer = tf.keras.optimizers.Adam(self.lr)
            else:
                print(f"Optimizer {data['optimizer']['name']} not implemented, switching to Adam")
                self.optimizer = tf.keras.optimizers.Adam(self.lr)
        else:
            self.optimizer = tf.keras.optimizers.Adam(self.lr)

        self.is_trained = True
        self.instantiated = True
    '''
