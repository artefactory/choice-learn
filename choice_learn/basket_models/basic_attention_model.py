"""Implementation of an attention-based model for item recommendation."""

import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tqdm

from ..tf_ops import softmax_with_availabilities
from .dataset import TripDataset


class AttentionBasedContextEmbedding:
    """
    Class for the attention-based model.

    Wang, Shoujin, Liang Hu, Longbing Cao, Xiaoshui Huang, Defu Lian, and Wei Liu.
    "Attention-based transactional context embedding for next-item recommendation."
    In Proceedings of the AAAI conference on artificial intelligence, vol. 32, no. 1. 2018.
    """

    def __init__(
        self,
        epochs,
        lr,
        embedding_dim,
        n_negative_samples,
        batch_size: int = 50,
        optimizer: str = "Adam",
    ) -> None:
        """Initialize the model with hyperparameters.

        Parameters
        ----------
            epochs : int
                Number of training epochs.
            lr : float
                Learning rate for the optimizer.
            embedding_dim : int
                Dimension of the item embeddings.
            n_negative_samples : int
                Number of negative samples to use in training.
            batch_size : int
                Size of the batches for training. Default is 50.
            optimizer : str
                Optimizer to use for training. Default is "Adam".
        """
        self.instantiated = False

        self.epochs = epochs
        self.lr = lr
        self.embedding_dim = embedding_dim
        self.n_negative_samples = n_negative_samples

        self.batch_size: int = batch_size

        if optimizer.lower() == "adam":
            self.optimizer = tf.keras.optimizers.Adam(self.lr)
        else:
            print(f"Optimizer {optimizer} not implemented, switching for default Adam")
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def instantiate(
        self, n_items, negative_samples_distribution=None, use_true_nce_distribution=True
    ) -> None:
        """Initialize the model parameters.

        Parameters
        ----------
            n_items : int
                Number of unique items in the dataset.
            negative_samples_distribution : list
                Probability distribution for negative sampling.
                If None, a uniform distribution is used.
                If use_true_nce_distribution in fit() is True,
                the distribution is calculated based on the dataset.
            use_true_nce_distribution : bool
                If True, uses the true distribution of items in the dataset
        """
        self.n_items = tf.constant(n_items, dtype=tf.int32)
        if negative_samples_distribution is None:
            if n_items <= 1:
                raise ValueError("n_items must be greater than 1 to define a uniform distribution.")

            self.negative_samples_distribution = tf.constant(
                [1.0 / (n_items - 1)] * n_items, dtype=tf.float32
            )
        else:
            if len(negative_samples_distribution) != n_items:
                raise ValueError(
                    "Negative samples distribution must have the same length as n_items."
                )
            self.negative_samples_distribution = tf.constant(
                negative_samples_distribution, dtype=tf.float32
            )
        self.use_true_nce_distribution = use_true_nce_distribution

        self.Wi = tf.Variable(
            tf.random.normal((self.n_items, self.embedding_dim), stddev=0.1), name="Wi"
        )
        self.Wo = tf.Variable(
            tf.random.normal((self.n_items, self.embedding_dim), stddev=0.1), name="Wo"
        )
        self.wa = tf.Variable(tf.random.normal((self.embedding_dim,), stddev=0.1), name="wa")

        self.empty_context_embedding = tf.Variable(
            tf.random.normal((self.embedding_dim,), stddev=0.1),
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

    tf.config.run_functions_eagerly(True)

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
                [batch_size, embedding_dim] tf.Tensor
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

    tf.config.run_functions_eagerly(True)

    def score(self, context_vec: tf.Tensor, items: tf.Tensor) -> tf.Tensor:
        """Return the score of the item given the context vector.

        Parameters
        ----------
            context_vec : tf.Tensor
                [batch_size, embedding_dim] tf.Tensor
                Tensor containing the contexts vector.
            items : tf.Tensor
                [batch_size, n_items] tf.Tensor
                Tensor containing the items to score.

        Returns
        -------
            tf.Tensor
                [batch_size, n_items] tf.Tensor
                Tensor containing the scores for each item.
        """
        return tf.map_fn(
            lambda args: tf.tensordot(tf.gather(self.Wo, args[1]), args[0], axes=1),
            (context_vec, items),
            fn_output_signature=tf.float32,
        )

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

    tf.config.run_functions_eagerly(True)

    def get_negative_samples(
        self, context_items: tf.Tensor, target_items: tf.Tensor, available_items: tf.Tensor
    ) -> tf.Tensor:
        """
        Generate negative samples for the given context and target items.

        Parameters
        ----------
            context_items : tf.Tensor
                [batch_size, variable_length] tf.
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
        return tf.cast(neg_samples, tf.int32)

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

    tf.config.run_functions_eagerly(True)

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
                [batch_size, embedding_dim] tf.Tensor
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

    tf.config.run_functions_eagerly(True)

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
        pos_score = self.score(context_vec, items_batch)
        list_neg_items = self.get_negative_samples(context_batch, items_batch, available_items)
        neg_scores = tf.map_fn(
            lambda neg_items: self.score(context_vec, neg_items),
            tf.transpose(list_neg_items),
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

    def evaluate(self, dataset: tf.Tensor) -> float:
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

        return total_loss / total_samples

    def save_model(self, filepath: str, overwrite: bool = True) -> None:
        """Save the model parameters to a file.

        Parameters
        ----------
            filepath : str
                Path to the file where the model parameters will be saved.
            overwrite : bool
                If True, overwrites the file if it already exists.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving. Call fit() first.")

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
            "embedding_dim": int(self.embedding_dim),
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
        self.embedding_dim = int(data["embedding_dim"])
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
