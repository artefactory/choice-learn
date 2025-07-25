"""
Implementation of an attention-based model for item recommendation.

Cf. "Attention-Based Transactional Context Embedding for Next-Item Recommendation".
Wang et al. (2018).
"""

import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from pathlib import Path

import numpy as np
import tensorflow as tf
import tqdm

from .dataset import TripDataset


class AttentionBasedContextEmbedding:
    """Class for the attention-based model."""

    def __init__(
        self,
        epochs,
        lr,
        embedding_dim,
        n_negative_samples,
        batch_size: str = 50,
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

        self.last_n_baskets_dataset = None

    def instantiate(self, n_items, negative_samples_distribution=None) -> None:
        """Initialize the model parameters.

        Parameters
        ----------
            n_items : int
                Number of unique items in the dataset.
            negative_samples_distribution : list
                Probability distribution for negative sampling.
                If None, a uniform distribution is used.
        """
        self.n_items = tf.constant(n_items, dtype=tf.int32)
        if negative_samples_distribution is None:
            if n_items <= 1:
                raise ValueError("n_items must be greater than 1 to define a uniform distribution.")

            self.negative_samples_distribution = tf.constant(
                [1.0 / (n_items - 1 + 1)] * n_items, dtype=tf.float32
            )
        else:
            if len(negative_samples_distribution) != n_items:
                raise ValueError(
                    "Negative samples distribution must have the same length as n_items."
                )
            self.negative_samples_distribution = tf.constant(
                negative_samples_distribution, dtype=tf.float32
            )

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

    def context_embed(self, context_items: tf.Tensor) -> tf.Tensor:
        """Return the context embedding matrix.

        Parameters
        ----------
            context_items : tf.Tensor
                Tensor containing the list of the context items.

        Returns
        -------
            tf.Tensor
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

    def score(self, context_vec: tf.Tensor, items: tf.Tensor) -> tf.Tensor:
        """Return the score of the item given the context vector.

        Parameters
        ----------
            context_vec : tf.Tensor
                Tensor containing the contexts vector.
            items : tf.Tensor
                Tensor containing the items to score.

        Returns
        -------
            tf.Tensor
                Tensor containing the scores for each item.
        """
        return tf.map_fn(
            lambda args: tf.tensordot(tf.gather(self.Wo, args[1]), args[0], axes=1),
            (context_vec, items),
            fn_output_signature=tf.float32,
        )

    def proba_positive_samples(self, pos_score: tf.Tensor, target_items: tf.Tensor) -> tf.Tensor:
        """Calculate the probability of positive samples.

        Parameters
        ----------
            pos_score : tf.Tensor
                Tensor containing the scores of positive samples.
            target_items : tf.Tensor
                Tensor containing the target items.

        Returns
        -------
            tf.Tensor
                Tensor containing the probabilities of positive samples.
        """
        q_dist = tf.gather(self.negative_samples_distribution, target_items)
        return 1 / (1 + self.n_negative_samples * q_dist * tf.exp(-pos_score))

    def proba_negative_samples(self, neg_score: tf.Tensor, target_items: tf.Tensor) -> tf.Tensor:
        """Calculate the probability of negative samples.

        Parameters
        ----------
            neg_score : tf.Tensor
                Tensor containing the scores of negative samples.
            target_items : tf.Tensor
                Tensor containing the target items.

        Returns
        -------
            tf.Tensor
                Tensor containing the probabilities of negative samples.
        """
        q_dist = tf.gather(self.negative_samples_distribution, target_items)
        return 1 - (1 / (1 + self.n_negative_samples * q_dist * tf.exp(-neg_score)))

    @tf.function
    def get_negative_samples(self, context_items: tf.Tensor, target_items: tf.Tensor) -> tf.Tensor:
        """Generate negative samples for the given context and target items.

        Parameters
        ----------
            context_items : tf.Tensor
                Tensor containing the context items.
            target_items : tf.Tensor
                Tensor containing the target items.

        Returns
        -------
            tf.Tensor
                Tensor containing the negative samples for each context item.
        """
        n_items = self.n_items
        target_items_exp = tf.expand_dims(target_items, axis=1)
        forbidden_items = tf.concat([context_items, target_items_exp], axis=1)
        if isinstance(forbidden_items, tf.RaggedTensor):
            forbidden_items = forbidden_items.to_tensor(default_value=-1)
        item_range = tf.range(n_items)
        item_range = tf.reshape(item_range, (1, 1, n_items))
        forbidden_items_exp = tf.expand_dims(forbidden_items, axis=-1)
        mask = tf.reduce_any(tf.equal(forbidden_items_exp, item_range), axis=1)
        candidates_mask = ~mask

        # For each batch, get indices of allowed items
        def sample_negatives(candidates):
            indices = tf.where(candidates)[:, 0]
            n_candidates = tf.shape(indices)[0]
            # If not enough candidates, sample with replacement
            shuffled = tf.random.shuffle(indices)
            return tf.cond(
                n_candidates >= self.n_negative_samples,
                lambda: shuffled[: self.n_negative_samples],
                lambda: tf.pad(
                    shuffled,
                    [[0, self.n_negative_samples - n_candidates]],
                    constant_values=shuffled[0],
                ),
            )

        neg_samples = tf.map_fn(
            sample_negatives,
            candidates_mask,
            fn_output_signature=tf.TensorSpec([self.n_negative_samples], dtype=tf.int64),
        )
        return tf.cast(neg_samples, tf.int32)

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
            context_items : tf.Tensor
                Tensor containing the context items.
            target_items : tf.Tensor
                Tensor containing the target items to predict.

        Returns
        -------
            tf.Tensor
                Tensor containing the NCE loss.
        """
        loss = -tf.math.log(self.proba_positive_samples(pos_score, target_items))
        for i in range(len(neg_scores)):
            loss -= tf.math.log(
                self.proba_negative_samples(neg_scores[i], tf.gather(list_neg_items, i, axis=1))
            )
        return loss

    @tf.autograph.experimental.do_not_convert
    def negative_log_likelihood_loss(
        self, context_vec: tf.Tensor, target_items: tf.Tensor
    ) -> tf.Tensor:
        """Calculate the loss using a simple, naive softmax cross-entropy approach.

        Parameters
        ----------
            context_vec : tf.Tensor
                Tensor containing the context vector for the batch.
            target_items : tf.Tensor
                Tensor containing the target items to predict.

        Returns
        -------
            tf.Tensor
                Tensor containing the negative log likelihood loss.
        """
        scores_softmax = tf.map_fn(
            lambda x: tf.nn.softmax(tf.tensordot(self.Wo, x, axes=1)), context_vec
        )

        target_items_scores = tf.gather_nd(
            scores_softmax,
            tf.stack([tf.range(tf.shape(target_items)[0]), target_items], axis=1),
        )
        return -tf.math.log(target_items_scores)

    def get_batch_loss(
        self,
        context_batch: tf.Tensor,
        items_batch: tf.Tensor,
    ) -> tf.Tensor:
        """Calculate the loss for a batch of baskets.

        Parameters
        ----------
            context_batch : tf.Tensor
                Tensor containing the context items for the batch.
            items_batch : tf.Tensor
                Tensor containing the target items for the batch.

        Returns
        -------
            tf.Tensor
                Tensor containing the total loss for the batch.
        """
        context_vec = self.context_embed(context_batch)
        pos_score = self.score(context_vec, items_batch)
        list_neg_items = self.get_negative_samples(context_batch, items_batch)
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
    def train_step(self, context_batch, items_batch) -> tf.Tensor:
        """Perform a single training step on the batch of baskets.

        Parameters
        ----------
            batch : tf.Tensor
                Tensor containing the batch of baskets.

        Returns
        -------
            tf.Tensor
                Tensor containing the total loss for the batch.
        """
        with tf.GradientTape() as tape:
            loss = self.get_batch_loss(context_batch, items_batch)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return loss

    def predict(self, context_items: tf.Tensor) -> np.ndarray:
        """Predicts the item probabilities given the context items.

        Parameters
        ----------
            context_items : tf.Tensor
                Tensor containing the context items for prediction.

        Returns
        -------
            np.ndarray
                Numpy array containing the predicted probabilities for each item.
        """
        if not self.instantiated:
            raise ValueError(
                "Model must be instantiated before prediction. Call instantiate() first."
            )

        if not self.is_trained:
            raise ValueError("Model must be trained before prediction. Call fit() first.")

        context_vec = self.context_embed(context_items)
        scores = tf.tensordot(self.Wo, tf.transpose(context_vec), axes=1)
        scores = tf.linalg.set_diag(scores, tf.fill([tf.shape(scores)[0]], float("-inf")))
        return tf.nn.softmax(scores, axis=1).numpy()

    def fit(self, dataset) -> None:
        """Trains the model for a specified number of epochs.

        Parameters
        ----------
            dataset : TripDataset
                Dataset of baskets to train the model on.
            repr : bool
                If True, represents the model after training.
        """
        if not self.instantiated:
            raise ValueError(
                "Model must be instantiated before training. Call instantiate() first."
            )
        if not isinstance(dataset, TripDataset):
            raise TypeError("Dataset must be a TripDataset.")

        if not dataset:
            raise ValueError("Dataset cannot be empty.")

        if (
            max([len(trip.purchases) for trip in dataset.trips]) + self.n_negative_samples
            > self.n_items
        ):
            raise ValueError(
                "The number of items in the dataset is less than the number of negative samples."
            )

        history = {"train_loss": []}

        iterable = tqdm.trange(self.epochs, desc="Training Epochs")
        for _ in iterable:
            epoch_loss = 0

            for batch in dataset.iter_batch(
                batch_size=self.batch_size, shuffle=False, data_method="aleacarta"
            ):
                ragged_batch = tf.ragged.constant(
                    [row[row != -1] for row in batch[1]], dtype=tf.int32
                )
                target_items = tf.constant(batch[0], dtype=tf.int32)
                loss = self.train_step(ragged_batch, target_items)
                epoch_loss += loss

            float_loss = float(epoch_loss.numpy())
            history["train_loss"].append(float_loss)
            iterable.set_postfix({"epoch_loss": float_loss})

        self.is_trained = True
        self.last_n_baskets_dataset = len(dataset.trips)

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
            context_vec = self.context_embed(ragged_batch)
            batch_loss = self.negative_log_likelihood_loss(context_vec, target_items)
            total_loss += tf.reduce_sum(batch_loss).numpy()
            total_samples += batch_loss.shape[0]

        return total_loss / total_samples

    def __repr__(self) -> str:
        """Return a string representation of the model."""
        return (
            "=" * 30 + "\n"
            "      Model Parameters\n"
            + "=" * 30
            + "\n"
            + f"{'Epochs':20}: {self.epochs}\n"
            + f"{'Number of Trips':20}: {self.last_n_baskets_dataset}\n"
            + f"{'n_negative_samples':20}: {self.n_negative_samples}\n"
            + f"{'Learning_rate':20}: {self.lr}\n"
            + f"{'Loss type':20}: NCE Loss\n"
            + f"{'Batch_size':20}: {self.batch_size}\n"
            + f"{'Embedding_dim':20}: {self.embedding_dim}\n"
            + "=" * 30
        )

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
            "Wi": self.Wi.numpy().tolist(),
            "Wo": self.Wo.numpy().tolist(),
            "wa": self.wa.numpy().tolist(),
            "empty_context_embedding": self.empty_context_embedding.numpy().tolist(),
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
        self.Wi = tf.Variable(np.array(data["Wi"], dtype=np.float32), name="Wi")
        self.Wo = tf.Variable(np.array(data["Wo"], dtype=np.float32), name="Wo")
        self.wa = tf.Variable(np.array(data["wa"], dtype=np.float32), name="wa")
        self.empty_context_embedding = tf.Variable(
            np.array(data["empty_context_embedding"], dtype=np.float32),
            name="empty_context_embedding",
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
