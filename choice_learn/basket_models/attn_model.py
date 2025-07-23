"""
Implementation of an attention-based model for item recommendation.

Cf. "Attention-Based Transactional Context Embedding for Next-Item Recommendation".
Wang et al. (2018).
"""

import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tqdm

from .dataset import TripDataset


class AttnModel:
    """Class for the attention-based model."""

    def __init__(
        self,
        lr: float = 0.005,
        epochs: int = 450,
        optimizer: str = "Adam",
        batch_size: str = 32,
        loss_type: str = "nce",
    ) -> None:
        """Initialize the model with hyperparameters.

        Parameters
        ----------
            lr : float
                Learning rate for the optimizer.
            epochs : int
                Number of training epochs.
            optimizer : str
                Optimizer to use for training. Default is "Adam".
            batch_size : int
                Size of the training batches.
            loss_type : str
                Type of loss function to use. Options are "nce"
                or "nll" (negative log likelihood).
        """
        self.lr: float = lr
        self.epochs: int = epochs
        self.batch_size: int = batch_size

        if optimizer.lower() == "adam":
            self.optimizer = tf.keras.optimizers.Adam(self.lr)
        else:
            print(f"Optimizer {optimizer} not implemented, switching for default Adam")
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self.loss_type = loss_type

        self.instantiated = False

    def instantiate(self, n_items, embedding_dim, k_noise, q_distribution=None) -> None:
        """Initialize the model parameters.

        Parameters
        ----------
            n_items : int
                Number of unique items in the dataset.
            embedding_dim : int
                Dimension of the item embeddings.
            K_noise : int
                Number of negative samples for Noise Contrastive Estimation.
            Q_distribution : list
                Probability distribution for negative sampling.
                If None, a uniform distribution is used.
        """
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.K_noise = k_noise

        if q_distribution is None:
            assert n_items > 1, (
                "n_items must be greater than 1 to define a uniform distribution."
            )

            self.Q_distribution = tf.constant(
                [1.0 / (n_items - 1 + 1)] * n_items, dtype=tf.float32
            )
        else:
            self.Q_distribution = tf.constant(q_distribution, dtype=tf.float32)

        self.Wi = tf.Variable(
            tf.random.normal((self.n_items, self.embedding_dim), stddev=0.1), name="Wi"
        )
        self.Wo = tf.Variable(
            tf.random.normal((self.n_items, self.embedding_dim), stddev=0.1), name="Wo"
        )
        self.wa = tf.Variable(
            tf.random.normal((self.embedding_dim,), stddev=0.1), name="wa"
        )

        self.Wo_bias = tf.Variable(tf.zeros([self.n_items]), name="Wo_bias")

        self.empty_context_emb = tf.Variable(
            tf.random.normal((self.embedding_dim,), stddev=0.1),
            name="empty_context_emb",
        )

        self.is_trained = False
        self.loss_history = []
        self.instantiated = True

    @property
    def trainable_weights(self):
        """Return the trainable weights of the model.

        Returns
        -------
            list
                List of trainable weights (Wi, wa, Wo).
        """
        return [self.Wi, self.wa, self.Wo, self.empty_context_emb]

    def get_batches(self, dataset: tf.Tensor) -> list:
        """Generate batches of baskets for training or testing.

        Parameters
        ----------
            dataset : tf.Tensor
                Tensor containing the dataset of baskets.
        """
        indices = list(range(dataset.shape[0]))
        random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            yield tf.gather(dataset, batch_indices)


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
                lambda: self.empty_context_emb,
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


    def nll_loss(self, context_items: tf.Tensor, target_items: tf.Tensor) -> tf.Tensor:
        """Calculate the loss using a simple, naive softmax cross-entropy approach.

        Parameters
        ----------
            context_items : tf.Tensor
                Tensor containing the context items.
            target_items : tf.Tensor
                Tensor containing the target items to predict.

        Returns
        -------
            tf.Tensor
                Tensor containing the negative log likelihood loss.
        """
        context_vec = self.context_embed(context_items)
        scores_softmax = tf.map_fn(
            lambda x: tf.nn.softmax(tf.tensordot(self.Wo, x, axes=1)), context_vec
        )

        target_items_scores = tf.gather_nd(
            scores_softmax,
            tf.stack([tf.range(tf.shape(target_items)[0]), target_items], axis=1),
        )
        return -tf.math.log(target_items_scores)


    def my_nce_loss(self,
                    pos_score: tf.Tensor,
                    target_items: tf.Tensor,
                    list_neg_items: tf.Tensor,
                    neg_scores: tf.Tensor
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
            loss -= tf.math.log(self.proba_negative_samples(
                neg_scores[i],
                tf.gather(list_neg_items, i, axis=1)))
        return loss



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
        pos_score = tf.convert_to_tensor(pos_score)
        target_items = tf.convert_to_tensor(target_items)
        if len(pos_score.shape) == 0:
            pos_score = tf.expand_dims(pos_score, 0)
        if len(target_items.shape) == 0:
            target_items = tf.expand_dims(target_items, 0)
        return tf.map_fn(
            lambda args: 1
            / (1 + self.K_noise * self.Q_distribution[args[1]] * tf.exp(-args[0])),
            (pos_score, target_items),
            fn_output_signature=tf.float32,
        )

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
        neg_score = tf.convert_to_tensor(neg_score)
        target_items = tf.convert_to_tensor(target_items)
        if len(neg_score.shape) == 0:
            neg_score = tf.expand_dims(neg_score, 0)
        if len(target_items.shape) == 0:
            target_items = tf.expand_dims(target_items, 0)
        return tf.map_fn(
            lambda args: 1 - self.proba_positive_samples(args[0], args[1]),
            (neg_score, target_items),
            fn_output_signature=tf.float32,
        )


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
        list_neg_items = []

        for i in range(context_items.shape[0]):
            neg_items = []
            force_ending_cpt = 0

            while len(neg_items) < self.K_noise and force_ending_cpt < 1000:
                force_ending_cpt += 1
                candidate_item = random.randint(0, self.n_items - 1)

                in_context = tf.reduce_any(tf.equal(context_items[i], candidate_item))
                is_target = tf.equal(target_items[i], candidate_item)

                if not in_context and not is_target:
                    neg_items.append(candidate_item)

            list_neg_items.append(neg_items)

        return tf.constant(list_neg_items, dtype=tf.int32)


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
            total_loss = 0

            if self.loss_type == "nce":
                context_vec = self.context_embed(context_batch)
                pos_score = self.score(context_vec, items_batch)
                list_neg_items = self.get_negative_samples(context_batch, items_batch)
                neg_scores = [self.score(context_vec, tf.gather(list_neg_items, i, axis=1))
                              for i in range(len(list_neg_items[0]))]

                total_loss += tf.reduce_sum(self.my_nce_loss(
                    pos_score = pos_score,
                    target_items = items_batch,
                    list_neg_items = list_neg_items,
                    neg_scores = neg_scores
                ))
            else:
                total_loss += tf.reduce_sum(self.nll_loss(context_batch, items_batch))

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return total_loss

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
            raise ValueError(
                "Model must be trained before prediction. Call fit() first."
            )



        context_vec = self.context_embed(context_items)
        scores = tf.tensordot(self.Wo, tf.transpose(context_vec), axes=1)
        scores = tf.linalg.set_diag(
            scores, tf.fill([tf.shape(scores)[0]], float("-inf"))
        )
        return tf.nn.softmax(scores, axis=1).numpy()

    def fit(
        self,
        dataset,
        repr: bool = False,
        loss_type: str = "nce",
    ) -> None:
        """Trains the model for a specified number of epochs.

        Parameters
        ----------
            dataset : list or np.ndarray
                Dataset of baskets to train the model on.
            repr : bool
                If True, represents the model after training.
            loss_type : str
                Type of loss function to use. Options are "nce"
                or "nll" (negative log likelihood).
        """
        if not self.instantiated:
            raise ValueError(
                "Model must be instantiated before training. Call instantiate() first."
            )

        if not isinstance(dataset, TripDataset):
            raise TypeError("Dataset must be a list or numpy array.")

        if not dataset:
            raise ValueError("Dataset cannot be empty.")

        self.loss_type = loss_type

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

            self.loss_history.append(epoch_loss)
            iterable.set_postfix({"epoch_loss": epoch_loss})

        print(f"Training completed. Final loss: {self.loss_history[-1]}")
        self.is_trained = True

        if repr:
            contexts = tf.constant([[i] for i in range(self.n_items)], dtype=tf.int32)
            self.represent(self.predict(contexts), hyperparams=True)

    def evaluate(self, dataset: tf.Tensor) -> float:
        """Evaluate the model on the given dataset and returns the average loss.

        Parameters
        ----------
            dataset : tf.Tensor
                Tensor containing the dataset of baskets to evaluate.

        Returns
        -------
            float
                Average loss over the dataset.
        """
        dataset = tf.ragged.constant(dataset, dtype=tf.int32)
        total_loss = 0
        num_batches = 0
        distribution_matrix = []

        for batch in self.get_batches(dataset):
            if batch.shape[0] != self.batch_size:
                continue
            target_items_idx_for_mask = tf.map_fn(
                lambda x: tf.random.uniform(
                    shape=[], maxval=tf.shape(x)[0], dtype=tf.int32
                ),
                batch,
                fn_output_signature=tf.int32,
            )

            target_items = tf.map_fn(
                lambda args: args[0][args[1]],
                (batch, target_items_idx_for_mask),
                fn_output_signature=tf.int32,
            )

            mask = tf.map_fn(
                lambda x: tf.one_hot(
                    x[0],
                    depth=tf.cast(x[1], tf.int32),
                    on_value=False,
                    off_value=True,
                    dtype=tf.bool,
                ),
                (target_items_idx_for_mask, batch.row_lengths()),
                fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=tf.bool),
            )

            context_items = tf.ragged.boolean_mask(batch, mask)

            if self.loss_type == "nce":
                total_loss += tf.reduce_sum(self.nce_loss(context_items, target_items))
            else:
                total_loss += tf.reduce_sum(self.nll_loss(context_items, target_items))

            distribution_matrix.append(self.predict(context_items))
            num_batches += 1

        distribution_matrix = np.concatenate(distribution_matrix, axis=0)
        self.represent(distribution_matrix, hyperparams=False)

        return total_loss / num_batches

    def represent(
        self,
        context_prediction: tf.Tensor,
        hyperparams: bool = True,
        show_loss: bool = True,
    ) -> None:
        """Print the model parameters.

        Parameters
        ----------
            hyperparams : bool
                If True, prints the hyperparameters of the model.
        """
        if not self.is_trained:
            raise ValueError(
                "Model must be trained before representation. Call fit() first."
            )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        im1 = axes[0].imshow(
            np.stack(context_prediction),
            vmin=0.0,
            vmax=np.max(np.stack(context_prediction)),
            cmap="Spectral",
        )
        axes[0].set_title("Model P(i|j) on elementary baskets")
        plt.colorbar(im1, ax=axes[0])
        if show_loss:
            axes[1].plot(self.loss_history, label="Training Loss")
            axes[1].set_xlabel("Training Steps")
            axes[1].set_ylabel("Loss")
            axes[1].set_title("Training Loss History")

        plt.tight_layout()
        plt.savefig("model_representation.png")
        plt.show()

        if hyperparams:
            print("=" * 30)
            print("      Model Parameters")
            print("=" * 30)
            print(f"{'Epochs':20}: {self.epochs}")
            print(f"{'K_noise':20}: {self.K_noise}")
            print(f"{'Learning_rate':20}: {self.lr}")
            print(f"{'Loss type':20}: {self.loss_type}")
            print(f"{'Batch_size':20}: {self.batch_size}")
            print(f"{'Embedding_dim':20}: {self.embedding_dim}")

            print("=" * 30)

    def save_model(self, filepath: str, overwrite: bool) -> None:
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
            "k_noise": int(self.K_noise),
            "lr": float(self.lr),
            "epochs": int(self.epochs),
            "optimizer": self.optimizer.get_config(),
            "batch_size": int(self.batch_size),
            "loss_type": self.loss_type,
            "Q_distribution": self.Q_distribution.numpy().tolist()
            if hasattr(self.Q_distribution, "numpy")
            else list(self.Q_distribution),
            "Wi": self.Wi.numpy().tolist(),
            "Wo": self.Wo.numpy().tolist(),
            "wa": self.wa.numpy().tolist(),
            "loss_history": [float(loss) for loss in self.loss_history],
        }
        with open(filepath, "w") as f:
            json.dump(data, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load the model parameters from a file.

        Parameters
        ----------
            filepath : str
                Path to the file from which the model parameters will be loaded.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} does not exist.")

        with open(filepath, "r") as f:
            data = json.load(f)

        # Set hyperparameters and attributes
        self.n_items = int(data["n_items"])
        self.embedding_dim = int(data["embedding_dim"])
        self.K_noise = int(data["k_noise"])
        self.lr = float(data["lr"])
        self.epochs = int(data["epochs"])
        self.batch_size = int(data["batch_size"])
        self.loss_type = data["loss_type"]
        self.Q_distribution = tf.constant(data["Q_distribution"], dtype=tf.float32)
        self.loss_history = [float(loss) for loss in data.get("loss_history", [])]

        # Re-instantiate optimizer
        if isinstance(data["optimizer"], dict) and "name" in data["optimizer"]:
            if data["optimizer"]["name"].lower() == "adam":
                self.optimizer = tf.keras.optimizers.Adam(self.lr)
            else:
                print(
                    f"Optimizer {data['optimizer']['name']} not implemented, switching to Adam"
                )
                self.optimizer = tf.keras.optimizers.Adam(self.lr)
        else:
            self.optimizer = tf.keras.optimizers.Adam(self.lr)

        # Re-instantiate variables
        self.Wi = tf.Variable(np.array(data["Wi"], dtype=np.float32), name="Wi")
        self.Wo = tf.Variable(np.array(data["Wo"], dtype=np.float32), name="Wo")
        self.wa = tf.Variable(np.array(data["wa"], dtype=np.float32), name="wa")

        self.is_trained = True
        self.instantiated = True
        print(f"Model loaded from {filepath}")

"""
    def nce_loss(self, context_items: tf.Tensor, target_items: tf.Tensor) -> tf.Tensor:
        context_vec = self.context_embed(context_items)  # [batch_size, embedding_dim]
        if len(target_items.shape) == 1:
            target_items = tf.expand_dims(target_items, axis=1)
        loss = tf.nn.nce_loss(
            weights=self.Wo,
            biases=self.Wo_bias,
            labels=target_items,
            inputs=context_vec,
            num_sampled=self.K_noise,
            num_classes=self.n_items,
            remove_accidental_hits=True,
        )
        return loss
"""
