"""Implementation of a simple attention-based model for item recommendation."""

import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import tqdm
from data import SyntheticDataGenerator


class BaseModel:
    def __init__(
        self,
        n_items: int = 8,
        embedding_dim: int = 4,
        k_noise: int = 8,
        lr: float = 0.005,
        epochs: int = 30,
        optimizer: str = "Adam",
        batch_size: str = 8,
        loss: str = "bad", # maybe you can find a clearer word than "bad"
        Q_distribution: int = None,
    ) -> None:

        self.n_items: int = n_items
        self.embedding_dim: int = embedding_dim
        self.K_noise: int = k_noise
        self.lr: float = lr
        self.epochs: int = epochs
        self.batch_size: int = batch_size

        # Useless with what happens in the following lines
        # self.optimizer: str = optimizer

        if optimizer == "Adam":
            self.optimizer = tf.keras.optimizers.Adam(self.lr)
        else:
            print(
                f"Optimizer {optimizer} not implemented, switching for default Adam"
            )
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.loss = loss

        # Should not be an attribute of the model
        # self.data_generator = DataGenerator()
        
        if Q_distribution is None:
            assert (
                n_items > 1
            ), "n_items must be greater than 1 to define a uniform distribution."

            self.Q_distribution = tf.constant(
                [1.0 / (n_items - 1)] * n_items, dtype=tf.float32
            )
        else:
            self.Q_distribution = tf.constant(Q, dtype=tf.float32)

        self.instantiate()


    def instantiate(self) -> None:
        """Initializes the model parameters."""

        self.Wi = tf.Variable(
            tf.random.normal((self.embedding_dim, self.n_items), stddev=0.1), name="Wi"
        )
        self.Wo = tf.Variable(
            tf.random.normal((self.n_items, self.embedding_dim), stddev=0.1), name="Wo"
        )
        self.wa = tf.Variable(
            tf.random.normal((self.embedding_dim,), stddev=0.1), name="wa"
        )
        self.bo = tf.Variable(tf.zeros((self.n_items,)), name="bo") # Shouldn't it be a tf.constant ?
        self.is_trained = False
        self.loss_type = "nce"
        self.loss_history = []

    def get_batches(self, dataset: list) -> list:
        """Generates batches of baskets for training or testing."""

        indices = list(range(len(dataset)))
        random.shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            yield [dataset[j] for j in batch_indices]

    @property
    def trainable_weights(self):
        return [self.Wi, self.wa, self.Wo]

    def context_embed(self, context_items: list) -> tf.Tensor:
        """Returns the context embedding matrix. [self.embedding_dim]"""

        context_items = tf.convert_to_tensor(context_items, dtype=tf.int32)
        context_emb = tf.gather(self.Wi, context_items, axis=1)
        attn_logits = tf.tensordot(self.wa, context_emb, axes=1)
        attn_weights = tf.nn.softmax(attn_logits)
        context_vec = tf.reduce_sum(
            context_emb * tf.expand_dims(attn_weights, axis=0), axis=1
        )
        return context_vec

    def score(self, context_vec: tf.Tensor, item: int) -> tf.Tensor:
        """Returns the score of the item given the context vector."""

        return tf.tensordot(self.Wo[item], context_vec, axes=1)

    def bad_loss(self, context_items: list, target_item: int) -> tf.Tensor:
        """Calculates the loss using a simple, naive softmax approach."""

        context_vec = self.context_embed(context_items)
        scores = tf.tensordot(self.Wo, context_vec, axes=1)
        return -tf.math.log(tf.nn.softmax(scores)[target_item])

    def nce_loss(self, context_items: list, target_item: int) -> tf.Tensor:
        """Calculates the loss using Noise Contrastive Estimation (NCE)."""
        context_items = tf.convert_to_tensor(context_items, dtype=tf.int32)

        context_vec = self.context_embed(context_items)
        pos_score = self.score(context_vec, target_item)
        # Negative sampling: exclude basket items
        all_items = tf.range(self.n_items, dtype=tf.int32)
        mask = ~tf.reduce_any(
            tf.equal(all_items[:, None], tf.expand_dims(context_items, 0)), axis=1
        )
        neg_pool = tf.boolean_mask(all_items, mask)
        neg_items = tf.random.shuffle(neg_pool)[: self.K_noise]
        # NCE probabilities
        KQ = self.K_noise * self.Q_distribution[target_item]
        P_1 = tf.exp(pos_score) / (tf.exp(pos_score) + KQ)
        loss = -tf.math.log(P_1)
        for i in neg_items:
            neg_score = self.score(context_vec, i)
            KQ_neg = self.K_noise * self.Q_distribution[i]
            P_0 = 1 - (tf.exp(neg_score) / (tf.exp(neg_score) + KQ_neg))
            loss -= tf.math.log(P_0)
        return loss

    def train_step(self, batch: list) -> tf.Tensor:
        """Performs a single training step on the batch of baskets."""

        def basket_loss(basket: list) -> tf.Tensor:
            """Calculates the loss for a single basket."""

            target_item = random.choice(basket)
            context_items = [i for i in basket if i != target_item]

            if self.loss_type == "softmax":
                return self.bad_loss(context_items, target_item)
            elif self.loss_type == "nce":
                return self.nce_loss(context_items, target_item)

        with tf.GradientTape() as tape:
            total_loss = 0
            for basket in batch:
                total_loss += basket_loss(basket)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return float(total_loss.numpy())

    def predict(self, context_items: list) -> np.ndarray:
        """Predicts the item probabilities given the context items."""

        context_vec = self.context_embed(context_items)
        scores = tf.tensordot(self.Wo, context_vec, axes=1)
        return tf.nn.softmax(scores).numpy()

    def fit(
        self,
        dataset,
        repr: bool = False,

    ) -> None:
        """Trains the model for a specified number of epochs."""
        # Should be at the end of the training
        # self.loss_type = loss_type
        # if epochs == None:
        #     epochs = self.epochs

        # dataset = self.data_generator.generate_dummy_dataset()
        # print(f"Generated dataset with {len(dataset)} baskets.")
        # print(f"Training with batch size: {batch_size}, epochs: {epochs}")

        iterable = tqdm.trange(self.epochs, desc="Training Epochs")
        for epoch in iterable:
            epoch_loss = 0
            for batch in self.get_batches(dataset):
                loss = self.train_step(batch)
                epoch_loss += loss
            self.loss_history.append(epoch_loss)
            iterable.set_postfix({"epoch_loss": epoch_loss})

        print(f"Training completed. Final loss: {self.loss_history[-1]}")
        self.is_trained = True
        if repr:
            self.represent()

    def evaluate(self, dataset: list) -> np.ndarray:
        """Evaluates the model on the provided dataset."""

        if not self.is_trained:
            raise ValueError(
                "Model must be trained before evaluation. Call fit() first."
            )

        correct_predictions = 0

        for batch in self.data_generator.get_batches(dataset):
            for basket in batch:
                target_item = random.choice(list(basket))
                context_items = [item for item in basket if item != target_item]
                scores = self.predict(context_items)
                mask = np.ones(I, dtype=bool)
                mask[context_items] = False
                scores_np = scores.numpy()
                scores_np[~mask] = -np.inf
                pred = np.argmax(scores_np)
                if pred == target_item:
                    correct_predictions += 1

        return correct_predictions / (len(dataset))

    def model_distribution_matrix(self) -> list:
        P = []
        for i in range(self.n_items):
            P.append(tf.nn.softmax(self.predict(np.array([i]))).numpy())
        return P

    def represent(self, hyperparams: bool = True) -> None:
        """Prints the model parameters."""

        if not self.is_trained:
            raise ValueError(
                "Model must be trained before representation. Call fit() first."
            )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        im1 = axes[0].imshow(np.stack(self.model_distribution_matrix()))
        axes[0].set_title("Model P(i|j) on elementary baskets")
        plt.colorbar(im1, ax=axes[0])

        line_plot = axes[1].plot(self.loss_history, label="Training Loss")
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
            print(f"{'Embedding_dim':20}: {self.embedding_dim}")
            print(f"{'K_noise':20}: {self.K_noise}")
            print(f"{'Learning_rate':20}: {self.lr}")
            print(f"{'Epochs':20}: {self.epochs}")
            print(f"{'Optimizer':20}: {self.optimizer._name}")
            print(f"{'Loss type':20}: {self.loss_type}")
            print("=" * 30)
