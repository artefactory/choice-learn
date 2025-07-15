"""Implementation of a simple attention-based model for item recommendation."""

import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import tqdm


class BaseModel:
    def __init__(
        self,
        n_items: int = 8,
        embedding_dim: int = 4,
        k_noise: int = 8,
        lr: float = 0.05,
        epochs: int = 100,
        optimizer: str = "Adam",
        batch_size: str = 16,
        loss_type: str = "nce", # maybe you can find a clearer word than "bad"
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
        self.loss_type = loss_type

        # Should not be an attribute of the model
        # self.data_generator = DataGenerator()
        
        if Q_distribution is None:
            assert (
                n_items > 1
            ), "n_items must be greater than 1 to define a uniform distribution."

            self.Q_distribution = tf.constant(
                [1.0 / (n_items - 1 + 1)] * n_items, dtype=tf.float32
            )
        else:
            self.Q_distribution = tf.constant(Q, dtype=tf.float32)

        self.instantiate()


    def instantiate(self) -> None:
        """Initializes the model parameters."""

        self.Wi = tf.Variable(
            tf.random.normal((self.n_items, self.embedding_dim), stddev=0.1), name="Wi"
        )
        self.Wo = tf.Variable(
            tf.random.normal((self.n_items, self.embedding_dim), stddev=0.1), name="Wo"
        )
        self.wa = tf.Variable(
            tf.random.normal((self.embedding_dim,), stddev=0.1), name="wa"
        )
        self.bo = tf.Variable(tf.zeros((self.n_items,)), name="bo") # Shouldn't it be a tf.constant ?
        self.is_trained = False
        self.loss_history = []

    def get_batches(self, dataset: tf.Tensor) -> list:
        """Generates batches of baskets for training or testing."""

        indices = list(range(dataset.shape[0]))
        random.shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            yield tf.gather(dataset, batch_indices)

    @property
    def trainable_weights(self):
        return [self.Wi, self.wa, self.Wo]

    def context_embed(self, context_items: tf.Tensor) -> tf.Tensor:
        """Returns the context embedding matrix. [self.embedding_dim]"""

        context_emb = tf.gather(self.Wi, context_items, axis=0)          
        attn_logits = tf.ragged.map_flat_values(
            lambda x: tf.tensordot(x, self.wa, axes=1), context_emb
        )
        attn_weights = tf.map_fn(
            lambda x: tf.nn.softmax(x), attn_logits
        )
        context_vec = tf.map_fn(
            lambda args: tf.reduce_sum(tf.transpose(args[0])*args[1] , axis=1),
            (context_emb, attn_weights), fn_output_signature=tf.float32
        )
        return context_vec

    def score(self, context_vec: tf.Tensor, target_items_idx: tf.Tensor) -> tf.Tensor:
        """Returns the score of the item given the context vector."""

        selected_Wo = tf.gather(self.Wo, target_items_idx) 
        return tf.reduce_sum(selected_Wo * context_vec, axis=1)
        
    """
    def bad_loss(self, context_items: tf.Tensor, target_items: tf.Tensor) -> tf.Tensor:

        context_vec = self.context_embed(context_items)
        scores = tf.map_fn(
            lambda x: tf.tensordot(self.Wo,x,  axes=1), context_vec)
        
        scores_softmax =  tf.nn.softmax(scores, axis=1)

        target_items_scores = tf.gather_nd(scores_softmax,
                               tf.stack([tf.range(tf.shape(target_items)[0]), target_items], axis=1))
        
        return tf.reduce_sum(tf.math.log(target_items_scores + 1))
    """

    def bad_loss(self, context_items: tf.Tensor, target_items: tf.Tensor) -> tf.Tensor:
        """Calculates the loss using a simple, naive softmax cross-entropy approach."""

        context_vec = self.context_embed(context_items)
        scores = tf.map_fn(
            lambda x: tf.tensordot(self.Wo, x, axes=1), context_vec
        )
        scores_softmax = tf.nn.softmax(scores, axis=1)

        # Gather the predicted probability for the true target item in each batch
        target_items_scores = tf.gather_nd(
            scores_softmax,
            tf.stack([tf.range(tf.shape(target_items)[0]), target_items], axis=1)
        )

        # Standard cross-entropy loss: -log(p(target))
        return -tf.reduce_sum(tf.math.log(target_items_scores + 1e-10))



    def nce_loss(self, context_items: tf.Tensor, target_items: tf.Tensor) -> tf.Tensor:
        """Calculates the loss using Noise Contrastive Estimation (NCE)."""

        batch_size = tf.shape(target_items)[0]
        context_vec = self.context_embed(context_items)
        pos_score = self.score(context_vec, target_items)

        all_items = tf.range(self.n_items, dtype=tf.int32)

        def get_neg_items(context, target):
            exclude = tf.concat([context, [target]], axis=0)
            mask = ~tf.reduce_any(tf.equal(all_items[:, None], exclude[None, :]), axis=1)
            neg_pool = tf.boolean_mask(all_items, mask)
            n_neg = tf.shape(neg_pool)[0]
            # If not enough negatives, pad with random items (could repeat); never happens 8
            neg_items = tf.cond(
                n_neg >= self.K_noise,
                lambda: tf.random.shuffle(neg_pool)[:self.K_noise],
                lambda: tf.pad(tf.random.shuffle(neg_pool),
                            [[0, self.K_noise - n_neg]],
                            constant_values=0)
            )
            return neg_items

        neg_items = tf.map_fn(
            lambda x: get_neg_items(x[0], x[1]),
            (context_items, target_items),
            fn_output_signature=tf.TensorSpec(shape=(self.K_noise,), dtype=tf.int32)
        )

        KQ = self.K_noise * tf.gather(self.Q_distribution, target_items)
        P_1 = tf.exp(pos_score) / (tf.exp(pos_score) + KQ)
        loss = -tf.math.log(P_1 + 1e-10)

        flat_context_vec = tf.repeat(context_vec, repeats=self.K_noise, axis=0)
        flat_neg_items = tf.reshape(neg_items, [-1])
        neg_scores = self.score(flat_context_vec, flat_neg_items)
        neg_scores = tf.reshape(neg_scores, [batch_size, self.K_noise])
        KQ_neg = self.K_noise * tf.gather(self.Q_distribution, neg_items)
        P_0 = 1 - (tf.exp(neg_scores) / (tf.exp(neg_scores) + KQ_neg))
        loss -= tf.reduce_sum(tf.math.log(P_0 + 1e-10), axis=1)

        return tf.reduce_sum(loss)


    def train_step(self, batch: tf.Tensor) -> tf.Tensor:
        """Performs a single training step on the batch of baskets."""

        def basket_loss(target_items_idx, context_items) -> tf.Tensor:
            """Calculates the loss for a single basket."""
            if self.loss_type == "softmax":
                return self.bad_loss(context_items, target_items_idx)
            elif self.loss_type == "nce":
                return self.nce_loss(context_items, target_items_idx)

        with tf.GradientTape() as tape:
            total_loss = 0
            target_items_idx = tf.map_fn(
                lambda x: tf.random.uniform(shape=[], maxval=x.shape[0], dtype=tf.int32), 
                batch, 
                fn_output_signature=tf.int32)
            mask = tf.map_fn(
                lambda x: tf.one_hot(x[0], depth=tf.cast(x[1], tf.int32), on_value=False, off_value=True, dtype=tf.bool),
                (target_items_idx, batch.row_lengths()),
                fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=tf.bool)
            )
            context_items = tf.ragged.boolean_mask(batch, mask)
            if self.loss_type == "nce":
                total_loss += self.nce_loss(context_items, target_items_idx)
            else:
                total_loss += self.bad_loss(context_items, target_items_idx)


        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return float(total_loss.numpy())

    def predict(self, context_items: tf.Tensor) -> np.ndarray:
        """Predicts the item probabilities given the context items."""

        context_vec = self.context_embed(context_items)
        scores = tf.tensordot(self.Wo, tf.transpose(context_vec), axes=1)
        return tf.nn.softmax(scores, axis = 1).numpy()

    def fit(
        self,
        dataset,
        repr: bool = False,
        loss_type: str = "nce",

    ) -> None:
        """Trains the model for a specified number of epochs."""
        # Should be at the end of the training
        # self.loss_type = loss_type
        # if epochs == None:
        #     epochs = self.epochs

        # dataset = self.data_generator.generate_dummy_dataset()
        # print(f"Generated dataset with {len(dataset)} baskets.")
        # print(f"Training with batch size: {batch_size}, epochs: {epochs}")
        self.loss_type = loss_type
        dataset = tf.ragged.constant(dataset, dtype=tf.int32)

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
        
        contexts = tf.constant([[i] for i in range(self.n_items)], dtype=tf.int32)
        return self.predict(contexts)

    def represent(self, hyperparams: bool = True) -> None:
        """Prints the model parameters."""

        if not self.is_trained:
            raise ValueError(
                "Model must be trained before representation. Call fit() first."
            )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        im1 = axes[0].imshow(np.stack(self.model_distribution_matrix()), vmin=0.0, vmax=1.0, cmap="Spectral")
        axes[0].set_title("Model P(i|j) on elementary baskets")
        plt.colorbar(im1, ax=axes[0])

        line_plot = axes[1].plot(self.loss_history, label="Training Loss")
        axes[1].set_xlabel("Training Steps")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Training Loss History")


        plt.tight_layout()
        plt.savefig("model_representation1.png")
        plt.show()

        if hyperparams:
            print("=" * 30)
            print("      Model Parameters")
            print("=" * 30)
            print(f"{'Embedding_dim':20}: {self.embedding_dim}")
            print(f"{'K_noise':20}: {self.K_noise}")
            print(f"{'Learning_rate':20}: {self.lr}")
            print(f"{'Epochs':20}: {self.epochs}")
            print(f"{'Loss type':20}: {self.loss_type}")
            print("=" * 30)
