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
        k_noise: int = 6,
        lr: float = 0.01,
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

        if optimizer == "Adam":
            self.optimizer = tf.keras.optimizers.Adam(self.lr)
        else:
            print(
                f"Optimizer {optimizer} not implemented, switching for default Adam"
            )
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.loss_type = loss_type
        
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
    

    def score(self, context_vec: tf.Tensor, items: tf.Tensor) -> tf.Tensor:
        """Returns the score of the item given the context vector."""

        return tf.map_fn(
            lambda args: tf.tensordot(self.Wo[args[1].numpy()], args[0], axes=1),
            (context_vec, items), fn_output_signature=tf.float32)
    

    def nll_loss(self, context_items: tf.Tensor, target_items: tf.Tensor) -> tf.Tensor:
        """Calculates the loss using a simple, naive softmax cross-entropy approach."""
        
        context_vec = self.context_embed(context_items)
        scores = tf.map_fn(
            lambda x: tf.tensordot(self.Wo, x, axes=1), context_vec
        )
        scores_softmax = tf.nn.softmax(scores)

        target_items_scores = tf.gather_nd(
            scores_softmax,
            tf.stack([tf.range(tf.shape(target_items)[0]), target_items], axis=1)
        )
        return -tf.math.log(target_items_scores)


    def nce_loss(self, context_items: tf.Tensor, target_items: tf.Tensor) -> tf.Tensor:
        """Calculates the loss using Noise Contrastive Estimation (NCE)."""

        context_vec = self.context_embed(context_items)
        pos_score = self.score(context_vec, target_items)
        list_neg_items = []

        for i in range(context_items.shape[0]):
            neg_items = []
            force_ending_cpt = 0
            while len(neg_items) < self.K_noise and force_ending_cpt < 1000:
                force_ending_cpt += 1
                candidate_item = random.randint(0, self.n_items - 1)
                if candidate_item not in context_items[i] and candidate_item != target_items[i]:
                    neg_items.append(candidate_item)
            list_neg_items.append(neg_items)
        list_neg_items = tf.constant(list_neg_items, dtype=tf.int32)

        

        P_1 = tf.map_fn(
            lambda args: 1 / (1 + self.K_noise * self.Q_distribution[args[1]]*tf.exp(-args[0])),
            (pos_score, target_items), fn_output_signature=tf.float32)
        loss = -tf.math.log(P_1)

        for i in range(len(neg_items)):
            column = tf.gather(list_neg_items, i, axis=1)
            neg_score = self.score(context_vec, column)
            P_0 = tf.map_fn(
            lambda args: 1- (1 / (1 + self.K_noise * self.Q_distribution[args[1]]*tf.exp(-args[0]))),
            (neg_score, column), fn_output_signature=tf.float32)
            loss -= tf.math.log(P_0)
        
        return loss



    def train_step(self, batch: tf.Tensor) -> tf.Tensor:
        """Performs a single training step on the batch of baskets."""

        with tf.GradientTape() as tape:
            total_loss = 0

            target_items_idx_for_mask = tf.map_fn(
                lambda x: tf.random.uniform(shape=[], maxval=x.shape[0], dtype=tf.int32), 
                batch, 
                fn_output_signature=tf.int32)
            
            target_items = tf.map_fn(
                lambda args: args[0][args[1]],
                (batch, target_items_idx_for_mask),
                fn_output_signature=tf.int32)
            

            mask = tf.map_fn(
                lambda x: tf.one_hot(x[0], depth=tf.cast(x[1], tf.int32), on_value=False, off_value=True, dtype=tf.bool),
                (target_items_idx_for_mask, batch.row_lengths()),
                fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=tf.bool))
            
            context_items = tf.ragged.boolean_mask(batch, mask)
            if self.loss_type == "nce":
                total_loss += tf.reduce_sum(self.nce_loss(context_items, target_items))
            else:
                total_loss += tf.reduce_sum(self.nll_loss(context_items, target_items))


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


    def model_distribution_matrix(self) -> list:
        """Returns the model distribution matrix P(i|j) for each item i given j."""
        
        contexts = tf.constant([[i] for i in range(self.n_items)], dtype=tf.int32)
        return self.predict(contexts)
    

    def represent(self, hyperparams: bool = True) -> None:
        """Prints the model parameters."""

        if not self.is_trained:
            raise ValueError(
                "Model must be trained before representation. Call fit() first."
            )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        im1 = axes[0].imshow(np.stack(self.model_distribution_matrix()), vmin=0.0, vmax=np.max(np.stack(self.model_distribution_matrix())), cmap="Spectral")
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
            print(f"{'Batch_size':20}: {self.batch_size}")
            print(f"{'Epochs':20}: {self.epochs}")
            print(f"{'Loss type':20}: {self.loss_type}")
            print("=" * 30)
    


## To implement:
# Save model
# Load model
# Save model configuration
# Load model configuration
# predict top-k items given a context
