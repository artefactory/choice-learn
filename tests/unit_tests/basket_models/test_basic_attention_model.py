"""Contain unit tests for the AttentionBasedContextEmbedding model."""

import numpy as np
import tensorflow as tf

from choice_learn.basket_models.basic_attention_model import AttentionBasedContextEmbedding
from choice_learn.basket_models.data import Trip, TripDataset
from choice_learn.basket_models.datasets import SyntheticDataGenerator

# Test hyperparameters
epochs = 4
lr = 0.01
embedding_dim = 5
n_negative_samples = 3
n_baskets_train = 10
n_baskets_eval = 10


# Generate Training & Evaluation Data

data_gen = SyntheticDataGenerator(
    proba_complementary_items=0.7,
    proba_neutral_items=0.3,
    noise_proba=0.15,
    items_nest={0: [0, 1, 2], 1: [3, 4, 5], 2: [6], 3: [7]},
    nests_interactions=[
        ["", "compl", "neutral", "neutral"],
        ["compl", "", "neutral", "neutral"],
        ["neutral", "neutral", "", "neutral"],
        ["neutral", "neutral", "neutral", ""],
    ],
)

assortments_matrix = np.array([[1, 0, 1, 1, 0, 1, 1, 1]])


train_trip_dataset = data_gen.generate_trip_dataset(n_baskets_train, assortments_matrix)
train_trip_dataset = data_gen.generate_trip_dataset(n_baskets_eval, assortments_matrix)

# Generate custom dataset
assortment = np.array([1, 0, 1, 1, 0, 1, 1, 0])
n_items = len(assortment)
baskets = [[0, 6], [1, 2, 3, 6], [3, 7], [0, 1, 7], [6], [7]]
contexts = [[0], [1, 3, 6], [3], [0, 1], [], []]
target_items = [6, 2, 7, 7, 6, 7]


ragged_batch = tf.ragged.constant(contexts, dtype=tf.int32)


# Define the model
model = AttentionBasedContextEmbedding(
    epochs=epochs,
    lr=lr,
    latent_size=embedding_dim,
    n_negative_samples=n_negative_samples,
)
model.instantiate(n_items=n_items)
Wi, wa, Wo, empty_context_emb = model.trainable_weights
Wi = tf.transpose(Wi)


def context_embed_tester(context_items: list) -> tf.Tensor:
    """Return the context embedding matrix."""
    if len(context_items) > 0:
        context_items = tf.convert_to_tensor(context_items, dtype=tf.int32)
        context_emb = tf.gather(Wi, context_items, axis=1)

        attn_logits = tf.tensordot(wa, context_emb, axes=1)
        attn_weights = tf.nn.softmax(attn_logits)

        return tf.reduce_sum(context_emb * tf.expand_dims(attn_weights, axis=0), axis=1)
    return empty_context_emb


def test_context_embedding():
    """Test the context embedding method of the AttentionBasedContextEmbedding model.

    This method should compute the context embeddings for the given contexts.
    """
    custom_context_emb = model.embed_context(ragged_batch)
    assert isinstance(custom_context_emb, tf.Tensor), "Context embedding should be a Tensor"
    custom_context_emb = model.embed_context(ragged_batch).numpy()
    expected_embedding = np.array(
        [context_embed_tester(context).numpy().tolist() for context in contexts]
    )
    assert custom_context_emb.shape == (len(baskets), embedding_dim), (
        "Context embedding shape mismatch"
    )
    assert custom_context_emb.all() == expected_embedding.all(), (
        "Context embedding values do not match expected values"
    )


def naive_basket_score(contexts: list, items: list) -> float:
    """Compute the naive basket score for each item in the context.

    Args:
        contexts (list): List of context items.
        items (list): List of target items.

    Returns
    -------
        np.ndarray: Array of scores for each item.
    """
    scores = []
    for i in range(len(contexts)):
        context = contexts[i]
        item = items[i]
        scores.append(tf.tensordot(Wo[item], context_embed_tester(context), axes=1).numpy())
    return np.array(scores)


def test_score():
    """
    Test the score method of the AttentionBasedContextEmbedding model.

    This method should compute the scores for the target items based on the context embeddings.
    """
    scores = model.compute_batch_utility(
        item_batch=tf.constant(target_items, dtype=tf.int32),
        basket_batch=ragged_batch,
        store_batch=tf.ones_like(target_items),
        week_batch=tf.ones_like(target_items),
        price_batch=tf.ones_like(target_items),
        available_item_batch=tf.ones_like(target_items),
    )
    expected_scores = naive_basket_score(contexts, target_items)

    assert isinstance(scores, tf.Tensor), "Scores should be a Tensor"
    assert scores.shape == (len(baskets),), "Scores shape mismatch"
    assert np.allclose(scores.numpy(), expected_scores), (
        f"Scores do not match expected values. Expected: {expected_scores}, Got: {scores.numpy()}"
    )


def test_get_negative_samples():
    """
    Test the get_negative_samples method.

    This method should return a list of negative samples for each basket.
    """
    for i in range(len(baskets)):
        negative_samples = model.get_negative_samples(
            purchased_items=ragged_batch[i],
            next_item=target_items[i],
            available_items=assortments_matrix[0],
            n_samples=n_negative_samples,
        )
        assert len(negative_samples) == n_negative_samples, "Negative samples length mismatch"
        basket = baskets[i]
        for item in basket:
            if item in negative_samples:
                assert item not in negative_samples[i].numpy(), (
                    f"Item {item} should not be in negative samples"
                )


def test_evaluate_uniform():
    """
    Test the evaluate method of the AttentionBasedContextEmbedding model with uniform logits.

    This method should return the expected loss for a uniform distribution.
    """
    n_items = 3
    n_trips = 3
    embedding_dim = 2

    # All items available
    assortment_matrix = np.ones((1, n_items), dtype=int)

    # Each trip contains a single item (0, 1, 2)
    trips = []
    for i in range(n_trips):
        purchases = np.array([i])
        prices = np.ones(n_items)
        assortment = assortment_matrix[0]
        trips.append(Trip(purchases, prices, assortment))

    dataset = TripDataset(trips, assortment_matrix)

    # Instantiate model and set weights to zero for uniform logits
    model = AttentionBasedContextEmbedding(
        epochs=1,
        lr=0.01,
        latent_size=embedding_dim,
        n_negative_samples=1,
        batch_size=1,
    )
    model.instantiate(n_items=n_items)
    model.is_trained = True
    model.Wi.assign(tf.zeros_like(model.Wi))
    model.Wo.assign(tf.zeros_like(model.Wo))
    model.wa.assign(tf.zeros_like(model.wa))
    model.empty_context_embedding.assign(tf.zeros_like(model.empty_context_embedding))

    # Expected loss for uniform distribution: -log(1/n_items)
    expected_loss = -np.log(1.0 / n_items)

    # Evaluate
    eval_loss = model.evaluate(dataset)["negative_log_likelihood"]
    assert np.allclose(eval_loss, expected_loss, atol=1e-5), "Loss does not match expected value!"


def test_trainable_weights_property():
    """Ensure trainable_weights property returns expected weights."""
    model = AttentionBasedContextEmbedding(epochs=1, lr=0.01, latent_size=5, n_negative_samples=1)
    model.instantiate(n_items=6)
    weights = model.trainable_weights

    assert len(weights) == 4, "Expected 4 trainable weight tensors"
    assert weights[0] is model.Wi, "First weight should be Wi"
    assert weights[1] is model.wa, "Second weight should be wa"
    assert weights[2] is model.Wo, "Third weight should be Wo"
    assert weights[3] is model.empty_context_embedding, (
        "Fourth weight should be empty_context_embedding"
    )


def test_count_items_occurrences():
    """Test item occurrence counting functionality."""
    n_items = 4

    # Create simple dataset
    trips = [
        Trip(np.array([0, 1]), np.ones(n_items), np.ones(n_items)),
        Trip(np.array([1, 2]), np.ones(n_items), np.ones(n_items)),
        Trip(np.array([0]), np.ones(n_items), np.ones(n_items)),
    ]
    assortment_matrix = np.ones((1, n_items))
    dataset = TripDataset(trips, assortment_matrix)

    model = AttentionBasedContextEmbedding(epochs=1, lr=0.01, latent_size=2, n_negative_samples=1)
    model.instantiate(n_items=n_items)

    distribution = model._get_items_frequencies(dataset)

    # Item 0 appears 2 times, item 1 appears 2 times, item 2 appears 1 time, item 3 appears 0 times
    expected_counts = np.array([2, 2, 1, 0]) / 5  # Total = 5 items across all baskets

    assert np.allclose(distribution.numpy(), expected_counts), (
        "Item occurrence distribution mismatch"
    )
    assert np.allclose(tf.reduce_sum(distribution), 1.0), "Distribution should sum to 1"


def test_optimizer_initialization():
    """Test optimizer initialization with different options."""
    # Test Adam optimizer (default)
    model1 = AttentionBasedContextEmbedding(
        epochs=1, lr=0.01, latent_size=2, n_negative_samples=1, optimizer="Adam"
    )
    assert isinstance(model1.optimizer, tf.keras.optimizers.Adam), "Should use Adam optimizer"

    # Test unsupported optimizer (should fall back to Adam)
    model2 = AttentionBasedContextEmbedding(
        epochs=1, lr=0.01, latent_size=2, n_negative_samples=1, optimizer="zblub"
    )
    assert isinstance(model2.optimizer, tf.keras.optimizers.Adam), "Should fall back to Adam"
