"""Contain unit tests for the AttentionBasedContextEmbedding model."""

import numpy as np
import tensorflow as tf

from choice_learn.basket_models.basic_attention_model import AttentionBasedContextEmbedding
from choice_learn.basket_models.dataset import Trip, TripDataset
from choice_learn.basket_models.synthetic_dataset import SyntheticDataGenerator

# Test hyperparameters
epochs = 4
lr = 0.01
embedding_dim = 5
n_negative_samples = 4
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
    embedding_dim=embedding_dim,
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
    custom_context_emb = model.embed_context(ragged_batch)
    scores = model.score(custom_context_emb, tf.constant(target_items, dtype=tf.int32))
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
    negative_samples = model.get_negative_samples(ragged_batch, target_items, assortments_matrix[0])
    print("Negative samples", negative_samples)

    assert isinstance(negative_samples, tf.Tensor), "Negative samples should be a tensor"
    assert len(negative_samples[0]) <= n_negative_samples, "Negative samples length mismatch"

    for i in range(len(baskets)):
        basket = baskets[i]
        for item in basket:
            if item in negative_samples[i].numpy():
                print("basket: ", basket)
                print("negative_samples: ", negative_samples[i].numpy())
            assert item not in negative_samples[i].numpy(), (
                f"Item {item} should not be in negative samples"
            )


def test_predict():
    """
    Test the predict method of the AttentionBasedContextEmbedding model.

    This method should predict the probabilities of items
    in the assortment based on the context embeddings.
    """
    assortment = np.array([[1, 1, 1, 1, 1, 1, 1, 0]])
    dataset = data_gen.generate_trip_dataset(2, assortment)
    model = AttentionBasedContextEmbedding(
        epochs=40, lr=0.01, embedding_dim=4, n_negative_samples=5
    )

    contexts = []
    for batch in dataset.iter_batch(1, data_method="aleacarta"):
        contexts.append(batch[1][0])
    contexts = tf.ragged.constant([row[row != -1] for row in contexts], dtype=tf.int32)
    available_items = batch[-1][0]
    context_prediction = model.predict(contexts, available_items=available_items)
    assert np.allclose(context_prediction.sum(axis=1), 1, atol=1e-2), "Each row must sum to 1"
    for i in np.where(assortment[0] == 0)[0]:
        assert np.all(context_prediction[:, i] == 0), (
            f"Column {i} is not all zeros and {i} is not in assortment"
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
        embedding_dim=embedding_dim,
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
    eval_loss = model.evaluate(dataset)
    assert np.allclose(eval_loss, expected_loss, atol=1e-5), "Loss does not match expected value!"
