"""Contain unit tests for the AttentionBasedContextEmbedding model."""

from pathlib import Path

import numpy as np
import tensorflow as tf

from choice_learn.basket_models.attn_model import AttentionBasedContextEmbedding
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
    custom_context_emb = model.context_embed(ragged_batch)
    assert isinstance(custom_context_emb, tf.Tensor), "Context embedding should be a Tensor"
    custom_context_emb = model.context_embed(ragged_batch).numpy()
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
    custom_context_emb = model.context_embed(ragged_batch)
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


def test_evaluate_save_load():
    """Test the evaluate method.

    This method should evaluate the model on the training dataset and save the model.
    """
    model.fit(train_trip_dataset)
    loss = model.evaluate(train_trip_dataset)
    assert isinstance(loss, np.float32), "Loss should be a float"
    model.save_model("test_model")

    loaded_model = AttentionBasedContextEmbedding(
        epochs=epochs,
        lr=lr,
        embedding_dim=embedding_dim,
        n_negative_samples=n_negative_samples,
    )
    loaded_model.load_model("test_model")
    epsilon = 1e-3

    for w1, w2 in zip(model.trainable_weights, loaded_model.trainable_weights):
        assert np.allclose(w1.numpy(), w2.numpy(), atol=epsilon), (
            f"Loaded model weights do not match original model weights. \
                {w1.numpy()} != {w2.numpy()}"
        )
    Path("test_model").unlink()
