"""Contain unit tests for the AttentionBasedContextEmbedding model."""

from pathlib import Path

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

    assert isinstance(negative_samples, tf.Tensor), "Negative samples should be a tensor"
    assert len(negative_samples[0]) <= n_negative_samples, "Negative samples length mismatch"

    for i in range(len(baskets)):
        basket = baskets[i]
        for item in basket:
            if item in negative_samples[i].numpy():
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


def test_negative_log_likelihood_loss_matches_manual():
    """Test that NLL loss matches manual computation."""
    n_items = 4
    batch_size = 3
    emb_dim = 2

    model = AttentionBasedContextEmbedding(
        epochs=1, lr=0.01, embedding_dim=emb_dim, n_negative_samples=1
    )
    model.instantiate(n_items=n_items)

    context_vec = tf.ones((batch_size, emb_dim))
    target_items = tf.constant([0, 1, 2], dtype=tf.int32)

    # Manual computation
    logits = tf.matmul(context_vec, model.Wo, transpose_b=True)
    expected_loss = tf.keras.losses.sparse_categorical_crossentropy(
        target_items, logits, from_logits=True
    )

    # Model computation
    loss = model.negative_log_likelihood_loss(context_vec, target_items)

    assert np.allclose(loss.numpy(), expected_loss.numpy()), "NLL loss mismatch"


def test_trainable_weights_property():
    """Ensure trainable_weights property returns expected weights."""
    model = AttentionBasedContextEmbedding(epochs=1, lr=0.01, embedding_dim=5, n_negative_samples=1)
    model.instantiate(n_items=6)
    weights = model.trainable_weights

    assert len(weights) == 4, "Expected 4 trainable weight tensors"
    assert weights[0] is model.Wi, "First weight should be Wi"
    assert weights[1] is model.wa, "Second weight should be wa"
    assert weights[2] is model.Wo, "Third weight should be Wo"
    assert weights[3] is model.empty_context_embedding, (
        "Fourth weight should be empty_context_embedding"
    )


def test_instantiate_validation_errors():
    """Ensure instantiate raises appropriate errors for invalid inputs."""
    model = AttentionBasedContextEmbedding(epochs=1, lr=0.01, embedding_dim=4, n_negative_samples=1)

    # Test n_items <= 1
    try:
        model.instantiate(n_items=1)
        assert False, "Should have failed with n_items <= 1"
    except ValueError as e:
        assert "greater than 1" in str(e)

    # Test wrong distribution length
    try:
        model.instantiate(n_items=3, negative_samples_distribution=[0.5, 0.5])
        assert False, "Should have failed with wrong distribution length"
    except ValueError as e:
        assert "same length" in str(e)


def test_load_model_missing_file():
    """Ensure load_model raises FileNotFoundError for missing files."""
    model = AttentionBasedContextEmbedding(epochs=1, lr=0.01, embedding_dim=3, n_negative_samples=1)

    try:
        model.load_model("non_existent_file.json")
        assert False, "Should have failed for missing file"
    except FileNotFoundError:
        pass


def test_proba_positive_and_negative_samples():
    """Test probability calculations for positive and negative samples."""
    n_items = 4
    model = AttentionBasedContextEmbedding(epochs=1, lr=0.01, embedding_dim=2, n_negative_samples=2)
    model.instantiate(n_items=n_items)

    # Test positive samples
    pos_score = tf.constant([1.0, 2.0], dtype=tf.float32)
    target_items = tf.constant([0, 1], dtype=tf.int32)
    pos_proba = model.positive_samples_probability(pos_score, target_items)

    assert pos_proba.shape == (2,), "Positive probabilities shape mismatch"
    assert tf.reduce_all(pos_proba > 0) and tf.reduce_all(pos_proba <= 1), (
        "Probabilities should be in (0,1]"
    )

    # Test negative samples
    neg_score = tf.constant([[0.5, 1.5], [1.0, 2.0]], dtype=tf.float32)
    neg_items = tf.constant([[2, 3], [0, 2]], dtype=tf.int32)
    neg_proba = model.negative_samples_probability(neg_score, neg_items)

    assert neg_proba.shape == (2, 2), "Negative probabilities shape mismatch"
    assert tf.reduce_all(neg_proba >= 0) and tf.reduce_all(neg_proba < 1), (
        "Probabilities should be in [0,1)"
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

    model = AttentionBasedContextEmbedding(epochs=1, lr=0.01, embedding_dim=2, n_negative_samples=1)
    model.instantiate(n_items=n_items)

    distribution = model._get_items_frequencies(dataset)

    # Item 0 appears 2 times, item 1 appears 2 times, item 2 appears 1 time, item 3 appears 0 times
    expected_counts = np.array([2, 2, 1, 0]) / 5  # Total = 5 items across all baskets

    assert np.allclose(distribution.numpy(), expected_counts), (
        "Item occurrence distribution mismatch"
    )
    assert np.allclose(tf.reduce_sum(distribution), 1.0), "Distribution should sum to 1"


def test_nce_loss_computation():
    """Test NCE loss computation with known values."""
    n_items = 3
    model = AttentionBasedContextEmbedding(epochs=1, lr=0.01, embedding_dim=2, n_negative_samples=1)
    model.instantiate(n_items=n_items)

    pos_score = tf.constant([1.0, 2.0], dtype=tf.float32)
    target_items = tf.constant([0, 1], dtype=tf.int32)
    list_neg_items = tf.constant([[2], [0]], dtype=tf.int32)
    neg_scores = [tf.constant([0.5, 1.5], dtype=tf.float32)]

    loss = model.nce_loss(pos_score, target_items, list_neg_items, neg_scores)

    assert loss.shape == (2,), "NCE loss shape should match batch size"
    assert tf.reduce_all(loss > 0), "NCE loss should be positive"


def test_optimizer_initialization():
    """Test optimizer initialization with different options."""
    # Test Adam optimizer (default)
    model1 = AttentionBasedContextEmbedding(
        epochs=1, lr=0.01, embedding_dim=2, n_negative_samples=1, optimizer="Adam"
    )
    assert isinstance(model1.optimizer, tf.keras.optimizers.Adam), "Should use Adam optimizer"

    # Test unsupported optimizer (should fall back to Adam)
    model2 = AttentionBasedContextEmbedding(
        epochs=1, lr=0.01, embedding_dim=2, n_negative_samples=1, optimizer="SGD"
    )
    assert isinstance(model2.optimizer, tf.keras.optimizers.Adam), "Should fall back to Adam"


def test_load_save():
    """Test save and load model functionality."""
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
    # Create evaluation dataset
    trip_dataset_train = data_gen.generate_trip_dataset(10, assortments_matrix)

    model1 = AttentionBasedContextEmbedding(
        epochs=2, lr=0.01, embedding_dim=2, n_negative_samples=2
    )

    model1.instantiate(n_items=len(assortments_matrix[0]), use_true_nce_distribution=True)
    model1.fit(trip_dataset_train)

    # Save model
    model1.save_model("attn_model.json")

    # Create a model 3 without instantiating
    model3 = AttentionBasedContextEmbedding(
        epochs=epochs, lr=lr, embedding_dim=embedding_dim, n_negative_samples=n_negative_samples
    )

    # Load first model and compare results on evaluation dataset
    model3.load_model("attn_model.json")
    Path("attn_model.json").unlink()
    Path("attn_model_empty_context_embedding.npy").unlink()
    Path("attn_model_wa.npy").unlink()
    Path("attn_model_Wi.npy").unlink()
    Path("attn_model_Wo.npy").unlink()


def test_fit_basic_functionality():
    """Test basic functionality of the fit method."""
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

    basic_model = AttentionBasedContextEmbedding(
        epochs=4, lr=0.01, embedding_dim=2, n_negative_samples=2
    )
    train_dataset = data_gen.generate_trip_dataset(10, assortments_matrix)
    n_items = 8
    basic_model.instantiate(n_items=n_items)

    # Test that fit runs without errors
    history = basic_model.fit(train_dataset)

    # Verify return type and structure
    assert isinstance(history, dict), "fit() should return a dictionary"
    assert len(history["train_loss"]) == epochs, f"Loss history should have {epochs} entries"

    # Verify model is marked as trained
    assert basic_model.is_trained, "Model should be marked as trained after fit()"
