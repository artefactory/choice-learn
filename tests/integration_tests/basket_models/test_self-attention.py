"""Integration tests for SelfAttentionModel."""

import numpy as np
import pytest
import tensorflow as tf

from choice_learn.basket_models.datasets import SyntheticDataGenerator
from choice_learn.basket_models.self_attention_model import SelfAttentionModel

items_nest = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}


nests_interactions = [
    ["", "compl", "neutral", "neutral"],
    ["compl", "", "neutral", "neutral"],
    ["neutral", "neutral", "", "neutral"],
]

user_profile = {0: {"nest": 0, "item": 0}, 1: {"nest": 0, "item": 1}, 2: {"nest": 0, "item": 2}}

data = SyntheticDataGenerator(
    items_nest=items_nest,
    nests_interactions=nests_interactions,
    proba_complementary_items=1,
    proba_neutral_items=0.0,
    noise_proba=0.0,
    user_profile=user_profile,
)

data = data.generate_trip_dataset(n_baskets=1000, assortments_matrix=np.ones((1, 9)))


def test_get_negative_samples() -> None:
    """Test the get_negative_samples method."""
    model = SelfAttentionModel()
    model.instantiate(
        n_items=data.n_items,
        n_users=data.n_users,
    )

    with pytest.raises(tf.errors.InvalidArgumentError):
        neg_samples = model.get_negative_samples(
            available_items=np.ones(data.n_items),
            purchased_items=np.array([1, 2]),
            next_item=0,
            n_samples=data.n_items,  # Too many samples
        )
        for item in [0, 1, 2]:
            assert item not in neg_samples


def test_fit() -> None:
    """Test the fit method."""
    model = SelfAttentionModel()
    model.instantiate(
        n_items=data.n_items,
        n_users=data.n_users,
    )
    # Test lazy instantiation + verbose + batch_size=-1
    model.fit(trip_dataset=data, val_dataset=data, verbose=1)


def test_mask_attention() -> None:
    """Test the masked_attention method."""
    model = SelfAttentionModel()
    model.instantiate(
        n_items=data.n_items,
        n_users=data.n_users,
    )

    attention_weights = model.masked_attention(
        basket_batch=tf.constant([[0, 3, 7], [1, 3, 9]]),
        scaled_scores=tf.constant(
            [
                [[1.2, 0.8, 0.1], [0.3, 1.0, 0.0], [0.1, 0.5, 2.9]],
                [[1.2, 0.8, 0.1], [0.3, 1.0, 0.0], [0.1, 0.5, 2.9]],
            ],
        ),
    )  # Shape: (batch_size, L, L)
    # Check attention weights shape
    assert attention_weights.shape == (2, 3, 3)

    for i in range(3):
        # Check that attention weights of the diagonal are zero and rows sum to 1
        assert attention_weights[0, i, i] == 0.0
        assert np.abs(np.sum(attention_weights[0], axis=1)[i] - 1.0) < 1e-4

        assert attention_weights[1, i, i] == 0.0
        assert np.abs(np.sum(attention_weights[1], axis=1)[i] - 1.0) < 1e-4

        # Check that attention weights for padding item is zero all along the row
        assert attention_weights[1, i, 2] == 0.0

    # Test the special case where all items except one are padding
    attention_weights = model.masked_attention(
        basket_batch=tf.constant([[0, 9, 9]]), scaled_scores=tf.constant([[1.2, 0.8, 0.8]])
    )

    assert attention_weights.shape == (1, 3, 3)
    assert (
        attention_weights[0].numpy().all()
        == tf.constant([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).numpy().all()
    )


def test_embed_context() -> None:
    """Test the embed_context method."""
    model = SelfAttentionModel(latent_sizes={"short_term": 5, "long_term": 2})
    model.instantiate(
        n_items=data.n_items,
        n_users=data.n_users,
    )

    m_batch, attention_weights = model.embed_basket(
        context_items=tf.constant([[0, 6, 3], [1, 3, 7]]), is_training=False
    )
    assert m_batch.shape == (2, 5)  # Shape = (batch_size, short_term_latent_size)
    assert attention_weights.shape == (
        2,
        3,
        3,
    )  # Shape = (batch_size, length of basket, length of basket)


def test_compute_distance() -> None:
    """Test the compute_distance method."""
    model = SelfAttentionModel()
    model.instantiate(
        n_items=data.n_items,
        n_users=data.n_users,
    )

    m_batch, _ = model.embed_basket(
        context_items=tf.constant([[0, 6, 3], [1, 3, 7]]), is_training=False
    )
    distances = model.compute_batch_distance(
        item_batch=tf.constant([1, 3]),
        m_batch=m_batch,
        user_batch=tf.constant([0, 1]),
    )

    assert distances.shape == (2,)  # Shape = (batch_size,)


def test_compute_loss() -> None:
    """Test the compute_loss method."""
    model = SelfAttentionModel()
    model.instantiate(
        n_items=data.n_items,
        n_users=data.n_users,
    )

    batch_size = 2
    loss, _ = model.compute_batch_loss(
        item_batch=tf.constant([1, 3]),
        basket_batch=tf.constant([[0, 3, 6], [1, 4, 7]]),
        future_batch=None,
        store_batch=None,
        week_batch=None,
        price_batch=None,
        available_item_batch=[[1] * data.n_items] * batch_size,
        user_batch=tf.constant([0, 1]),
    )
    assert loss.dtype == tf.float32  # Scalar loss


def hit_rate(all_distances, item_batch, hit_k):
    """Compute the hit rate at k for the given distances."""
    hit_list = []
    for k in hit_k:
        top_k_indices = tf.math.top_k(-all_distances, k=k).indices  # Shape: (batch_size, hit_k)
        hits_per_batch = tf.reduce_any(
            tf.equal(
                tf.cast(top_k_indices, tf.int32),
                tf.cast(tf.expand_dims(item_batch, axis=1), tf.int32),
            ),
            axis=1,
        )
        hits = tf.reduce_sum(tf.cast(hits_per_batch, tf.float32))
        hit_list.append(hits)

    return tf.convert_to_tensor(hit_list)


def test_hit_rate():
    """Test the hit_rate method."""
    model = SelfAttentionModel()
    model.instantiate(
        n_items=data.n_items,
        n_users=data.n_users,
    )

    hr = hit_rate(
        all_distances=tf.constant([[0.1, 9.9, 0.2, 9.9], [9.9, 0.1, 9.9, 0.2]]),
        item_batch=np.array([3, 1]),
        hit_k=[1, 2],
    )

    assert hr.shape == (2,)  # Scalar hit rate
    assert hr[0].numpy() == 1.0  # Hit@1
    assert hr[1].numpy() == 1.0  # Hit@2


def mean_reciprocal_rank(all_distances, item_batch, _):
    """Compute the mean reciprocal rank for the given distances."""
    batch_size = tf.shape(item_batch)[0]
    ranks = (
        tf.argsort(tf.argsort(all_distances, axis=1), axis=1) + 1
    )  # Shape: (batch_size, n_items)
    item_batch_indices = tf.stack(
        [tf.range(batch_size), item_batch], axis=1
    )  # Shape: (batch_size, 2)
    item_ranks = tf.gather_nd(ranks, item_batch_indices)  # Shape: (batch_size,)

    return tf.reduce_sum(tf.cast(1 / item_ranks, dtype=tf.float32))


def test_mrr():
    """Test the mrr method."""
    model = SelfAttentionModel()
    model.instantiate(
        n_items=data.n_items,
        n_users=data.n_users,
    )
    mrr = mean_reciprocal_rank(
        all_distances=tf.constant([[0.1, 9.9, 0.2, 9.9], [9.9, 0.1, 9.9, 0.2]]),
        item_batch=np.array([3, 1]),
        _=None,
    )

    assert mrr.shape == ()  # Scalar MRR


def test_evaluate():
    """Test the evaluate method."""
    model = SelfAttentionModel()
    model.instantiate(
        n_items=data.n_items,
        n_users=data.n_users,
    )

    score = model.evaluate(
        trip_dataset=data,
        batch_size=32,
        hit_k=[1, 5],
        metrics=[mean_reciprocal_rank, hit_rate],
    )

    assert isinstance(score, dict)  # Score is a dictionary
