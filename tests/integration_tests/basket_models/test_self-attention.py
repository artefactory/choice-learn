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
    model = SelfAttentionModel(epochs=1)
    model.instantiate(n_items=data.n_items, n_users=data.n_users)
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
        basket_batch=tf.constant([[0, 6, 3], [1, 3, 7]]), is_training=False
    )
    assert m_batch.shape == (2, 5)  # Shape = (batch_size, short_term_latent_size)
    assert attention_weights.shape == (
        2,
        3,
        3,
    )  # Shape = (batch_size, length of basket, length of basket)

    model.X = [
        [-0.38314673, 1.2642082, -0.04192788, -0.54980755, -0.6190078],
        [1.29161, 0.4715454, -0.96005285, 1.2759234, -0.18724886],
        [0.6078521, -1.1132072, -0.00812762, -0.7213017, 0.63749003],
        [-2.0417902, -1.2831244, 0.07948236, -0.14204586, -1.2623296],
        [-0.12414064, 1.2700547, 1.1942786, 1.08837, 0.44708633],
        [1.8842306, 0.50381213, -1.2076277, 1.2309052, -0.9271277],
        [0.93309706, 0.65104663, -0.1238246, -2.235176, 0.8270921],
        [-0.7682185, -1.5417612, -0.5850125, 1.4384562, -0.95027596],
        [-0.4900044, 1.1855007, 1.2202692, 1.0480481, 1.3066727],
    ]
    model.Wq = [
        [0.08005401, 1.3216059, 0.5105733, -0.66905314, 0.19061662],
        [0.02782486, 0.17023416, 0.20554847, 0.9867587, -0.69792676],
        [0.59941924, -1.7979192, 0.21347761, 0.4268342, -0.5839359],
        [-1.4264288, -0.4373422, 1.3919772, -1.4671614, 1.0249721],
        [1.202534, 0.31624648, -0.5609615, 1.1609291, 0.30938107],
    ]
    model.Wk = [
        [1.3321146, -0.5260284, -0.31010678, 1.1888672, -0.72644967],
        [1.8431323, -1.8530965, 0.6781888, -0.76233566, -0.8803108],
        [-0.4999859, -0.441501, 0.41671997, -0.80546844, -1.3554088],
        [0.40653583, -1.1605769, 1.9338123, 1.4623568, 1.1258445],
        [0.5642827, -0.7798367, -1.4498279, 1.5575027, -2.616824],
    ]
    m_batch, attention_weights = model.embed_basket(basket_batch=[[0, 1, 2, 3]])

    np.testing.assert_almost_equal(
        m_batch.numpy(), [[-0.17117432, -0.2969993, -0.37172782, 0.32959175, -0.5147827]], decimal=5
    )
    np.testing.assert_almost_equal(
        attention_weights.numpy(),
        [
            [
                [0.0000000e00, 7.5905293e-01, 1.7927325e-01, 6.1673805e-02],
                [2.2249160e-05, 0.0000000e00, 1.1792018e-04, 9.9985981e-01],
                [4.5206651e-02, 5.5638880e-01, 0.0000000e00, 3.9840463e-01],
                [3.3333334e-01, 3.3333334e-01, 3.3333334e-01, 0.0000000e00],
            ]
        ],
        decimal=4,
    )


def test_compute_distance() -> None:
    """Test the compute_distance method."""
    model = SelfAttentionModel()
    model.instantiate(
        n_items=data.n_items,
        n_users=data.n_users,
    )

    distances = model.compute_batch_distance(
        item_batch=tf.constant([[1], [2]]),
        basket_batch=tf.constant([[0, 6, 3], [1, 3, 7]]),
        user_batch=tf.constant([0, 1]),
        is_training=False,
    )

    assert distances.shape == (2, 1)  # Shape = (batch_size,)


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
