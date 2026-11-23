"""Test the losses functions and classes."""

import numpy as np

from choice_learn.utils.losses import (
    CustomCategoricalCrossEntropy,
    ExactCategoricalCrossEntropy,
    NoiseConstrastiveEstimation,
)


def test_crossentropy_smoothing():
    """Test label smoothing of softmax."""
    smoothed_loss = CustomCategoricalCrossEntropy(label_smoothing=0.1)
    exact_loss = CustomCategoricalCrossEntropy()

    assert exact_loss([[0.05, 0.95]], [[0.9, 0.1]]) == smoothed_loss([[0.0, 1.0]], [[0.9, 0.1]])


def test_custom_categorical_crossentropy():
    """Test the CustomCategoricalCrossEntropy loss class."""
    loss = CustomCategoricalCrossEntropy()
    assert loss([[0.0, 1.0]], [[0.0, 1.0]]) == 0.0
    assert loss([[0.0, 1.0]], [[1.0, 0.0]]) > 20.0

    loss = CustomCategoricalCrossEntropy(sparse=True)
    assert loss([1], [[0.0, 1.0]]) == 0.0
    assert loss([1], [[1.0, 0.0]]) > 20.0

    loss = CustomCategoricalCrossEntropy(sparse=True, from_logits=True)
    assert loss([1], [[21.0, 0.0]]) > 20.0


def test_exact_categorical_crossentropy():
    """Test the CustomCategoricalCrossEntropy loss class."""
    loss = ExactCategoricalCrossEntropy()
    assert loss([[0.0, 1.0]], [[0.0, 1.0]]) == 0.0
    assert loss([[0.0, 1.0]], [[1.0, 0.0]]) > 20.0

    loss = ExactCategoricalCrossEntropy(sparse=True)
    assert loss([1], [[0.0, 1.0]]) == 0.0
    assert loss([1], [[1.0, 0.0]]) > 20.0

    loss = ExactCategoricalCrossEntropy(sparse=True, from_logits=True)
    assert loss([0], [[21.0, 0.0]]) < 1e-3
    assert loss([1], [[21.0, 0.0]]) > 20.0
    assert loss([1], [[2.0, 4.0]]) > 0.1269
    assert loss([1], [[2.0, 4.0]]) < 0.1270
    assert loss([1], [[4.0, 2.0]]) > 2.1269
    assert loss([1], [[4.0, 2.0]]) < 2.1270


def test_nce():
    """Test the Noise Constrastice Estimation Loss."""
    loss = NoiseConstrastiveEstimation()
    loss_ref = loss(
        logit_true=np.array([10.0, 10.0]).astype("float32"),
        logit_negative=np.array([[0.0, 0.0], [0.0, 0.0]]).astype("float32"),
        freq_true=np.array([0.1, 0.1]).astype("float32"),
        freq_negative=np.array([[0.1, 0.1], [0.1, 0.1]]).astype("float32"),
    )

    loss_more = loss(
        logit_true=np.array([10.0, 10.0]).astype("float32"),
        logit_negative=np.array([[0.0, 0.0], [0.0, 0.0]]).astype("float32"),
        freq_true=np.array([0.8, 0.8]).astype("float32"),
        freq_negative=np.array([[0.1, 0.1], [0.1, 0.1]]).astype("float32"),
    )
    assert loss_more > loss_ref

    loss_more = loss(
        logit_true=np.array([10.0, 10.0]).astype("float32"),
        logit_negative=np.array([[0.0, 0.0], [0.0, 0.0]]).astype("float32"),
        freq_true=np.array([0.1, 0.1]).astype("float32"),
        freq_negative=np.array([[0.01, 0.1], [0.1, 0.01]]).astype("float32"),
    )
    assert loss_more > loss_ref

    loss_less = loss(
        logit_true=np.array([12.0, 12.0]).astype("float32"),
        logit_negative=np.array([[0.0, 0.0], [0.0, 0.0]]).astype("float32"),
        freq_true=np.array([0.1, 0.1]).astype("float32"),
        freq_negative=np.array([[0.1, 0.1], [0.1, 0.1]]).astype("float32"),
    )
    assert loss_less < loss_ref

    loss_less = loss(
        logit_true=np.array([10.0, 10.0]).astype("float32"),
        logit_negative=np.array([[-4.0, 0.0], [0.0, -4.0]]).astype("float32"),
        freq_true=np.array([0.1, 0.1]).astype("float32"),
        freq_negative=np.array([[0.1, 0.1], [0.1, 0.1]]).astype("float32"),
    )
    assert loss_less < loss_ref

    loss_ref = loss(
        logit_true=np.array([10.0, 10.0]).astype("float32"),
        logit_negative=np.array([[0.0, 0.0], [0.0, 0.0]]).astype("float32"),
        freq_true=np.array([0.1, 0.1]).astype("float32"),
        freq_negative=np.array([[0.1, 0.1], [0.1, 0.1]]).astype("float32"),
        sample_weight=np.array([1.0, 1.0]),
    )

    loss_more = loss(
        logit_true=np.array([10.0, 10.0]).astype("float32"),
        logit_negative=np.array([[0.0, 0.0], [0.0, 0.0]]).astype("float32"),
        freq_true=np.array([0.8, 0.8]).astype("float32"),
        freq_negative=np.array([[0.1, 0.1], [0.1, 0.1]]).astype("float32"),
        sample_weight=np.array([1.0, 1.0]),
    )
    assert loss_more > loss_ref

    loss_ref = loss(
        logit_true=np.array([10.0, 10.0]).astype("float32"),
        logit_negative=np.array([[0.0, 0.0], [0.0, 0.0]]).astype("float32"),
        freq_true=np.array([0.1, 0.1]).astype("float32"),
        freq_negative=np.array([[0.1, 0.1], [0.1, 0.1]]).astype("float32"),
        sample_weight=np.array([1.0, 1.0]),
    )

    loss_less = loss(
        logit_true=np.array([10.0, 10.0]).astype("float32"),
        logit_negative=np.array([[0.0, 0.0], [0.0, 0.0]]).astype("float32"),
        freq_true=np.array([0.8, 0.8]).astype("float32"),
        freq_negative=np.array([[0.1, 0.1], [0.1, 0.1]]).astype("float32"),
        sample_weight=np.array([0.1, 0.1]),
    )
    assert loss_more > loss_ref


def test_sample_weights():
    """Test sample_weight parametrization of CCE loss."""
    exact_nll = CustomCategoricalCrossEntropy(
        from_logits=False,
        label_smoothing=0.0,
        sparse=False,
        axis=-1,
        epsilon=1e-35,
        name="exact_categorical_crossentropy",
        reduction="sum_over_batch_size",
    )

    value_ref = exact_nll(
        y_true=[[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]], y_pred=[[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]]
    )
    value_1 = exact_nll(
        y_true=[[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
        y_pred=[[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]],
        sample_weight=[1.0, 1.0, 1.0],
    )
    value_2 = exact_nll(
        y_true=[[0.0, 1.0], [1.0, 0.0]], y_pred=[[0.2, 0.8], [0.2, 0.8]], sample_weight=[1.0, 2.0]
    )
    value_3 = exact_nll(
        y_true=[[0.0, 1.0], [1.0, 0.0]], y_pred=[[0.2, 0.8], [0.2, 0.8]], sample_weight=[0.5, 1.0]
    )

    assert value_ref == value_1
    assert value_ref == value_2 * 2 / 3
    assert value_ref == value_3 * 4 / 3
