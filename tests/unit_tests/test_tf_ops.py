"""Test the tf_ops functions and classes."""

import numpy as np

from choice_learn.tf_ops import (
    CustomCategoricalCrossEntropy,
    ExactCategoricalCrossEntropy,
    NoiseConstrastiveEstimation,
    softmax_with_availabilities,
)


def test_softmax():
    """Test the softmax function."""
    logits = np.array(
        [[1, 1, 1, 1], [2, 1, 2, 1], [np.log(0.125), np.log(0.125), np.log(0.25), np.log(0.5)]]
    )
    availabilities = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]])

    probabilities = np.array(
        [[0.25, 0.25, 0.25, 0.25], [0.5, 0.0, 0.5, 0.0], [0.125, 0.125, 0.250, 0.5]]
    )

    softmax_probabilities = softmax_with_availabilities(
        items_logit_by_choice=logits, available_items_by_choice=availabilities
    ).numpy()
    assert (np.abs(softmax_probabilities - probabilities) < 0.01).all()


def test_softmax_exit():
    """Test the softmax function with normalized exit."""
    logits = np.array(
        [[np.log(1.0), np.log(1.0), np.log(1.0), np.log(1.0)], [np.log(2.0), 1.0, np.log(2.0), 1.0]]
    )
    availabilities = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.0]])

    probabilities = np.array([[0.20, 0.20, 0.20, 0.20], [0.4, 0.0, 0.4, 0.0]])

    softmax_probabilities = softmax_with_availabilities(
        items_logit_by_choice=logits, available_items_by_choice=availabilities, normalize_exit=True
    ).numpy()
    assert (np.abs(softmax_probabilities - probabilities) < 0.01).all()


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
        logit_true=np.array([10.0, 10.0]),
        logit_negative=np.array([[0.0, 0.0], [0.0, 0.0]]),
        freq_true=np.array([0.1, 0.1]),
        freq_negative=np.array([[0.1, 0.1], [0.1, 0.1]]),
    )

    loss_more = loss(
        logit_true=np.array([10.0, 10.0]),
        logit_negative=np.array([[0.0, 0.0], [0.0, 0.0]]),
        freq_true=np.array([0.8, 0.8]),
        freq_negative=np.array([[0.1, 0.1], [0.1, 0.1]]),
    )
    assert loss_more > loss_ref

    loss_more = loss(
        logit_true=np.array([10.0, 10.0]),
        logit_negative=np.array([[0.0, 0.0], [0.0, 0.0]]),
        freq_true=np.array([0.1, 0.1]),
        freq_negative=np.array([[0.01, 0.1], [0.1, 0.01]]),
    )
    assert loss_more > loss_ref

    loss_less = loss(
        logit_true=np.array([12.0, 12.0]),
        logit_negative=np.array([[0.0, 0.0], [0.0, 0.0]]),
        freq_true=np.array([0.1, 0.1]),
        freq_negative=np.array([[0.1, 0.1], [0.1, 0.1]]),
    )
    assert loss_less < loss_ref

    loss_less = loss(
        logit_true=np.array([10.0, 10.0]),
        logit_negative=np.array([[-4.0, 0.0], [0.0, -4.0]]),
        freq_true=np.array([0.1, 0.1]),
        freq_negative=np.array([[0.1, 0.1], [0.1, 0.1]]),
    )
    assert loss_less < loss_ref
