"""Test the tf_ops functions and classes."""

import numpy as np

from choice_learn.tf_ops import CustomCategoricalCrossEntropy, ExactCategoricalCrossEntropy, softmax_with_availabilities


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
    assert (np.abs  (softmax_probabilities - probabilities) < 0.01).all()

def test_softmax_smoothing():
    """Test label smoothing of softmax."""
    pass


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
