"""Test the tf_ops functions and classes."""

import numpy as np

from choice_learn.tf_ops import CustomCategoricalCrossEntropy, softmax_with_availabilities


def test_softmax():
    """Test the softmax function."""
    logits = np.array(
        [[1, 1, 1, 1], [2, 1, 2, 1], [np.log(0.125), np.log(0.125), np.log(0.25), np.log(0.5)]]
    )
    availabilities = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    """
    probabilities = np.array([[0.25, 0.25, 0.25, 0.25],
                              [0.5, 0.0, 0.5, 0.0],
                              [0.125, 0.125, 0.250, 0.5]])
    """
    softmax_probabilities = softmax_with_availabilities(
        items_logit_by_choice=logits, available_items_by_choice=availabilities
    ).numpy()
    assert (softmax_probabilities[0] < 0.26).all()
    assert (softmax_probabilities[0] > 0.24).all()

    assert (softmax_probabilities[1][[0, 2]] < 0.51).all()
    assert (softmax_probabilities[1][[0, 2]] > 0.49).all()
    assert (softmax_probabilities[1][[1, 3]] < 0.01).all()

    assert (softmax_probabilities[2][0] < 0.126).all()
    assert (softmax_probabilities[2][0] > 0.124).all()
    assert (softmax_probabilities[2][1] < 0.126).all()
    assert (softmax_probabilities[2][1] > 0.124).all()
    assert (softmax_probabilities[2][2] < 0.251).all()
    assert (softmax_probabilities[2][2] > 0.249).all()
    assert (softmax_probabilities[2][3] < 0.501).all()
    assert (softmax_probabilities[2][3] > 0.499).all()


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
