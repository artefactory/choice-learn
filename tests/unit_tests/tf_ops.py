"""Test the tf_ops functions and classes."""

import numpy as np
from choice_learn.tf_ops import softmax_with_availabilities, CustomCategoricalCrossEntropy

def test_softmax():
    """Test the softmax function."""
    logits = np.array([[1, 1, 1, 1],
                       [2, 1, 2, 1],
                       [np.log(0.125), np.log(0.125), np.log(0.25), np.log(0.5)]])
    availabilities = np.array([[1., 1., 1., 1.],
                               [1., 0., 1., 0.],
                               [1., 1., 1., 1.]])
    probabilities = np.array([[0.25, 0.25, 0.25, 0.25],
                              [0.5, 0.0, 0.5, 0.0],
                              [0.125, 0.125, 0.250, 0.5]])
    
    assert (softmax_with_availabilities(items_logit_by_choice=logits, available_items_by_choice=availabilities) == probabilities).all()