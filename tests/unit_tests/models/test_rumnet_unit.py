"""Basic tests for the RUMnet model."""

import numpy as np
import tensorflow as tf

from choice_learn.models.rumnet import (
    AssortmentParallelDense,
    AssortmentUtilityDenseNetwork,
    ParallelDense,
)


def test_parallel_dense():
    """Tests the ParallelDense Layer."""
    tf.config.run_functions_eagerly(True)

    layer = ParallelDense(
        width=8,
        depth=4,
        heterogeneity=2,
        activation="relu",
    )
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    output_tensor = layer(input_tensor)
    assert output_tensor.shape == (3, 8, 2)
    assert len(layer.trainable_variables) == 8

    for i, w in enumerate(layer.trainable_variables):
        if i == 0:
            assert w.shape == (2, 8, 2)
        elif i % 2 == 0:
            assert w.shape == (8, 8, 2)
        else:
            assert w.shape == (8, 2)


def test_assortment_utility_dense():
    """Tests the AssortmentUtilityDenseNetwork Layer."""
    tf.config.run_functions_eagerly(True)

    layer = AssortmentUtilityDenseNetwork(
        width=4,
        depth=2,
        add_last=True,
        activation="relu",
    )
    input_tensor = np.array(
        [
            [[[1.0, 2.0], [1.4, 2.4]], [[3.0, 4.0], [3.4, 4.4]], [[5.0, 6.0], [5.4, 6.4]]],
            [[[1.1, 2.1], [1.5, 2.5]], [[3.1, 4.1], [3.5, 4.5]], [[5.1, 6.1], [5.5, 6.5]]],
            [[[1.2, 2.2], [1.6, 2.6]], [[3.2, 4.2], [3.6, 4.6]], [[5.2, 6.2], [5.6, 6.6]]],
            [[[1.3, 2.3], [1.7, 2.7]], [[3.3, 4.3], [3.7, 4.7]], [[5.3, 6.3], [5.7, 6.7]]],
        ]
    )
    output_tensor = layer(input_tensor)
    assert output_tensor.shape == (4, 3, 1, 2)
    assert len(layer.trainable_variables) == 5

    for i, w in enumerate(layer.trainable_variables):
        if i == 0:
            assert w.shape == (2, 4)
        elif i == 4:
            assert w.shape == (4, 1)
        elif i % 2 == 0:
            assert w.shape == (4, 4)
        else:
            assert w.shape == (4, 1)


def test_assortment_parallel_dense():
    """Tests the AssortmentParallelDense Layer."""
    tf.config.run_functions_eagerly(True)

    layer = AssortmentParallelDense(
        width=8,
        depth=4,
        heterogeneity=2,
        activation="relu",
    )
    input_tensor = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]],
            [[1.2, 2.2], [3.2, 4.2], [5.2, 6.2]],
            [[1.3, 2.3], [3.3, 4.3], [5.3, 6.3]],
        ]
    )
    output_tensor = layer(input_tensor)
    assert output_tensor.shape == (4, 3, 8, 2)
    assert len(layer.trainable_variables) == 8

    for i, w in enumerate(layer.trainable_variables):
        if i == 0:
            assert w.shape == (2, 8, 2)
        elif i % 2 == 0:
            assert w.shape == (8, 8, 2)
        else:
            assert w.shape == (8, 2)
