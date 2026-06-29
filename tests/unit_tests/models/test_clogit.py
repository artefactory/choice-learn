"""Tests for the ConditionalLogit model."""

import numpy as np
import tensorflow as tf

from choice_learn.data import ChoiceDataset
from choice_learn.models import ConditionalLogit

test_dataset = ChoiceDataset(
    shared_features_by_choice=(np.array([[1, 3, 0], [0, 3, 1], [3, 2, 1], [3, 3, 1]]),),
    items_features_by_choice=(
        np.array(
            [
                [[1.1, 2.2], [2.9, 3.3], [3.3, 4.4]],
                [[1.2, 3.3], [2.3, 2.2], [4.3, 4.5]],
                [[1.4, 3.1], [2.4, 4.5], [3.4, 2.1]],
                [[1.7, 3.3], [2.3, 4.4], [3.7, 2.2]],
            ]
        ),
    ),
    items_features_by_choice_names=(["if1", "if2"],),
    shared_features_by_choice_names=(["sf1", "sf2", "sf3"],),
    available_items_by_choice=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0]]),
    choices=[0, 1, 2, 0],
)


def test_clogit_adam():
    """Tests the ConditionalLogit model with Adam optimizer."""
    tf.config.run_functions_eagerly(True)
    model = ConditionalLogit(optimizer="Adam", epochs=12, batch_size=-1)
    model.add_coefficients(feature_name="sf1", items_indexes=[0, 1, 2])
    model.add_coefficients(feature_name="sf2", items_indexes=[1, 2])
    model.add_shared_coefficient(feature_name="if1", items_indexes=[0, 1, 2])
    model.add_shared_coefficient(feature_name="if2", items_indexes=[0, 2])
    model.instantiate(test_dataset)

    nll_a = model.evaluate(test_dataset)
    model.fit(test_dataset)
    nll_b = model.evaluate(test_dataset)
    assert nll_b < nll_a
