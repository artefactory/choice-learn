"""Tests specific config of cLogit and .evaluate() with ground truth weight."""

import tensorflow as tf

from choice_learn.datasets import load_modecanada
from choice_learn.models import ConditionalLogit


def test_mode_canada_gt():
    """Tests specific config of cLogit and .evaluate()."""
    # Instantiation with the coefficients dictionnary
    coefficients = {
        "income": "item",
        "cost": "constant",
        "freq": "constant",
        "ovt": "constant",
        "ivt": "item-full",
        "intercept": "item",
    }

    canada_dataset = load_modecanada(as_frame=False, preprocessing="tutorial")

    # Here are the values obtained in the references:
    gt_weights = [
        tf.constant([[-0.0890796, -0.0279925, -0.038146]]),
        tf.constant([[-0.0333421]]),
        tf.constant([[0.0925304]]),
        tf.constant([[-0.0430032]]),
        tf.constant([[0.0595089, -0.00678188, -0.00645982, -0.00145029]]),
        tf.constant([[0.697311, 1.8437, 3.27381]]),
    ]
    gt_model = ConditionalLogit(coefficients=coefficients)
    gt_model.instantiate(canada_dataset)

    gt_model.trainable_weights = gt_weights
    assert (gt_model.evaluate(canada_dataset) * len(canada_dataset)) == 1874.3630792600002
