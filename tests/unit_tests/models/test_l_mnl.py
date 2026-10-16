"""Basic tests for the Learning-MNL model."""

import numpy as np
import tensorflow as tf

from choice_learn.data import ChoiceDataset
from choice_learn.models.learning_mnl import LearningMNL

dataset = ChoiceDataset(
    items_features_by_choice=(
        np.array(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]],
                [[1.2, 2.2], [3.2, 4.2], [5.2, 6.2]],
                [[1.3, 2.3], [3.3, 4.3], [5.3, 6.3]],
            ]
        ).astype("float32"),
    ),
    items_features_by_choice_names=(["if1", "if2"],),
    shared_features_by_choice=(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]).astype(
            "float32"
        ),
    ),
    shared_features_by_choice_names=(["sf1", "sf2", "sf3"],),
    choices=np.array([0, 1, 2, 0]),
)


def test_l_mnl():
    """Tests the L-MNL model."""
    tf.config.run_functions_eagerly(True)
    global dataset

    model = LearningMNL(
        nn_features=["sf1", "sf3"],
        nn_layers_widths=[4, 4],
        optimizer="adam",
        lr=0.01,
        epochs=3,
    )
    model.add_coefficients(feature_name="sf2", items_indexes=[0, 1, 2])
    model.add_shared_coefficient(feature_name="if2", items_indexes=[0, 1, 2])
    model.add_coefficients(feature_name="intercept", items_indexes=[1, 2])
    model.instantiate(dataset)
    assert model.batch_predict(
        dataset.shared_features_by_choice[0],
        dataset.items_features_by_choice[0],
        np.ones((4, 3)),
        dataset.choices,
        None,
    )[1].shape == (4, 3)
    nll_a = model.evaluate(dataset)
    model.fit(dataset)
    nll_b = model.evaluate(dataset)
    assert nll_b < nll_a

    assert model.batch_predict(
        (dataset.shared_features_by_choice[0],),
        (dataset.items_features_by_choice[0],),
        np.ones((4, 3)),
        dataset.choices,
        None,
    )[1].shape == (4, 3)


def test_clone():
    """Tests the clone method."""
    tf.config.run_functions_eagerly(True)
    global dataset

    model = LearningMNL(
        nn_features=["sf1", "sf3"],
        nn_layers_widths=[4, 4],
        optimizer="adam",
        lr=0.01,
        epochs=3,
    )
    model.add_coefficients(feature_name="sf2", items_indexes=[0, 1, 2])
    model.add_shared_coefficient(feature_name="if2", items_indexes=[0, 1, 2])
    model.add_coefficients(feature_name="intercept", items_indexes=[1, 2])
    model.instantiate(dataset)

    model_clone = model.clone()
    assert np.sum(model_clone.predict_probas(dataset) - model.predict_probas(dataset)) < 1e-5
