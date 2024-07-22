"""Testing dumb - baseline models."""

import numpy as np
import pytest

from choice_learn.data import ChoiceDataset
from choice_learn.models.baseline_models import DistribMimickingModel, RandomChoiceModel

np.random.seed(101)
shared_features = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [5.0, 3.0]])
items_features = np.array(
    [
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],
        [[7.0, 7.0], [8.0, 8.0], [9.0, 9.0]],
        [[10.0, 10.0], [11.0, 11.0], [12.0, 12.0]],
    ]
)
choices = np.array([0, 2, 1, 0])

dataset = ChoiceDataset(
    shared_features_by_choice=shared_features,
    items_features_by_choice=items_features,
    choices=choices,
)


def test_random_choice_model():
    """Test basic stuff about the RandomChoiceModel."""
    global dataset

    model = RandomChoiceModel()
    _ = model.fit(choice_dataset=dataset)

    y_pred = model.predict_probas(dataset)
    assert y_pred.shape == (4, 3)
    assert (np.sum(y_pred, axis=1) < 1.01).all()
    assert (np.sum(y_pred, axis=1) > 0.99).all()
    assert (y_pred >= 0).numpy().all()
    assert (y_pred <= 1.0).numpy().all()

    utility = model.compute_batch_utility(
        shared_features_by_choice=dataset.shared_features_by_choice,
        items_features_by_choice=dataset.items_features_by_choice,
        available_items_by_choice=np.ones((4, 3)),
        choices=dataset.choices,
    )
    assert utility.shape == (4, 3)


def test_mimicking_choice_model():
    """Test basic stuff about the DistribMimickingModel."""
    global dataset

    model = DistribMimickingModel()
    _ = model.fit(choice_dataset=dataset)
    print(model.is_fitted)
    y_pred = model.predict_probas(dataset)
    assert y_pred.shape == (4, 3)
    assert (np.sum(y_pred, axis=1) < 1.01).all()
    assert (np.sum(y_pred, axis=1) > 0.99).all()
    assert (y_pred >= 0).numpy().all()
    assert (y_pred <= 1.0).numpy().all()

    assert (np.abs(y_pred[:, 0] - 0.5) < 0.01).all()
    assert (np.abs(y_pred[:, 1] - 0.25) < 0.01).all()
    assert (np.abs(y_pred[:, 2] - 0.25) < 0.01).all()


def test_catch_not_fitted_issue():
    """Verify that the model raises an error if not fitted."""
    global dataset

    model = DistribMimickingModel()
    with pytest.raises(ValueError):
        model.predict_probas(dataset)
