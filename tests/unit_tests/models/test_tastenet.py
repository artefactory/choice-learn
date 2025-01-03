"""Simple tests for the TasteNet model."""

import pytest

from choice_learn.datasets import load_swissmetro
from choice_learn.models import TasteNet

customers_id, dataset = load_swissmetro(preprocessing="tastenet", as_frame=False)
dataset = dataset[:20]

taste_net_layers = []
taste_net_activation = "relu"
items_features_by_choice_parametrization = [
    [-1.0, "-exp", "-exp", 0.0, "linear", 0.0, 0.0],
    [-1.0, "-exp", "-exp", "linear", 0.0, "linear", 0.0],
    [-1.0, "-exp", 0.0, 0.0, 0.0, 0.0, 0.0],
]


def test_activation():
    """Tests TasteNet activation."""
    tastenet = TasteNet(
        taste_net_layers=taste_net_layers,
        taste_net_activation=taste_net_activation,
        items_features_by_choice_parametrization=items_features_by_choice_parametrization,
        optimizer="Adam",
        epochs=40,
        lr=0.001,
        batch_size=32,
    )
    for act in ["linear", "relu", "-relu", "exp", "-exp", "tanh", "sigmoid"]:
        _ = tastenet.get_activation_function(act)
    with pytest.raises(ValueError):
        tastenet.get_activation_function("abc")


def test_fit_adam():
    """Test fit with Gradient Descent."""
    tastenet = TasteNet(
        taste_net_layers=taste_net_layers,
        taste_net_activation=taste_net_activation,
        items_features_by_choice_parametrization=items_features_by_choice_parametrization,
        optimizer="Adam",
        epochs=5,
        lr=0.001,
        batch_size=32,
    )
    hist = tastenet.fit(dataset)
    assert True


def test_fit_lbfgs():
    """Test fit with Gradient Descent."""
    tastenet = TasteNet(
        taste_net_layers=taste_net_layers,
        taste_net_activation=taste_net_activation,
        items_features_by_choice_parametrization=items_features_by_choice_parametrization,
        optimizer="lbfgs",
        epochs=5,
        lr=0.001,
        batch_size=32,
    )
    hist = tastenet.fit(dataset)
    assert True


def test_errors_raised():
    """Test diverse errors that should be raised."""
    with pytest.raises(ValueError):
        tastenet = TasteNet(
            taste_net_layers=taste_net_layers,
            taste_net_activation=taste_net_activation,
            items_features_by_choice_parametrization=[
                [-1.0, "-exp", "-exp", 0.0, "linear", 0.0, 0.0],
                [-1.0, "-exp", "-exp", "linear", 0.0, "linear", 0.0],
            ],
            optimizer="Adam",
            epochs=5,
            lr=0.001,
            batch_size=32,
        )
        hist = tastenet.fit(dataset)

    with pytest.raises(ValueError):
        tastenet = TasteNet(
            taste_net_layers=taste_net_layers,
            taste_net_activation=taste_net_activation,
            items_features_by_choice_parametrization=[
                [-1.0, "-exp", "-exp", 0.0, "linear", 0.0, 0.0, 0.0],
                [-1.0, "-exp", "-exp", "linear", 0.0, "linear", 0.0, 0.0],
                [-1.0, "-exp", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            optimizer="Adam",
            epochs=5,
            lr=0.001,
            batch_size=32,
        )
        hist = tastenet.fit(dataset)

    with pytest.raises(ValueError):
        tastenet = TasteNet(
            taste_net_layers=taste_net_layers,
            taste_net_activation=taste_net_activation,
            items_features_by_choice_parametrization=[
                [-1.0, "-exp", "-exp", 0.0, "linear", 0.0, 0.0],
                [-1.0, "-exp", "-exp", "linear", 0.0, "linear", 0.0],
            ],
            optimizer="lbfgs",
            epochs=5,
            lr=0.001,
            batch_size=32,
        )
        hist = tastenet.fit(dataset)

    with pytest.raises(ValueError):
        tastenet = TasteNet(
            taste_net_layers=taste_net_layers,
            taste_net_activation=taste_net_activation,
            items_features_by_choice_parametrization=[
                [-1.0, "-exp", "-exp", 0.0, "linear", 0.0, 0.0, 0.0],
                [-1.0, "-exp", "-exp", "linear", 0.0, "linear", 0.0, 0.0],
                [-1.0, "-exp", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            optimizer="lbfgs",
            epochs=5,
            lr=0.001,
            batch_size=32,
        )
        hist = tastenet.fit(dataset)
