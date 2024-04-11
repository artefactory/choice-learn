"""Testing that model instantiation works as expected."""

from choice_learn.datasets import load_modecanada
from choice_learn.models import ConditionalLogit, RUMnet, SimpleMNL, TasteNet

canada_dataset = load_modecanada(as_frame=False, preprocessing="tutorial")


def test_clogit_dict():
    """Tests cLogit instantiation with coefficients dictionary."""
    coefficients = {
        "income": "item",
        "cost": "constant",
        "freq": "constant",
        "ovt": "constant",
        "ivt": "item-full",
        "intercept": "item",
    }

    cmnl = ConditionalLogit(coefficients=coefficients)
    cmnl.instantiate(canada_dataset)
    assert True


def test_clogit_specification():
    """Tests cLogit instantiation with MNLCoefficient specification."""
    model = ConditionalLogit()
    model.add_shared_coefficient(feature_name="cost", items_indexes=[0, 1, 2, 3])
    model.add_shared_coefficient(
        feature_name="freq", coefficient_name="beta_frequence", items_indexes=[0, 1, 2, 3]
    )
    model.add_shared_coefficient(feature_name="ovt", items_indexes=[0, 1, 2, 3])
    model.add_coefficients(feature_name="ivt", items_indexes=[0, 1, 2, 3])
    model.add_coefficients(feature_name="intercept", items_indexes=[1, 2, 3])
    model.add_coefficients(feature_name="income", items_indexes=[1, 2, 3])
    model.instantiate(canada_dataset)
    assert True


def test_simplemnl_instantiation():
    """Tests SimpleMNL instantiation."""
    model = SimpleMNL(intercept="item-full")
    model.instantiate(n_items=4, n_items_features=10, n_shared_features=20)
    assert True


def test_rumnet_instantiation():
    """Tests RUMnet instantiation."""
    model_args = {
        "num_products_features": 6,
        "num_customer_features": 83,
        "width_eps_x": 20,
        "depth_eps_x": 5,
        "heterogeneity_x": 10,
        "width_eps_z": 20,
        "depth_eps_z": 5,
        "heterogeneity_z": 10,
        "width_u": 20,
        "depth_u": 5,
        "optimizer": "Adam",
        "lr": 0.0002,
        "logmin": 1e-10,
        "label_smoothing": 0.02,
        "callbacks": [],
        "epochs": 140,
        "batch_size": 32,
        "tol": 0,
    }
    model = RUMnet(**model_args)
    model.instantiate()
    assert True


def test_tastenet_intstantiation():
    """Tests TasteNet instantiation."""
    taste_net_layers = []
    taste_net_activation = "relu"
    items_features_by_choice_parametrization = [
        [-1.0, "-exp", "-exp", 0.0, "linear", 0.0, 0.0],
        [-1.0, "-exp", "-exp", "linear", 0.0, "linear", 0.0],
        [-1.0, "-exp", 0.0, 0.0, 0.0, 0.0, 0.0],
    ]

    tastenet = TasteNet(
        taste_net_layers=taste_net_layers,
        taste_net_activation=taste_net_activation,
        items_features_by_choice_parametrization=items_features_by_choice_parametrization,
        optimizer="Adam",
        epochs=40,
        lr=0.001,
        batch_size=32,
    )
    tastenet.instantiate(n_shared_features=17)
    assert True
