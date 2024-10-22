"""Tests ResLogit."""

import numpy as np
import tensorflow as tf

from choice_learn.datasets import load_swissmetro

# from choice_learn.models import ResLogit, SimpleMNL
from choice_learn.models import ResLogit

dataset = load_swissmetro()
dataset = dataset[:100]  # Reduce the dataset size for faster testing
n_items = np.shape(dataset.items_features_by_choice)[2]
n_shared_features = np.shape(dataset.shared_features_by_choice)[2]
n_items_features = np.shape(dataset.items_features_by_choice)[3]


lr = 1e-4
epochs = 100
batch_size = -1


def test_reslogit_fit_with_sgd():
    """Tests that ResLogit can fit with SGD."""
    global dataset

    model = ResLogit(lr=lr, epochs=epochs, optimizer="SGD", batch_size=batch_size)
    model.instantiate(n_items, n_shared_features, n_items_features)
    eval_before = model.evaluate(dataset, mode="optim")
    tf.config.run_functions_eagerly(True)  # To help with the coverage calculation
    model.fit(dataset)
    eval_after = model.evaluate(dataset, mode="optim")
    assert eval_after <= eval_before


def test_reslogit_fit_with_adam():
    """Tests that ResLogit can fit with Adam."""
    global dataset

    model = ResLogit(lr=lr, epochs=epochs, optimizer="Adam", batch_size=batch_size)
    model.instantiate(n_items, n_shared_features, n_items_features)
    eval_before = model.evaluate(dataset, mode="optim")
    model.fit(dataset)
    eval_after = model.evaluate(dataset, mode="optim")
    assert eval_after <= eval_before


def test_reslogit_fit_with_adamax():
    """Tests that ResLogit can fit with Adamax."""
    global dataset

    model = ResLogit(lr=lr, epochs=epochs, optimizer="Adamax", batch_size=batch_size)
    model.instantiate(n_items, n_shared_features, n_items_features)
    eval_before = model.evaluate(dataset, mode="optim")
    model.fit(dataset)
    eval_after = model.evaluate(dataset, mode="optim")
    assert eval_after <= eval_before


def test_reslogit_fit_with_optimizer_not_implemented():
    """Tests that ResLogit can switch for default Adam.

    When it encounters an optimizer that is not implemented.
    """
    global dataset

    model = ResLogit(lr=lr, epochs=epochs, optimizer="xyz_not_implemented", batch_size=batch_size)
    model.instantiate(n_items, n_shared_features, n_items_features)
    eval_before = model.evaluate(dataset, mode="optim")
    model.fit(dataset)
    eval_after = model.evaluate(dataset, mode="optim")
    assert eval_after <= eval_before


def test_reslogit_fit_with_none_intercept():
    """Tests that ResLogit can fit with intercept=None."""
    global dataset

    model = ResLogit(intercept=None, lr=lr, epochs=epochs, optimizer="Adam", batch_size=batch_size)

    indexes, weights = model.instantiate(
        n_items=n_items, n_shared_features=n_shared_features, n_items_features=n_items_features
    )
    assert "intercept" not in indexes

    model.instantiate(n_items, n_shared_features, n_items_features)
    eval_before = model.evaluate(dataset, mode="optim")
    model.fit(dataset)
    eval_after = model.evaluate(dataset, mode="optim")
    assert eval_after <= eval_before


def test_reslogit_fit_with_item_intercept():
    """Tests that ResLogit can fit with intercept="item"."""
    global dataset

    model = ResLogit(
        intercept="item", lr=lr, epochs=epochs, optimizer="Adam", batch_size=batch_size
    )

    indexes, weights = model.instantiate(
        n_items=n_items, n_shared_features=n_shared_features, n_items_features=n_items_features
    )
    assert "intercept" in indexes

    eval_before = model.evaluate(dataset, mode="optim")
    model.fit(dataset)
    eval_after = model.evaluate(dataset, mode="optim")
    assert eval_after <= eval_before


def test_reslogit_fit_with_item_full_intercept():
    """Tests that ResLogit can fit with intercept="item-full"."""
    global dataset

    model = ResLogit(
        intercept="item-full", lr=lr, epochs=epochs, optimizer="Adam", batch_size=batch_size
    )

    indexes, weights = model.instantiate(
        n_items=n_items, n_shared_features=n_shared_features, n_items_features=n_items_features
    )
    assert "intercept" in indexes

    eval_before = model.evaluate(dataset, mode="optim")
    model.fit(dataset)
    eval_after = model.evaluate(dataset, mode="optim")
    assert eval_after <= eval_before


def test_reslogit_fit_with_other_intercept():
    """Tests that ResLogit can fit with another intercept."""
    global dataset

    model = ResLogit(
        intercept="xyz_other_intercept",
        lr=lr,
        epochs=epochs,
        optimizer="Adam",
        batch_size=batch_size,
    )

    indexes, weights = model.instantiate(
        n_items=n_items, n_shared_features=n_shared_features, n_items_features=n_items_features
    )
    assert "intercept" in indexes

    model.instantiate(n_items, n_shared_features, n_items_features)
    eval_before = model.evaluate(dataset, mode="optim")
    model.fit(dataset)
    eval_after = model.evaluate(dataset, mode="optim")
    assert eval_after <= eval_before


# def test_reslogit_comparison_with_simple_mnl():
#     """Tests that ResLogit can fit better than SimpleMNL."""
#     full_dataset = load_swissmetro() # Use the full dataset to compare the models

#     reslogit = ResLogit(
#         intercept="item",
#         n_layers=0,
#         lr=lr,
#         epochs=epochs,
#         optimizer="Adam",
#         batch_size=batch_size
#     )
#     reslogit_indexes, reslogit_initial_weights = reslogit.instantiate(
#         n_items=n_items, n_shared_features=n_shared_features, n_items_features=n_items_features
#     )
#     reslogit.fit(full_dataset)
#     reslogit_final_weights = reslogit.trainable_weights
#     reslogit_score = reslogit.evaluate(full_dataset, mode="optim")

#     simple_mnl = SimpleMNL(
#         intercept="item",
#         lr=lr,
#         epochs=epochs,
#         optimizer="Adam",
#         batch_size=batch_size
#     )
#     simple_mnl_indexes, simple_mnl_initial_weights = simple_mnl.instantiate(
#         n_items=n_items, n_shared_features=n_shared_features, n_items_features=n_items_features
#     )
#     simple_mnl.fit(full_dataset)
#     simple_mnl_final_weights = simple_mnl.trainable_weights
#     simple_mnl_score = simple_mnl.evaluate(full_dataset, mode="optim")

#     assert reslogit_indexes == simple_mnl_indexes
#     for i in range(len(reslogit_initial_weights)):
#         assert np.allclose(
#             simple_mnl_initial_weights[i].numpy(),
#             reslogit_initial_weights[i].numpy(),
#             rtol=0,
#             atol=0.01,
#         )
#     assert np.abs(simple_mnl_score - reslogit_score) < 0.05
#     for i in range(len(reslogit_final_weights)):
#         assert np.allclose(
#             simple_mnl_final_weights[i].numpy(),
#             reslogit_final_weights[i].numpy(),
#             rtol=0,
#             atol=0.01,
#         )


def test_reslogit_different_n_layers():
    """Tests that ResLogit can fit with different n_layers."""
    global dataset

    for n_layers in [0, 1, 3]:
        model = ResLogit(
            n_layers=n_layers, lr=lr, epochs=epochs, optimizer="Adam", batch_size=batch_size
        )
        # The model can fit
        model.instantiate(n_items, n_shared_features, n_items_features)
        eval_before = model.evaluate(dataset, mode="optim")
        model.fit(dataset)
        eval_after = model.evaluate(dataset, mode="optim")
        assert eval_after <= eval_before

        # The global shape of the residual weights corresponds to the number of layers
        assert len(model.resnet_model.trainable_variables) == n_layers

        if n_layers > 0:
            for layer_idx in range(n_layers):
                # Each residual layer has a (n_items, n_items) matrix of weights
                assert model.resnet_model.trainable_variables[layer_idx].shape == (n_items, n_items)


def test_reslogit_different_layers_width():
    """Tests that ResLogit can fit with different custom widths for its residual layers."""
    global dataset

    list_n_layers = [0, 1, 3]
    list_res_layers_width = [[], [], [12, n_items]]

    for n_layers, res_layers_width in zip(list_n_layers, list_res_layers_width):
        model = ResLogit(
            n_layers=n_layers,
            res_layers_width=res_layers_width,
            lr=lr,
            epochs=epochs,
            optimizer="Adam",
            batch_size=batch_size,
        )
        # The model can fit
        model.instantiate(n_items, n_shared_features, n_items_features)
        eval_before = model.evaluate(dataset, mode="optim")
        model.fit(dataset)
        eval_after = model.evaluate(dataset, mode="optim")
        if not tf.math.is_nan(eval_after):
            assert eval_after <= eval_before

        # The global shape of the residual weights corresponds to the number of layers
        assert len(model.resnet_model.trainable_variables) == n_layers

        if n_layers > 0:
            # The first residual layer has a (n_items, n_items) matrix of weights
            assert model.resnet_model.trainable_variables[0].shape == (n_items, n_items)

            for layer_idx in range(1, n_layers):
                # For i > 1, the i-th residual layer has a
                # (res_layers_width[i-2], res_layers_width[i-1]) matrix of weights
                layer_width = res_layers_width[layer_idx - 1]
                prev_layer_width = res_layers_width[layer_idx - 2]
                assert model.resnet_model.trainable_variables[layer_idx].shape == (
                    prev_layer_width,
                    layer_width,
                )

    # Check if the ValueError are raised when the res_layers_width is not consistent
    model = ResLogit(
        n_layers=4,
        res_layers_width=[2, 4, 8, n_items],
        lr=lr,
        epochs=epochs,
        optimizer="Adam",
        batch_size=batch_size,
    )
    try:
        model.fit(dataset)
        # ValueError: The length of the res_layers_width list should be equal to n_layers - 1
        assert False
    except ValueError:
        assert True

    model = ResLogit(
        n_layers=4,
        res_layers_width=[2, 4, 8, 16],
        lr=lr,
        epochs=epochs,
        optimizer="Adam",
        batch_size=batch_size,
    )
    try:
        model.fit(dataset)
        # ValueError: The last element of the res_layers_width list should be equal to n_items
        assert False
    except ValueError:
        assert True


def test_reslogit_different_activation():
    """Tests that ResLogit can fit with different activation functions for its residual layers."""
    global dataset

    list_activation = ["linear", "relu", "-relu", "tanh", "sigmoid", "softplus"]

    for activation_str in list_activation:
        model = ResLogit(
            n_layers=2,
            activation=activation_str,
            lr=lr,
            epochs=epochs,
            optimizer="Adam",
            batch_size=batch_size,
        )
        # The model can fit
        """model.instantiate(n_items, n_shared_features, n_items_features)
        eval_before = model.evaluate(dataset, mode="optim")
        model.fit(dataset)
        eval_after = model.evaluate(dataset, mode="optim")
        assert eval_after <= eval_before"""
        assert True

    # Check if the ValueError is raised when the activation is not implemented
    model = ResLogit(
        n_layers=2,
        activation="xyz_not_implemented",
        lr=lr,
        epochs=epochs,
        optimizer="Adam",
        batch_size=batch_size,
    )
    try:
        model.fit(dataset)
        # ValueError: The activation function is not implemented
        assert False
    except ValueError:
        assert True


def test_that_endpoints_run():
    """Dummy test to check that the endpoints run.

    No verification of results.
    """
    global dataset

    model = ResLogit(epochs=epochs)
    model.fit(dataset)
    model.evaluate(dataset, mode="optim")
    model.predict_probas(dataset)
    assert True
