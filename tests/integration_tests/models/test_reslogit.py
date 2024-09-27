"""Tests ResLogit."""

import numpy as np

from choice_learn.datasets import load_swissmetro
from choice_learn.models import ResLogit, SimpleMNL

dataset = load_swissmetro()
n_items = np.shape(dataset.items_features_by_choice)[2]
n_shared_features = np.shape(dataset.shared_features_by_choice)[2]
n_items_features = np.shape(dataset.items_features_by_choice)[3]


def test_reslogit_fit_with_sgd():
    """Tests that ResLogit can fit with SGD."""
    global dataset

    model = ResLogit(lr=1e-6, epochs=30, optimizer="SGD", batch_size=32)
    model.fit(dataset)
    model.evaluate(dataset)
    assert model.evaluate(dataset) < 1.0


def test_reslogit_fit_with_adam():
    """Tests that ResLogit can fit with Adam."""
    global dataset

    model = ResLogit(epochs=20, optimizer="Adam", batch_size=32)
    model.fit(dataset)
    model.evaluate(dataset)
    assert model.evaluate(dataset) < 1.0


def test_reslogit_fit_with_adamax():
    """Tests that ResLogit can fit with Adamax."""
    global dataset

    model = ResLogit(epochs=20, optimizer="Adamax", batch_size=32)
    model.fit(dataset)
    model.evaluate(dataset)
    assert model.evaluate(dataset) < 1.0


def test_reslogit_fit_with_optimizer_not_implemented():
    """Tests that ResLogit can switch for default Adam.

    When it encounters an optimizer that is not implemented.
    """
    global dataset

    model = ResLogit(epochs=20, optimizer="xyz_not_implemented", batch_size=32)
    model.fit(dataset)
    model.evaluate(dataset)
    assert model.evaluate(dataset) < 1.0


def test_reslogit_fit_with_none_intercept():
    """Tests that ResLogit can fit with intercept=None."""
    global dataset

    model = ResLogit(intercept=None, lr=1e-6, epochs=20, optimizer="SGD", batch_size=32)

    indexes, weights = model.instantiate(
        n_items=n_items, n_shared_features=n_shared_features, n_items_features=n_items_features
    )
    assert "intercept" not in indexes

    model.fit(dataset)
    model.evaluate(dataset)
    assert model.evaluate(dataset) < 1.0


def test_reslogit_fit_with_item_intercept():
    """Tests that ResLogit can fit with intercept="item"."""
    global dataset

    model = ResLogit(intercept="item", lr=1e-6, epochs=20, optimizer="SGD", batch_size=32)

    indexes, weights = model.instantiate(
        n_items=n_items, n_shared_features=n_shared_features, n_items_features=n_items_features
    )
    assert "intercept" in indexes

    model.fit(dataset)
    model.evaluate(dataset)
    assert model.evaluate(dataset) < 1.0


def test_reslogit_fit_with_item_full_intercept():
    """Tests that ResLogit can fit with intercept="item-full"."""
    global dataset

    model = ResLogit(intercept="item-full", lr=1e-6, epochs=20, optimizer="SGD", batch_size=32)

    indexes, weights = model.instantiate(
        n_items=n_items, n_shared_features=n_shared_features, n_items_features=n_items_features
    )
    assert "intercept" in indexes

    model.fit(dataset)
    model.evaluate(dataset)
    assert model.evaluate(dataset) < 1.0


def test_reslogit_fit_with_other_intercept():
    """Tests that ResLogit can fit with another intercept."""
    global dataset

    model = ResLogit(
        intercept="xyz_other_intercept", lr=1e-6, epochs=20, optimizer="SGD", batch_size=32
    )

    indexes, weights = model.instantiate(
        n_items=n_items, n_shared_features=n_shared_features, n_items_features=n_items_features
    )
    assert "intercept" in indexes

    model.fit(dataset)
    model.evaluate(dataset)
    assert model.evaluate(dataset) < 1.0


def test_reslogit_comparison_with_simple_mnl():
    """Tests that ResLogit can fit better than SimpleMNL."""
    global dataset

    reslogit = ResLogit(
        intercept="item", lr=1e-6, n_layers=0, epochs=100, optimizer="SGD", batch_size=32
    )
    reslogit_indexes, reslogit_initial_weights = reslogit.instantiate(
        n_items=n_items, n_shared_features=n_shared_features, n_items_features=n_items_features
    )
    reslogit.fit(dataset)
    reslogit_final_weights = reslogit.trainable_weights
    reslogit_score = reslogit.evaluate(dataset)

    simple_mnl = SimpleMNL(intercept="item", lr=1e-6, epochs=100, optimizer="SGD", batch_size=32)
    simple_mnl_indexes, simple_mnl_initial_weights = simple_mnl.instantiate(
        n_items=n_items, n_shared_features=n_shared_features, n_items_features=n_items_features
    )
    simple_mnl.fit(dataset)
    simple_mnl_final_weights = simple_mnl.trainable_weights
    simple_mnl_score = simple_mnl.evaluate(dataset)

    assert reslogit_indexes == simple_mnl_indexes
    for i in range(len(reslogit_initial_weights)):
        assert np.allclose(
            simple_mnl_initial_weights[i].numpy(),
            reslogit_initial_weights[i].numpy(),
            rtol=0,
            atol=0.01,
        )
    assert np.abs(simple_mnl_score - reslogit_score) < 0.05
    for i in range(len(reslogit_final_weights)):
        assert np.allclose(
            simple_mnl_final_weights[i].numpy(),
            reslogit_final_weights[i].numpy(),
            rtol=0,
            atol=0.01,
        )


def test_that_endpoints_run():
    """Dummy test to check that the endpoints run.

    No verification of results.
    """
    global dataset

    model = ResLogit(epochs=20)
    model.fit(dataset)
    model.evaluate(dataset)
    model.predict_probas(dataset)
    assert True
