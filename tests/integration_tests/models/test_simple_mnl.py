"""Tests SimpleMNL."""

from choice_learn.datasets import load_swissmetro
from choice_learn.models import SimpleMNL

dataset = load_swissmetro()


def test_simple_mnl_lbfgs_fit_with_lbfgs():
    """Tests that SimpleMNL can fit with LBFGS."""
    global dataset

    model = SimpleMNL(epochs=20)
    model.fit(dataset)
    model.evaluate(dataset)
    assert model.evaluate(dataset) < 1.0


def test_simple_mnl_lbfgs_fit_with_adam():
    """Tests that SimpleMNL can fit with Adam."""
    global dataset

    model = SimpleMNL(epochs=20, optimizer="adam", batch_size=256)
    model.fit(dataset)
    model.evaluate(dataset)
    assert model.evaluate(dataset) < 1.0


def test_that_endpoints_run():
    """Dummy test to check that the endpoints run.

    No verification of results.
    """
    global dataset

    model = SimpleMNL(epochs=20)
    model.fit(dataset)
    model.evaluate(dataset)
    model.predict_probas(dataset)
    assert True
