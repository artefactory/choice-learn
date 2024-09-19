"""Tests for the SimpleMNL model."""

from choice_learn.models import SimpleMNL


def test_simplemnl_instantiation():
    """Tests SimpleMNL instantiation."""
    model = SimpleMNL(intercept="item-full")
    model.instantiate(n_items=4, n_items_features=10, n_shared_features=20)
    assert True


# Test diverse instantiation
# Test .fit
# Test .predict
# Test report
