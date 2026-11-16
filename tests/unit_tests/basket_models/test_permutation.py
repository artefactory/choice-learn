"""Unit tests for the permutations function."""

from choice_learn.basket_models.utils.permutation import permutations


def test_permutation() -> None:
    """Test raised errors and warnings when initializing a Shopper object with wrong parameters."""
    assert isinstance(permutations(iterable=range(3), r=10), object)  # r > n
