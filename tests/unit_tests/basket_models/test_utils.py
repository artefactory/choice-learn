"""Unit tests for basket utils."""

from choice_learn.basket_models.utils.permutation import permutations


def test_quick_permutation() -> None:
    """Quickly tests one line of permutation."""
    items = [1, 2, 3]
    for k in permutations(items, r=5):
        assert k is None
