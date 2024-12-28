"""Testing base ChoiceModel."""

import numpy as np
import pytest

from choice_learn.toolbox.assortment_optimizer import (
    LatentClassAssortmentOptimizer,
    LatentClassPricingOptimizer,
    MNLAssortmentOptimizer,
)

solvers = ["or-tools"]


def test_mnl_assort_instantiate():
    """Test instantiation with both solvers."""
    for solv in solvers:
        opt = MNLAssortmentOptimizer(
            solver=solv,
            utilities=np.array([1.0, 2.0, 3.0]),
            itemwise_values=np.array([0.5, 0.5, 0.5]),
            assortment_size=2,
        )
        opt.solve()


def test_lc_assort_instantiate():
    """Test instantiation with both solvers."""
    for solv in solvers:
        opt = LatentClassAssortmentOptimizer(
            solver=solv,
            class_weights=np.array([0.2, 0.8]),
            class_utilities=np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]),
            itemwise_values=np.array([0.5, 0.5, 0.5]),
            assortment_size=2,
        )
        opt.solve()


def test_lc_pricing_instantiate():
    """Test instantiation with both solvers."""
    for solv in solvers:
        opt = LatentClassPricingOptimizer(
            solver=solv,
            class_weights=np.array([0.2, 0.8]),
            class_utilities=np.array(
                [[[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]], [[3.0, 3.1], [2.0, 2.1], [1.0, 1.1]]]
            ),
            itemwise_values=np.array([[0.5, 1.2], [0.5, 1.2], [0.5, 1.2]]),
            assortment_size=2,
        )
        opt.solve()


def test_wrong_solver():
    """Test error raised when specifying wrong solver."""
    solver = "rotools"
    with pytest.raises(ValueError):
        MNLAssortmentOptimizer(
            solver=solver,
            utilities=np.array([1.0, 2.0, 3.0]),
            itemwise_values=np.array([0.5, 0.5, 0.5]),
            assortment_size=2,
        )
    with pytest.raises(ValueError):
        LatentClassAssortmentOptimizer(
            solver=solver,
            class_weights=np.array([0.2, 0.8]),
            class_utilities=np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]),
            itemwise_values=np.array([0.5, 0.5, 0.5]),
            assortment_size=2,
        )
    with pytest.raises(ValueError):
        LatentClassPricingOptimizer(
            solver=solver,
            class_weights=np.array([0.2, 0.8]),
            class_utilities=np.array(
                [[[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]], [[3.0, 3.1], [2.0, 2.1], [1.0, 1.1]]]
            ),
            itemwise_values=np.array([[0.5, 1.2], [0.5, 1.2], [0.5, 1.2]]),
            assortment_size=2,
        )


def test_raised_errors():
    """Test diverse parametrization that should raise errors."""
    with pytest.raises(ValueError):
        MNLAssortmentOptimizer(
            solver="ortools",
            utilities=np.array([1.0, 2.0, 3.0, 4.0]),
            itemwise_values=np.array([0.5, 0.5, 0.5]),
            assortment_size=2,
        )
    with pytest.raises(ValueError):
        LatentClassAssortmentOptimizer(
            solver="ortools",
            class_weights=np.array([0.2, 0.8]),
            class_utilities=np.array([[1.0, 2.0], [3.0, 2.0]]),
            itemwise_values=np.array([0.5, 0.5, 0.5]),
            assortment_size=2,
        )

    with pytest.raises(ValueError):
        LatentClassAssortmentOptimizer(
            solver="ortools",
            class_weights=np.array([0.2, 0.8]),
            class_utilities=np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [4.0, 4.0, 4.0]]),
            itemwise_values=np.array([0.5, 0.5, 0.5]),
            assortment_size=2,
        )

    with pytest.raises(ValueError):
        LatentClassPricingOptimizer(
            solver="ortools",
            class_weights=np.array([0.2, 0.8]),
            class_utilities=np.array(
                [[[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]], [[3.0, 3.1], [2.0, 2.1], [1.0, 1.1]]]
            ),
            itemwise_values=np.array([[0.5, 1.2, 2.4], [0.5, 1.2, 2.4], [0.5, 1.2, 2.4]]),
            assortment_size=2,
        )

    with pytest.raises(ValueError):
        LatentClassPricingOptimizer(
            solver="ortools",
            class_weights=np.array([0.2, 0.7, 0.1]),
            class_utilities=np.array(
                [[[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]], [[3.0, 3.1], [2.0, 2.1], [1.0, 1.1]]]
            ),
            itemwise_values=np.array([[0.5, 1.2], [0.5, 1.2], [0.5, 1.2]]),
            assortment_size=2,
        )

    with pytest.raises(ValueError):
        LatentClassPricingOptimizer(
            solver="ortools",
            class_weights=np.array([0.2, 0.8]),
            class_utilities=np.array(
                [[[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]], [[3.0, 3.1], [2.0, 2.1], [1.0, 1.1]]]
            ),
            itemwise_values=np.array([[0.5, 1.2], [0.5, 1.2], [0.5, 1.2], [0.5, 1.2]]),
            assortment_size=2,
        )
