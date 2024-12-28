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
        MNLAssortmentOptimizer(
            solver=solv,
            utilities=np.array([1.0, 2.0, 3.0]),
            itemwise_values=np.array([0.5, 0.5, 0.5]),
            assortment_size=2,
        )


def test_lc_assort_instantiate():
    """Test instantiation with both solvers."""
    for solv in solvers:
        LatentClassAssortmentOptimizer(
            solver=solv,
            class_weights=np.array([0.2, 0.8]),
            class_utilities=np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]),
            itemwise_values=np.array([0.5, 0.5, 0.5]),
            assortment_size=2,
        )


def test_lc_pricing_instantiate():
    """Test instantiation with both solvers."""
    for solv in solvers:
        LatentClassPricingOptimizer(
            solver=solv,
            class_weights=np.array([0.2, 0.8]),
            class_utilities=np.array(
                [[[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]], [[3.0, 3.1], [2.0, 2.1], [1.0, 1.1]]]
            ),
            itemwise_values=np.array([[0.5, 1.2], [0.5, 1.2], [0.5, 1.2]]),
            assortment_size=2,
        )

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
