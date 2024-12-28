"""Testing base ChoiceModel."""

import numpy as np

from choice_learn.toolbox.assortment_optimizer import MNLAssortmentOptimizer, LatentClassAssortmentOptimizer, LatentClassPricingOptimizer

solvers = ["or-tools"]

def test_mnl_assort_instantiate():
    """Test instantiation with both solvers."""
    for solv in solvers:
        MNLAssortmentOptimizer(
        solver=solv,
        utilities=np.array([1., 2., 3.]),
        itemwise_values=np.array([0.5, 0.5, 0.5]),
        assortment_size=2)


def test_lc_assort_instantiate():
    """Test instantiation with both solvers."""
    for solv in solvers:
        LatentClassAssortmentOptimizer(
    solver=solv,
    class_weights=np.array([.2, .8]),
    class_utilities=np.array([[1., 2., 3.], [3., 2., 1.]]),
    itemwise_values=np.array([0.5, 0.5, 0.5]),
    assortment_size=2)


def test_lc_pricing_instantiate():
    """Test instantiation with both solvers."""
    for solv in solvers:
        LatentClassPricingOptimizer(
    solver=solv,
    class_weights=np.array([.2, .8]),
    class_utilities=np.array([[[1., 1.1], [2., 2.1], [3., 3.1]],
    [[3., 3.1], [2., 2.1], [1., 1.1]]]),
    itemwise_values=np.array([[0.5, 1.2], [0.5, 1.2], [0.5, 1.2]]),
    assortment_size=2)