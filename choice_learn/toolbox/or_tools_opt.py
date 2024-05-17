"""Tool functions for optimization with OR-Tools."""

import numpy as np
from ortools.linear_solver import pywraplp


class ORToolsAssortmentOptimizer(object):
    """Base class for assortment optimization."""

    def __init__(self, utilities, itemwise_values, assortment_size, outside_option_given=False):
        """Initialize the AssortmentOptimizer object.

        Parameters
        ----------
        utilities : Iterable
            List of utilities for each item.
        itemwise_values: Iterable
            List of to-be-optimized values for each item, e.g. prices.
        assortment_size : int
            maximum size of the requested assortment.
        outside_option_given : bool
            Whether the outside option is given or not (and thus is automatically added).
        """
        if len(utilities) != len(itemwise_values):
            raise ValueError(
                f"You should provide as many utilities as itemwise values.\
                             Found {len(utilities)} and {len(itemwise_values)} instead."
            )
        self.outside_option_given = outside_option_given
        if not self.outside_option_given:
            self.utilities = np.concatenate([[np.exp(0.0)], utilities], axis=0)
            self.itemwise_values = np.concatenate([[0.0], itemwise_values], axis=0)
        self.n_items = len(self.utilities) - 1
        self.assortment_size = assortment_size

        self.solver = self.base_instantiate()
        self.set_base_constraints()

    def base_instantiate(self):
        """Instantiate of the solver.

        Returns
        -------
        gurobipy.Model
            solver with basic variables and constraints.
        """
        # Create a new model
        solver = pywraplp.Solver.CreateSolver("GLOP")
        infinity = solver.infinity()

        # Create variables
        y = {}

        for j in range(self.n_items + 1):
            y[j] = solver.NumVar(
                lb=0,
                ub=infinity,
                name=f"y_{j}",
            )
        self.y = y
        # Integrate new variables

        return solver

    def set_base_constraints(self):
        """Functions to set LP base constraints.

        In particular, ensures Charnes-Cooper transformation constraints
        and assortment size constraint.
        """
        # Base Charnes-Cooper Constraints for Integers
        for j in range(1, self.n_items + 1):
            self.solver.Add(self.y[j] <= self.y[0])

        # Base Charnes-Cooper Constraint for Normalization
        charnes_cooper = sum([self.y[j] * self.utilities[j] for j in range(self.n_items + 1)])
        self.solver.Add(charnes_cooper == 1)

        # Assortment size constraint
        if self.assortment_size is not None:
            self.solver.Add(
                sum([self.y[j] for j in range(1, self.n_items + 1)])
                <= self.assortment_size * self.y[0]
            )
            self.solver.Add(
                sum([-self.y[j] for j in range(1, self.n_items + 1)])
                <= -self.assortment_size * self.y[0]
            )

    def set_objective_function(self, itemwise_values):
        """Define the objective function to maximize with the assortment.

        Parameters
        ----------
        itemwise_values : list-like
            List of values for each item - total value to be optimized.
        """
        raise NotImplementedError

    def add_constraint(self):
        """Aadd constraints."""
        raise NotImplementedError

    def solve(self):
        """Solve the optimization problem.

        Returns
        -------
        np.ndarray:
            Array of 0s and 1s, indicating the presence of each item in the optimal assortment.
        """
        objective = sum(
            [self.y[j] * self.itemwise_values[j] * self.utilities[j] for j in range(len(self.y))]
        )

        # -- Optimize --
        self.solver.Maximize(objective)
        status = self.solver.Solve()
        self.status = status

        assortment = np.zeros(self.n_items + 1)
        for i in range(0, self.n_items + 1):
            if self.y[i].solution_value() > 0:
                assortment[i] = 1
        chosen_utilities = assortment * self.utilities
        norm = np.sum(chosen_utilities)

        recomputed_obj = np.sum(chosen_utilities * self.itemwise_values / norm)

        if not self.outside_option_given:
            assortment = assortment[1:]
        return assortment, recomputed_obj
