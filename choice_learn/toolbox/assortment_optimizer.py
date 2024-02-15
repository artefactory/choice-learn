"""Tool function for assortment optimization."""
import gurobipy as gp
import numpy as np

"""TODO: clarify outside good integration
TODO 2: ADD easy integration of additionnal constraints
"""


class AssortmentOptimizer(object):
    """Base class for assortment optimization."""

    def __init__(self, utilities, itemwise_values, assortment_size):
        """Initializes the AssortmentOptimizer object.

        Parameters
        ----------
        utilities : Iterable
            List of utilities for each item.
        itemwise_values: Iterable
            List of to-be-optimized values for each item, e.g. prices.
        assortment_size : int
            maximum size of the requested assortment.
        """
        if len(utilities) != len(itemwise_values):
            raise ValueError(
                f"You should provide as many utilities as itemwise values.\
                             Found {len(utilities)} and {len(itemwise_values)} instead."
            )
        self.utilities = utilities
        self.itemwise_values = itemwise_values
        self.assortment_size = assortment_size

        self.n_items = len(utilities)

        self.solver = self.base_instantiate()

    def base_instantiate(self):
        """Base instantiation of the solver.

        Returns:
        --------
        gurobipy.Model
            solver with basic variables and constraints.
        """
        # Create a new model
        solver = gp.Model("Assortment_IP")
        solver.ModelSense = -1
        solver.setParam("OutputFlag", False)

        # Create variables
        y = {}

        for j in range(self.n_items + 1):
            y[j] = solver.addVar(
                vtype=gp.GRB.CONTINUOUS, obj=self.itemwise_values[j], name="y_%s" % j
            )
        self.y = y
        # Integrate new variables
        solver.update()

        # Add constraints
        for j in range(1, self.n_items + 1):
            solver.addConstr(y[j] * self.utilities[0] <= y[0] * self.utilities[j])

        charnes_cooper = gp.quicksum(y[j] for j in range(self.n_items + 1))
        solver.addConstr(charnes_cooper == 1)
        assort_size = gp.quicksum(y[j] for j in range(1, self.n_items + 1))
        solver.addConstr(assort_size == self.assortment_size * y[0])

        # Integrate constraints
        solver.update()
        return solver

    def set_objective_function(self, itemwise_values):
        """Function to define the objective function to maximize with the assortment.

        Parameters:
        -----------
        itemwise_values : list-like
            List of values for each item - total value to be optimized.
        """
        raise NotImplementedError

    def add_constraint(self):
        """Function to add constraints."""
        raise NotImplementedError

    def solve(self):
        """Function to solve the optimization problem.

        Returns:
        --------
        np.ndarray:
            Array of 0s and 1s, indicating the presence of each item in the optimal assortment.
        """
        self.solver.update()

        # -- Optimize --
        self.solver.optimize()
        self.status = self.solver.Status

        assortment = np.zeros(self.n_items + 1)
        for i in range(self.n_items + 1):
            if self.y[i].x > 0:
                assortment[i] = 1

        return assortment
