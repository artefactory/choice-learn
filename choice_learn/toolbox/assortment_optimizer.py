"""Tool function for assortment optimization."""

import gurobipy as gp


class AssortmentOptimizer(object):
    """Base class for assortment optimization."""

    def __init__(self, utilities, assortment_size):
        """Initializes the AssortmentOptimizer object.

        Parameters
        ----------
        utilities : list-like
            List of utilities for each item.
        assortment_size : int
            maximum size of the requested assortment.
        """
        self.utilities = utilities
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
        solver = gp.Model("ClusterUTA")
        self.binary_z = {
            i: solver.addVar(vtype=gp.GRB.BINARY, name=f"z_{i}") for i in range(self.n_items)
        }
        solver.addConstr(
            gp.quicksum(self.binary_z[i] for i in range(self.n_items)) == self.assortment_size
        )
        return solver

    def set_objective_function(self):
        """Function to define the objective function to maximize with the assortment."""
        pass

    def add_constraint(self):
        """Function to add constraints."""
        pass

    def solve(self):
        """Function to solve the optimization problem."""
        self.solver.update()

        # -- Résolution --
        self.solver.optimize()
        self.status = self.solver.Status
        assortment = []
        for i in range(self.n_items):
            assortment.append(self.binary_z[i].x)
        return assortment
