"""Tool function for assortment optimization."""
import logging

import gurobipy as gp
import numpy as np

"""TODO: clarify outside good integration
TODO 2: ADD easy integration of additionnal constraints
"""


class AssortmentOptimizer(object):
    """Base class for assortment optimization."""

    def __init__(self, utilities, itemwise_values, assortment_size, outside_option_given=False):
        """Initializes the AssortmentOptimizer object.

        Parameters:
        -----------
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

        return solver

    def set_base_constraints(self):
        """Functions to set LP base constraints.

        In particular, ensures Charnes-Cooper transformation constraints
        and assortment size constraint.
        """
        # Base Charnes-Cooper Constraints for Integers
        for j in range(1, self.n_items + 1):
            self.solver.addConstr(self.y[j] <= self.y[0])

        # Base Charnes-Cooper Constraint for Normalization
        charnes_cooper = gp.quicksum(self.y[j] for j in range(self.n_items + 1))
        self.solver.addConstr(charnes_cooper == 1)

        # Assortment size constraint
        if self.assortment_size is not None:
            self.solver.addConstr(
                gp.quicksum([self.y[j] for j in range(1, self.n_items)])
                <= self.assortment_size * self.y[0]
            )
            self.solver.addConstr(
                gp.quicksum([-self.y[j] for j in range(1, self.n_items)])
                <= -self.assortment_size * self.y[0]
            )

        # Integrate constraints
        self.solver.update()

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
        for i in range(0, self.n_items + 1):
            if self.y[i].x > 0:
                assortment[i] = 1

        chosen_utilities = assortment * self.utilities
        norm = np.sum(chosen_utilities)

        recomputed_obj = np.sum(chosen_utilities * self.itemwise_values / norm)

        if not self.outside_option_given:
            assortment = assortment[1:]
        return assortment, recomputed_obj


class LatentClassAssortmentOptimizer(object):
    """Assortment optimizer for latent class models.

    Implementation of the paper:
    "A branch-and-cuta algorithm for the latent-class logit assortment problem"
    by Isabel Mendez-Diaz, Juan José Miranda-Bront, Gustavo Vulcano and Paula Zabala.
    2012.
    """

    def __init__(
        self,
        class_weights,
        class_utilities,
        itemwise_values,
        assortment_size,
        outside_option_given=False,
    ):
        """Initializes the AssortmentOptimizer object.

        Parameters:
        -----------
        utilities : Iterable
            List of utilities for each item.
        itemwise_values: Iterable
            List of to-be-optimized values for each item, e.g. prices.
        assortment_size : int
            maximum size of the requested assortment.
        outside_option_given : bool
            Whether the outside option is given or not (and thus is automatically added).
        """
        if len(class_utilities[0]) != len(itemwise_values):
            raise ValueError(
                f"You should provide as many utilities as itemwise values.\
                             Found {len(class_utilities[0])} and {len(itemwise_values)} instead."
            )
        if len(class_weights) != len(class_utilities):
            raise ValueError("Number of classes and class weights should be the same.")

        self.outside_option_given = outside_option_given
        if not self.outside_option_given:
            self.class_utilities = np.stack(
                [
                    np.concatenate([[np.exp(0.0)], class_utilities[i]], axis=0)
                    for i in range(len(class_weights))
                ]
            )
            self.itemwise_values = np.concatenate([[0.0], itemwise_values], axis=0)
        self.n_items = len(self.utilities) - 1
        self.assortment_size = assortment_size
        self.class_weights = class_weights
        if self.assortment_size > self.n_items:
            logging.warning(
                """Assortment size is greater than the number of items.
                Setting it to the number of items."""
            )

        self.solver = self.base_instantiate()
        self.set_base_constraints()

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

        # Create yi variables
        y = {}

        for i in range(self.n_items + 1):
            y[i] = solver.addVar(vtype=gp.GRB.BINARY, name="y_%s" % i)
        self.y = y

        # Create zli variables
        z = {}
        for i in range(self.n_items + 1):
            for class_index in range(len(self.class_weights)):
                z[(class_index, i)] = solver.addVar(
                    vtype=gp.GRB.CONTINUOUS, name=f"z_{class_index}_{i}", lb=0
                )
        self.z = z

        # Create xl variables
        x = {}
        for class_index in range(len(self.class_weights)):
            x[class_index] = solver.addVar(vtype=gp.GRB.CONTINUOUS, name=f"x_{class_index}", lb=0)
        self.x = x
        # Integrate new variables
        solver.update()

        return solver

    def set_base_constraints(self):
        """Functions to set LP base constraints.

        In particular, ensures Charnes-Cooper transformation constraints
        and assortment size constraint.
        """
        # Base Charnes-Cooper Constraint for Normalization
        for class_index in range(len(self.class_weights)):
            charnes_cooper = gp.quicksum(
                self.class_utilities[class_index][i] * self.z[(class_index, i)]
                for i in range(1, self.n_items + 1)
            )
            charnes_cooper = (
                charnes_cooper + self.x[class_index] * self.class_utilities[class_index][0]
            )
            self.solver.addConstr(charnes_cooper == 1)

        # Charnes-Cooper for variables constraints
        for class_index in range(len(self.class_weights)):
            for i in range(1, self.n_items + 1):
                self.solver.addConstr(self.z[(class_index, i)] <= self.x[class_index])
                self.solver.addConstr(
                    self.x[class_index] * self.class_utilities[class_index][0]
                    - self.z[(class_index, i)] * self.class_utilities[class_index][0]
                    <= 1 - self.y[i]
                )
                self.solver.addConstr(
                    self.z[(class_index, i)]
                    * (self.class_utilities[class_index][0] + self.class_utilities[class_index][i])
                    <= self.y[i]
                )

        # Assortment size constraint
        if self.assortment_size is not None:
            self.solver.addConstr(
                gp.quicksum([self.y[j] for j in range(1, self.n_items)]) <= self.assortment_size
            )

        # Integrate constraints
        self.solver.update()

    def base_set_objective_function(self):
        """Base function to set optimization objective."""
        objective = 0
        for class_index in range(len(self.class_weights)):
            for j in range(self.n_items):
                objective += (
                    self.class_weights[class_index]
                    * self.itemwise_values[j]
                    * self.class_utilities[class_index][j]
                    * self.z[(class_index, j)]
                )
        self.solver.setObjective(objective, gp.GRB.MAXIMIZE)

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
        self.base_set_objective_function()
        self.solver.update()

        # -- Optimize --
        self.solver.optimize()
        self.status = self.solver.Status

        assortment = np.zeros(self.n_items + 1)
        for i in range(0, self.n_items + 1):
            if self.y[i].x > 0:
                assortment[i] = 1

        chosen_utilities = assortment * self.utilities
        norm = np.sum(chosen_utilities)

        recomputed_obj = np.sum(chosen_utilities * self.itemwise_values / norm)

        if not self.outside_option_given:
            assortment = assortment[1:]
        return assortment, recomputed_obj
