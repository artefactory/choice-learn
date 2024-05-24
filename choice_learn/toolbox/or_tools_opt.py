"""Tool functions for optimization with OR-Tools."""

import logging

import numpy as np
from ortools.linear_solver import pywraplp


class ORToolsMNLAssortmentOptimizer(object):
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
            self.solver.Add(self.y[j] <= self.y[0], name="Norm")

        # Base Charnes-Cooper Constraint for Normalization
        charnes_cooper = sum([self.y[j] * self.utilities[j] for j in range(self.n_items + 1)])
        self.solver.Add(charnes_cooper == 1, name="Charnes-Cooper")

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


class ORToolsLatentClassAssortmentOptimizer(object):
    """Assortment optimizer for latent class models.

    Implementation of the paper:
    Isabel Méndez-Díaz, Juan José Miranda-Bront, Gustavo Vulcano, Paula Zabala,
    A branch-and-cut algorithm for the latent-class logit assortment problem,
    Discrete Applied Mathematics,
    Volume 164, Part 1,
    2014,
    Pages 246-263,
    ISSN 0166-218X,
    https://doi.org/10.1016/j.dam.2012.03.003.
    """

    def __init__(
        self,
        class_weights,
        class_utilities,
        itemwise_values,
        assortment_size,
        outside_option_given=False,
    ):
        """Initialize the AssortmentOptimizer object.

        Parameters
        ----------
        class_weights: Iterable
            List of weights for each latent class.
        class_utilities: Iterable
            List of utilities for each item of each latent class.
            Must have a shape of (n_classes, n_items)
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
        else:
            self.class_utilities = class_utilities
            self.itemwise_values = itemwise_values
        self.n_items = len(self.itemwise_values) - 1
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
        """Instantiate the solver.

        Returns
        -------
        gurobipy.Model
            solver with basic variables and constraints.
        """
        # Create a new model
        solver = pywraplp.Solver.CreateSolver("SCIP")
        infinity = solver.infinity()

        # Create yi variables
        y = {}

        for i in range(self.n_items + 1):
            y[i] = solver.IntVar(lb=0, ub=2, name="y_%s" % i)
        self.y = y

        # Create zli variables
        z = {}
        for i in range(self.n_items + 1):
            for class_index in range(len(self.class_weights)):
                z[(class_index, i)] = solver.NumVar(
                    lb=0,
                    ub=infinity,
                    name=f"z_{class_index}_{i}",
                )
        self.z = z

        # Create xl variables
        x = {}
        for class_index in range(len(self.class_weights)):
            x[class_index] = solver.NumVar(lb=0, ub=infinity, name=f"x_{class_index}")
        self.x = x

        return solver

    def set_base_constraints(self):
        """Functions to set LP base constraints.

        In particular, ensures Charnes-Cooper transformation constraints
        and assortment size constraint.
        """
        # Base Charnes-Cooper Constraint for Normalization
        for class_index in range(len(self.class_weights)):
            charnes_cooper = sum(
                [
                    self.class_utilities[class_index][i] * self.z[(class_index, i)]
                    for i in range(1, self.n_items + 1)
                ]
            )
            charnes_cooper = (
                charnes_cooper + self.x[class_index] * self.class_utilities[class_index][0]
            )
            self.solver.Add(charnes_cooper == 1, name="Charnes-Cooper")

        # Charnes-Cooper for variables constraints
        for class_index in range(len(self.class_weights)):
            for i in range(1, self.n_items + 1):
                self.solver.Add(
                    self.z[(class_index, i)] - self.x[class_index] <= 0,
                    name=f"Norm class {class_index} item {i}",
                )
                self.solver.Add(
                    self.x[class_index] * self.class_utilities[class_index][0]
                    - self.z[(class_index, i)] * self.class_utilities[class_index][0]
                    + self.y[i]
                    <= 1
                )
                self.solver.Add(
                    self.z[(class_index, i)]
                    * (self.class_utilities[class_index][0] + self.class_utilities[class_index][i])
                    - self.y[i]
                    <= 0
                )

        # Assortment size constraint
        if self.assortment_size is not None:
            self.solver.Add(
                sum([self.y[j] for j in range(1, self.n_items + 1)]) <= self.assortment_size,
                name="assortment size",
            )

    def base_set_objective_function(self):
        """Set optimization objective."""
        objective = []
        for class_index in range(len(self.class_weights)):
            for j in range(self.n_items + 1):
                objective.append(
                    self.class_weights[class_index]
                    * self.itemwise_values[j]
                    * self.class_utilities[class_index][j]
                    * self.z[(class_index, j)]
                )
        self.solver.Maximize(sum(objective))

    def set_objective_function(self, itemwise_values):
        """Define the objective function to maximize with the assortment.

        Parameters
        ----------
        itemwise_values : list-like
            List of values for each item - total value to be optimized.
        """
        raise NotImplementedError

    def add_constraint(self):
        """Add constraints."""
        raise NotImplementedError

    def add_maximal_capacity_constraint(self, itemwise_capacities, maximum_capacity):
        """Add maximal capacity constraint.

        The added constraint is basically sum_{i} y_i * itemwise_capacities[i] <= maximum_capacity

        Parameters
        ----------
        itemwise_capacities : Iterable
            Values of capacity for each item in the assortment.
            Shape must match with itemwise_values.
        maximum_capacity : int/float
            Value of the maximal capacity.
        """
        assortment_capacity = sum(
            [self.y[j] * itemwise_capacities[j - 1] for j in range(1, self.n_items + 1)]
        )
        self.solver.Add(assortment_capacity <= maximum_capacity)

    def add_minimal_capacity_constraint(self, itemwise_capacities, minimum_capacity):
        """Add maximal capacity constraint.

        The added constraint is basically sum_{i} y_i * itemwise_capacities[i] <= maximum_capacity

        Parameters
        ----------
        itemwise_capacities : Iterable
            Values of capacity for each item in the assortment.
            Shape must match with itemwise_values.
        minimum_capacity : int/float
            Value of the maximal capacity.
        """
        assortment_capacity = sum(
            [self.y[j] * itemwise_capacities[j - 1] for j in range(1, self.n_items + 1)]
        )
        self.solver.Add(assortment_capacity >= minimum_capacity)

    def solve(self):
        """Solve the optimization problem.

        Returns
        -------
        np.ndarray:
            Array of 0s and 1s, indicating the presence of each item in the optimal assortment.
        """
        # self.base_set_objective_function()
        objective = []
        for class_index in range(len(self.class_weights)):
            for j in range(self.n_items + 1):
                objective.append(
                    self.class_weights[class_index]
                    * self.itemwise_values[j]
                    * self.class_utilities[class_index][j]
                    * self.z[(class_index, j)]
                )
        self.solver.Maximize(sum(objective))

        # -- Optimize --
        status = self.solver.Solve()
        self.status = status

        assortment = np.zeros(self.n_items + 1)
        for i in range(0, self.n_items + 1):
            if self.y[i].solution_value() > 0:
                assortment[i] = 1

        recomputed_obj = 0
        for class_index in range(len(self.class_weights)):
            chosen_utilities = assortment * self.class_utilities[class_index]
            chosen_probabilities = chosen_utilities / (np.sum(chosen_utilities) + 1)
            recomputed_obj += self.class_weights[class_index] * np.sum(
                chosen_probabilities * self.itemwise_values
            )

        if not self.outside_option_given:
            assortment = assortment[1:]
        return assortment, recomputed_obj


class ORToolsLatentClassPricingOptimizer(object):
    """Assortment optimizer for latent class models with additional pricing optimization.

    Implementation of the paper:
    Isabel Méndez-Díaz, Juan José Miranda-Bront, Gustavo Vulcano, Paula Zabala,
    A branch-and-cut algorithm for the latent-class logit assortment problem,
    Discrete Applied Mathematics,
    Volume 164, Part 1,
    2014,
    Pages 246-263,
    ISSN 0166-218X,
    https://doi.org/10.1016/j.dam.2012.03.003.
    """

    def __init__(
        self,
        class_weights,
        class_utilities,
        itemwise_values,
        assortment_size,
        outside_option_given=False,
    ):
        """Initialize the AssortmentOptimizer object.

        Parameters
        ----------
        class_weights: Iterable
            List of weights for each latent class.
            Must have shape of (n_classes)
        class_utilities: Iterable
            List of utilities for each item of each latent class.
            Must have a shape of (n_classes, n_items, n_prices)
        itemwise_values: Iterable
            List of to-be-optimized values for each item, e.g. prices.
            Must have shape of (n_items, n_prices)
        assortment_size : int
            maximum size of the requested assortment.
        outside_option_given : bool
            Whether the outside option is given or not (and thus is automatically added).
        """
        for i in range(len(class_utilities)):
            if len(class_utilities[i]) != len(itemwise_values):
                raise ValueError(
                    f"You should provide as many utilities as itemwise values.\
                                Found {len(class_utilities[0])} and {len(itemwise_values)} instead."
                )
        if len(class_weights) != len(class_utilities):
            raise ValueError("Number of classes and class weights should be the same.")

        for class_index in range(len(class_weights)):
            for item_index in range(len(itemwise_values)):
                if len(class_utilities[class_index][item_index]) != len(
                    itemwise_values[item_index]
                ):
                    raise ValueError(
                        f"You should provide as many utilities as itemwise values for each item\
                            Did not match for class {class_index} and item {item_index}."
                    )

        self.outside_option_given = outside_option_given
        if not self.outside_option_given:
            self.outside_utility = [np.exp(0.0) for _ in range(len(class_weights))]
            self.outside_value = [0.0 for _ in range(len(class_weights))]

            self.class_utilities = class_utilities
            self.itemwise_values = itemwise_values
        else:
            # TO DO
            self.class_utilities = class_utilities
            self.itemwise_values = itemwise_values
        self.n_items = len(self.itemwise_values) - 1
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
        """Instantiate the solver.

        Returns
        -------
        gurobipy.Model
            solver with basic variables and constraints.
        """
        # Create a new model
        solver = pywraplp.Solver("PricingMIP", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        # solver = pywraplp.Solver.CreateSolver("SCIP")
        infinity = solver.infinity()

        # Create yik variables
        y = {(0, 0): solver.IntVar(lb=0.0, ub=1.0, name="y_0_0")}

        for i in range(1, self.n_items + 1):
            for k in range(len(self.itemwise_values[i - 1])):
                y[(i, k)] = solver.IntVar(lb=0.0, ub=1.0, name=f"y_{i}_{k}")
        self.y = y

        # Create zlik variables
        z = {
            (0, 0, 0): solver.NumVar(ub=infinity, name="z_0_0_0", lb=0),
            (1, 0, 0): solver.NumVar(ub=infinity, name="z_1_0_0", lb=0),
            (2, 0, 0): solver.NumVar(ub=infinity, name="z_2_0_0", lb=0),
        }
        for i in range(1, self.n_items + 1):
            for class_index in range(len(self.class_weights)):
                for k in range(len(self.itemwise_values[i - 1])):
                    z[(class_index, i, k)] = solver.NumVar(
                        ub=infinity, name=f"z_{class_index}_{i}_{k}", lb=0
                    )
        self.z = z

        # Create xl variables
        x = {}
        for class_index in range(len(self.class_weights)):
            x[class_index] = solver.NumVar(ub=infinity, name=f"x_{class_index}", lb=0)
        self.x = x

        return solver

    def set_base_constraints(self):
        """Functions to set LP base constraints.

        In particular, ensures Charnes-Cooper transformation constraints
        and assortment size constraint.
        """
        # Base Charnes-Cooper Constraint for Normalization
        for class_index in range(len(self.class_weights)):
            charnes_cooper = 0
            for i in range(1, self.n_items + 1):
                for k in range(len(self.itemwise_values[i - 1])):
                    charnes_cooper += (
                        self.class_utilities[class_index][i - 1][k] * self.z[(class_index, i, k)]
                    )
            charnes_cooper = (
                charnes_cooper + self.x[class_index] * self.outside_utility[class_index]
            )
            self.solver.Add(charnes_cooper == 1)

        # Charnes-Cooper for variables constraints
        for class_index in range(len(self.class_weights)):
            for i in range(1, self.n_items + 1):
                for k in range(len(self.itemwise_values[i - 1])):
                    self.solver.Add(self.z[(class_index, i, k)] <= self.x[class_index])
                    u = (
                        self.x[class_index] * self.outside_utility[class_index]
                        - self.z[(class_index, i, k)] * self.outside_utility[class_index]
                    )

                    self.solver.Add(u <= (1 - self.y[(i, k)]))
                    self.solver.Add(
                        self.z[(class_index, i, k)]
                        * (
                            self.outside_utility[class_index]
                            + self.class_utilities[class_index][i - 1][k]
                        )
                        <= self.y[(i, k)]
                    )
        self.ctr = {}
        # Unique price constraint
        for i in range(1, self.n_items + 1):
            self.ctr[i] = self.solver.Add(
                sum([self.y[(i, k)] for k in range(len(self.itemwise_values[i - 1]))]) <= 1
            )
        # Assortment size constraint
        if self.assortment_size is not None:
            self.solver.Add(
                sum(
                    [
                        self.y[(j, k)]
                        for j in range(1, self.n_items + 1)
                        for k in range(len(self.itemwise_values[j - 1]))
                    ]
                )
                <= self.assortment_size
            )

    def base_set_objective_function(self):
        """Set optimization objective."""
        objective = 0
        for class_index in range(len(self.class_weights)):
            for j in range(1, self.n_items + 1):
                for k in range(len(self.itemwise_values[j - 1])):
                    objective += (
                        self.class_weights[class_index]
                        * self.itemwise_values[j - 1][k]
                        * self.class_utilities[class_index][j - 1][k]
                        * self.z[(class_index, j, k)]
                    )
        self.solver.Maximize(objective)

    def set_objective_function(self, itemwise_values):
        """Define the objective function to maximize with the assortment.

        Parameters
        ----------
        itemwise_values : list-like
            List of values for each item - total value to be optimized.
        """
        raise NotImplementedError

    def add_constraint(self):
        """Add constraints."""
        raise NotImplementedError

    def add_maximal_capacity_constraint(self, itemwise_capacities, maximum_capacity):
        """Add maximal capacity constraint.

        The added constraint is basically sum_{i} y_i * itemwise_capacities[i] <= maximum_capacity

        Parameters
        ----------
        itemwise_capacities : Iterable
            Values of capacity for each item in the assortment.
            Shape must match with itemwise_values.
        maximum_capacity : int/float
            Value of the maximal capacity.
        """
        assortment_capacity = sum(
            [
                self.y[(j, k)] * itemwise_capacities[j - 1]
                for j in range(1, self.n_items + 1)
                for k in range(len(self.itemwise_values[j - 1]))
            ]
        )
        self.solver.Add(assortment_capacity <= maximum_capacity)

    def add_minimal_capacity_constraint(self, itemwise_capacities, minimum_capacity):
        """Add maximal capacity constraint.

        The added constraint is basically sum_{i} y_i * itemwise_capacities[i] <= maximum_capacity

        Parameters
        ----------
        itemwise_capacities : Iterable
            Values of capacity for each item in the assortment.
            Shape must match with itemwise_values.
        minimum_capacity : int/float
            Value of the maximal capacity.
        """
        assortment_capacity = sum(
            [self.y[j] * itemwise_capacities[j - 1] for j in range(1, self.n_items + 1)]
        )
        self.solver.Add(assortment_capacity >= minimum_capacity)

    def solve(self):
        """Solve the optimization problem.

        Returns
        -------
        np.ndarray:
            Array of 0s and 1s, indicating the presence of each item in the optimal assortment.
        """
        self.base_set_objective_function()

        # -- Optimize --
        status = self.solver.Solve()
        self.status = status

        assortment = np.zeros(self.n_items + 1)
        for i in range(1, self.n_items + 1):
            for k in range(len(self.itemwise_values[i - 1])):
                if self.y[(i, k)].solution_value() > 0:
                    assortment[i - 1] = self.itemwise_values[i - 1][k]

        return assortment, self.solver.Objective().Value()
        # return assortment, recomputed_obj
