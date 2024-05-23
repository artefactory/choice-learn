"""Tool function for assortment and pricing optimization."""


class MNLAssortmentOptimizer(object):
    """Base class for assortment optimization."""

    def __new__(
        cls, solver, utilities, itemwise_values, assortment_size, outside_option_given=False
    ):
        """Create the AssortmentOptimizer object.

        Basically used to handle the choice of solver.

        Parameters
        ----------
        solver: str
            Name of the solver to be used. Currently only "gurobi" and "or-tools" is supported.
        utilities : Iterable
            List of utilities for each item.
        itemwise_values: Iterable
            List of to-be-optimized values for each item, e.g. prices.
        assortment_size : int
            maximum size of the requested assortment.
        outside_option_given : bool
            Whether the outside option is given or not (and thus is automatically added).
        """
        if solver.lower() == "gurobi":
            from .gurobi_opt import GurobiMNLAssortmentOptimizer

            return GurobiMNLAssortmentOptimizer(
                utilities=utilities,
                itemwise_values=itemwise_values,
                assortment_size=assortment_size,
                outside_option_given=outside_option_given,
            )
        if solver.lower() == "or-tools" or solver.lower() == "ortools":
            from .or_tools_opt import ORToolsMNLAssortmentOptimizer

            return ORToolsMNLAssortmentOptimizer(
                utilities=utilities,
                itemwise_values=itemwise_values,
                assortment_size=assortment_size,
                outside_option_given=outside_option_given,
            )

        raise ValueError("Unknown solver. Please choose between 'gurobi' and 'or-tools'.")


class LatentClassAssortmentOptimizer(object):
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

    def __new__(
        cls,
        solver,
        class_weights,
        class_utilities,
        itemwise_values,
        assortment_size,
        outside_option_given=False,
    ):
        """Create the AssortmentOptimizer object.

        Basically used to handle the choice of solver.

        Parameters
        ----------
        solver: str
            Name of the solver to be used. Currently only "gurobi" and "or-tools" is supported.
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
        if solver.lower() == "gurobi":
            from .gurobi_opt import GurobiLatentClassAssortmentOptimizer

            return GurobiLatentClassAssortmentOptimizer(
                class_weights=class_weights,
                class_utilities=class_utilities,
                itemwise_values=itemwise_values,
                assortment_size=assortment_size,
                outside_option_given=outside_option_given,
            )
        if solver.lower() == "or-tools" or solver.lower() == "ortools":
            from .or_tools_opt import ORToolsLatentClassAssortmentOptimizer

            return ORToolsLatentClassAssortmentOptimizer(
                class_weights=class_weights,
                class_utilities=class_utilities,
                itemwise_values=itemwise_values,
                assortment_size=assortment_size,
                outside_option_given=outside_option_given,
            )

        raise ValueError("Unknown solver. Please choose between 'gurobi' and 'or-tools'.")


class LatentClassPricingOptimizer(object):
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

    def __new__(
        cls,
        solver,
        class_weights,
        class_utilities,
        itemwise_values,
        assortment_size,
        outside_option_given=False,
    ):
        """Create the AssortmentOptimizer object.

        Basically used to handle the choice of solver.

        Parameters
        ----------
        solver: str
            Name of the solver to be used. Currently only "gurobi" and "or-tools" is supported.
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
        if solver.lower() == "gurobi":
            from .gurobi_opt import GurobiLatentClassPricingOptimizer

            return GurobiLatentClassPricingOptimizer(
                class_weights=class_weights,
                class_utilities=class_utilities,
                itemwise_values=itemwise_values,
                assortment_size=assortment_size,
                outside_option_given=outside_option_given,
            )
        if solver.lower() == "or-tools" or solver.lower() == "ortools":
            from .or_tools_opt import ORToolsLatentClassPricingOptimizer

            return ORToolsLatentClassPricingOptimizer(
                class_weights=class_weights,
                class_utilities=class_utilities,
                itemwise_values=itemwise_values,
                assortment_size=assortment_size,
                outside_option_given=outside_option_given,
            )

        raise ValueError("Unknown solver. Please choose between 'gurobi' and 'or-tools'.")
