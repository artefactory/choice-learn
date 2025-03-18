"""Tests for the assortment_optimizer.py module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from choice_learn.toolbox.assortment_optimizer import (
    LatentClassAssortmentOptimizer,
    LatentClassPricingOptimizer,
    MNLAssortmentOptimizer,
)


class TestMNLAssortmentOptimizer:
    """Tests for the MNLAssortmentOptimizer class."""

    @patch("choice_learn.toolbox.assortment_optimizer.GurobiMNLAssortmentOptimizer")
    def test_gurobi_solver(self, mock_gurobi):
        """Test using the Gurobi solver."""
        # Setup test data
        utilities = [0.1, 0.2, 0.3]
        itemwise_values = [10, 20, 30]
        assortment_size = 2

        # Create mock return instance
        mock_instance = MagicMock()
        mock_gurobi.return_value = mock_instance

        # Call the optimizer
        optimizer = MNLAssortmentOptimizer(
            solver="gurobi",
            utilities=utilities,
            itemwise_values=itemwise_values,
            assortment_size=assortment_size,
        )

        # Check if correct solver was instantiated
        mock_gurobi.assert_called_once_with(
            utilities=utilities,
            itemwise_values=itemwise_values,
            assortment_size=assortment_size,
            outside_option_given=False,
        )

        # Check if the returned instance is the mock
        assert optimizer == mock_instance

    @patch("choice_learn.toolbox.assortment_optimizer.ORToolsMNLAssortmentOptimizer")
    def test_ortools_solver(self, mock_ortools):
        """Test using the OR-Tools solver."""
        # Setup test data
        utilities = [0.1, 0.2, 0.3]
        itemwise_values = [10, 20, 30]
        assortment_size = 2

        # Create mock return instance
        mock_instance = MagicMock()
        mock_ortools.return_value = mock_instance

        # Call the optimizer with 'or-tools'
        optimizer = MNLAssortmentOptimizer(
            solver="or-tools",
            utilities=utilities,
            itemwise_values=itemwise_values,
            assortment_size=assortment_size,
        )

        # Check if correct solver was instantiated
        mock_ortools.assert_called_once_with(
            utilities=utilities,
            itemwise_values=itemwise_values,
            assortment_size=assortment_size,
            outside_option_given=False,
        )

        # Call the optimizer with 'ortools'
        optimizer = MNLAssortmentOptimizer(
            solver="ortools",
            utilities=utilities,
            itemwise_values=itemwise_values,
            assortment_size=assortment_size,
        )

        # Check if correct solver was instantiated again
        mock_ortools.assert_called_with(
            utilities=utilities,
            itemwise_values=itemwise_values,
            assortment_size=assortment_size,
            outside_option_given=False,
        )

        # Check if the returned instance is the mock
        assert optimizer == mock_instance

    def test_invalid_solver(self):
        """Test using an invalid solver name."""
        # Setup test data
        utilities = [0.1, 0.2, 0.3]
        itemwise_values = [10, 20, 30]
        assortment_size = 2

        # Call the optimizer with invalid solver
        with pytest.raises(
            ValueError, match="Unknown solver. Please choose between 'gurobi' and 'or-tools'."
        ):
            MNLAssortmentOptimizer(
                solver="invalid",
                utilities=utilities,
                itemwise_values=itemwise_values,
                assortment_size=assortment_size,
            )


class TestLatentClassAssortmentOptimizer:
    """Tests for the LatentClassAssortmentOptimizer class."""

    @patch("choice_learn.toolbox.assortment_optimizer.GurobiLatentClassAssortmentOptimizer")
    def test_gurobi_solver(self, mock_gurobi):
        """Test using the Gurobi solver."""
        # Setup test data
        class_weights = [0.3, 0.7]
        class_utilities = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        itemwise_values = [10, 20, 30]
        assortment_size = 2

        # Create mock return instance
        mock_instance = MagicMock()
        mock_gurobi.return_value = mock_instance

        # Call the optimizer
        optimizer = LatentClassAssortmentOptimizer(
            solver="gurobi",
            class_weights=class_weights,
            class_utilities=class_utilities,
            itemwise_values=itemwise_values,
            assortment_size=assortment_size,
        )

        # Check if correct solver was instantiated
        mock_gurobi.assert_called_once_with(
            class_weights=class_weights,
            class_utilities=class_utilities,
            itemwise_values=itemwise_values,
            assortment_size=assortment_size,
            outside_option_given=False,
        )

        # Check if the returned instance is the mock
        assert optimizer == mock_instance

    @patch("choice_learn.toolbox.assortment_optimizer.ORToolsLatentClassAssortmentOptimizer")
    def test_ortools_solver(self, mock_ortools):
        """Test using the OR-Tools solver."""
        # Setup test data
        class_weights = [0.3, 0.7]
        class_utilities = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        itemwise_values = [10, 20, 30]
        assortment_size = 2

        # Create mock return instance
        mock_instance = MagicMock()
        mock_ortools.return_value = mock_instance

        # Call the optimizer
        optimizer = LatentClassAssortmentOptimizer(
            solver="or-tools",
            class_weights=class_weights,
            class_utilities=class_utilities,
            itemwise_values=itemwise_values,
            assortment_size=assortment_size,
        )

        # Check if correct solver was instantiated
        mock_ortools.assert_called_once_with(
            class_weights=class_weights,
            class_utilities=class_utilities,
            itemwise_values=itemwise_values,
            assortment_size=assortment_size,
            outside_option_given=False,
        )

        # Check if the returned instance is the mock
        assert optimizer == mock_instance

    def test_invalid_solver(self):
        """Test using an invalid solver name."""
        # Setup test data
        class_weights = [0.3, 0.7]
        class_utilities = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        itemwise_values = [10, 20, 30]
        assortment_size = 2

        # Call the optimizer with invalid solver
        with pytest.raises(
            ValueError, match="Unknown solver. Please choose between 'gurobi' and 'or-tools'."
        ):
            LatentClassAssortmentOptimizer(
                solver="invalid",
                class_weights=class_weights,
                class_utilities=class_utilities,
                itemwise_values=itemwise_values,
                assortment_size=assortment_size,
            )


class TestLatentClassPricingOptimizer:
    """Tests for the LatentClassPricingOptimizer class."""

    @patch("choice_learn.toolbox.assortment_optimizer.GurobiLatentClassPricingOptimizer")
    def test_gurobi_solver(self, mock_gurobi):
        """Test using the Gurobi solver."""
        # Setup test data
        class_weights = [0.3, 0.7]
        class_utilities = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        itemwise_values = [10, 20, 30]
        assortment_size = 2

        # Create mock return instance
        mock_instance = MagicMock()
        mock_gurobi.return_value = mock_instance

        # Call the optimizer
        optimizer = LatentClassPricingOptimizer(
            solver="gurobi",
            class_weights=class_weights,
            class_utilities=class_utilities,
            itemwise_values=itemwise_values,
            assortment_size=assortment_size,
        )

        # Check if correct solver was instantiated
        mock_gurobi.assert_called_once_with(
            class_weights=class_weights,
            class_utilities=class_utilities,
            itemwise_values=itemwise_values,
            assortment_size=assortment_size,
            outside_option_given=False,
        )

        # Check if the returned instance is the mock
        assert optimizer == mock_instance

    @patch("choice_learn.toolbox.assortment_optimizer.ORToolsLatentClassPricingOptimizer")
    def test_ortools_solver(self, mock_ortools):
        """Test using the OR-Tools solver."""
        # Setup test data
        class_weights = [0.3, 0.7]
        class_utilities = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        itemwise_values = [10, 20, 30]
        assortment_size = 2

        # Create mock return instance
        mock_instance = MagicMock()
        mock_ortools.return_value = mock_instance

        # Call the optimizer
        optimizer = LatentClassPricingOptimizer(
            solver="or-tools",
            class_weights=class_weights,
            class_utilities=class_utilities,
            itemwise_values=itemwise_values,
            assortment_size=assortment_size,
        )

        # Check if correct solver was instantiated
        mock_ortools.assert_called_once_with(
            class_weights=class_weights,
            class_utilities=class_utilities,
            itemwise_values=itemwise_values,
            assortment_size=assortment_size,
            outside_option_given=False,
        )

        # Check if the returned instance is the mock
        assert optimizer == mock_instance

    def test_invalid_solver(self):
        """Test using an invalid solver name."""
        # Setup test data
        class_weights = [0.3, 0.7]
        class_utilities = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        itemwise_values = [10, 20, 30]
        assortment_size = 2

        # Call the optimizer with invalid solver
        with pytest.raises(
            ValueError, match="Unknown solver. Please choose between 'gurobi' and 'or-tools'."
        ):
            LatentClassPricingOptimizer(
                solver="invalid",
                class_weights=class_weights,
                class_utilities=class_utilities,
                itemwise_values=itemwise_values,
                assortment_size=assortment_size,
            )
