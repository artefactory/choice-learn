"""Tests ResLogit."""

import numpy as np
import pytest

from choice_learn.data import ChoiceDataset
from choice_learn.datasets import load_swissmetro
from choice_learn.models import ResLogit

dataset = load_swissmetro()
dataset = dataset[:10]  # Reduce the dataset size for faster testing
n_items = np.shape(dataset.items_features_by_choice)[2]
n_shared_features = np.shape(dataset.shared_features_by_choice)[2]
n_items_features = np.shape(dataset.items_features_by_choice)[3]


lr = 1e-4
epochs = 2
batch_size = -1


def test_reslogit_different_layers_width():
    """Tests that ResLogit can fit with different custom widths for its residual layers."""
    global dataset

    list_n_layers = [3, 3]
    list_res_layers_width = [[12, n_items, 2], [12, 2]]

    for n_layers, res_layers_width in zip(list_n_layers, list_res_layers_width):
        model = ResLogit(
            n_layers=n_layers,
            res_layers_width=res_layers_width,
            lr=lr,
            epochs=epochs,
            optimizer="Adam",
            batch_size=batch_size,
        )
        # The model can fit
        with pytest.raises(ValueError):
            model.instantiate(n_items, n_shared_features, n_items_features)


def test_different_data_types():
    """Tests that ResLogit can fit with different custom widths for its residual layers."""
    global lr, epochs, batch_size
    choices = [0, 2, 1]
    available_items_by_choice = [
        [1, 1, 1],  # All items available at session 1
        [1, 1, 1],  # All items available at session 2
        [0, 1, 1],  # Item 1 not available at session 3
    ]

    shared_features_by_choice = [
        [100, 20],  # session 1, customer 1 [budget, age]
        [200, 40],  # session 2, customer 2 [budget, age]
        [80, 20],  # session 3, customer 1 [budget, age]
    ]

    items_features_by_choice = [
        [
            [100, 0],  # Session 1, Item 1 [price, promotion]
            [140, 0],  # Session 1, Item 2 [price, promotion]
            [200, 0],  # Session 1, Item 2 [price, promotion]
        ],
        [
            [100, 0],  # Session 2 Item 1 [price, promotion]
            [120, 1],  # Session 2, Item 2 [price, promotion]
            [200, 0],  # Session 2, Item 2 [price, promotion]
        ],
        [
            [0, 0],  # Session 3, Item 1 [price, promotion],
            # values do not really matter, but needs to exist for shapes sake
            [120, 1],  # Session 3, Item 2 [price, promotion]
            [180, 1],  # Session 3, Item 2 [price, promotion]
        ],
    ]

    dataset = ChoiceDataset(
        shared_features_by_choice=shared_features_by_choice,
        items_features_by_choice=items_features_by_choice,
        available_items_by_choice=available_items_by_choice,
        choices=choices,
    )

    model = ResLogit(
        n_layers=3,
        res_layers_width=[12, 3],
        lr=lr,
        epochs=epochs,
        optimizer="Adam",
        batch_size=batch_size,
    )
    # The model can fit
    model.instantiate(n_items=3, n_shared_features=2, n_items_features=2)
    model.fit(dataset)

    dataset = ChoiceDataset(
        shared_features_by_choice=None,
        items_features_by_choice=(items_features_by_choice, items_features_by_choice),
        available_items_by_choice=available_items_by_choice,
        choices=choices,
    )

    model = ResLogit(
        n_layers=3,
        res_layers_width=[12, 3],
        lr=lr,
        epochs=epochs,
        optimizer="Adam",
        batch_size=batch_size,
    )
    # The model can fit
    model.instantiate(n_items=3, n_shared_features=0, n_items_features=4)
    model.fit(dataset)

    dataset = ChoiceDataset(
        shared_features_by_choice=(shared_features_by_choice, shared_features_by_choice),
        items_features_by_choice=None,
        available_items_by_choice=available_items_by_choice,
        choices=choices,
    )

    model = ResLogit(
        n_layers=3,
        res_layers_width=[12, 3],
        lr=lr,
        epochs=epochs,
        optimizer="Adam",
        batch_size=batch_size,
    )
    # The model can fit
    model.instantiate(n_items=3, n_shared_features=4, n_items_features=0)
    model.fit(dataset)
