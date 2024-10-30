"""Basic tests for the RUMnet model."""

import numpy as np
import pytest
import tensorflow as tf

from choice_learn.data import ChoiceDataset
from choice_learn.models.rumnet import (
    AssortmentParallelDense,
    AssortmentUtilityDenseNetwork,
    CPURUMnet,
    GPURUMnet,
    PaperRUMnet,
    ParallelDense,
)

dataset = ChoiceDataset(
    items_features_by_choice=(
        np.array(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]],
                [[1.2, 2.2], [3.2, 4.2], [5.2, 6.2]],
                [[1.3, 2.3], [3.3, 4.3], [5.3, 6.3]],
            ]
        ).astype("float32"),
    ),
    shared_features_by_choice=(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]).astype(
            "float32"
        ),
    ),
    choices=np.array([0, 1, 2, 0]),
)


def test_parallel_dense():
    """Tests the ParallelDense Layer."""
    tf.config.run_functions_eagerly(True)

    layer = ParallelDense(
        width=8,
        depth=4,
        heterogeneity=2,
        activation="relu",
    )
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    output_tensor = layer(input_tensor)
    assert output_tensor.shape == (3, 8, 2)
    assert len(layer.trainable_variables) == 8

    for i, w in enumerate(layer.trainable_variables):
        if i == 0:
            assert w.shape == (2, 8, 2)
        elif i % 2 == 0:
            assert w.shape == (8, 8, 2)
        else:
            assert w.shape == (8, 2)


def test_assortment_utility_dense():
    """Tests the AssortmentUtilityDenseNetwork Layer."""
    tf.config.run_functions_eagerly(True)

    layer = AssortmentUtilityDenseNetwork(
        width=4,
        depth=2,
        add_last=True,
        activation="relu",
    )
    input_tensor = np.array(
        [
            [[[1.0, 2.0], [1.4, 2.4]], [[3.0, 4.0], [3.4, 4.4]], [[5.0, 6.0], [5.4, 6.4]]],
            [[[1.1, 2.1], [1.5, 2.5]], [[3.1, 4.1], [3.5, 4.5]], [[5.1, 6.1], [5.5, 6.5]]],
            [[[1.2, 2.2], [1.6, 2.6]], [[3.2, 4.2], [3.6, 4.6]], [[5.2, 6.2], [5.6, 6.6]]],
            [[[1.3, 2.3], [1.7, 2.7]], [[3.3, 4.3], [3.7, 4.7]], [[5.3, 6.3], [5.7, 6.7]]],
        ]
    )
    output_tensor = layer(input_tensor)
    assert output_tensor.shape == (4, 3, 1, 2)
    assert len(layer.trainable_variables) == 5

    for i, w in enumerate(layer.trainable_variables):
        if i == 0:
            assert w.shape == (2, 4)
        elif i == 4:
            assert w.shape == (4, 1)
        elif i % 2 == 0:
            assert w.shape == (4, 4)
        else:
            assert w.shape == (4, 1)


def test_assortment_parallel_dense():
    """Tests the AssortmentParallelDense Layer."""
    tf.config.run_functions_eagerly(True)

    layer = AssortmentParallelDense(
        width=8,
        depth=4,
        heterogeneity=2,
        activation="relu",
    )
    input_tensor = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]],
            [[1.2, 2.2], [3.2, 4.2], [5.2, 6.2]],
            [[1.3, 2.3], [3.3, 4.3], [5.3, 6.3]],
        ]
    )
    output_tensor = layer(input_tensor)
    assert output_tensor.shape == (4, 3, 8, 2)
    assert len(layer.trainable_variables) == 8

    for i, w in enumerate(layer.trainable_variables):
        if i == 0:
            assert w.shape == (2, 8, 2)
        elif i % 2 == 0:
            assert w.shape == (8, 8, 2)
        else:
            assert w.shape == (8, 2)


def test_paper_rumnet_errors():
    """Tests errors raisded by PaperRUMnet model."""
    with pytest.raises(ValueError):
        model = PaperRUMnet(
            num_products_features=0,
            num_customer_features=3,
            width_eps_x=3,
            depth_eps_x=2,
            heterogeneity_x=2,
            width_eps_z=3,
            depth_eps_z=2,
            heterogeneity_z=2,
            width_u=3,
            depth_u=1,
            tol=1e-5,
            optimizer="adam",
            lr=0.001,
        )
        model.instantiate()
    with pytest.raises(ValueError):
        model = PaperRUMnet(
            num_products_features=2,
            num_customer_features=0,
            width_eps_x=3,
            depth_eps_x=2,
            heterogeneity_x=2,
            width_eps_z=3,
            depth_eps_z=2,
            heterogeneity_z=2,
            width_u=3,
            depth_u=1,
            tol=1e-5,
            optimizer="adam",
            lr=0.001,
            epochs=2,
        )
        model.instantiate()


def test_paper_rumnet():
    """Tests the PaperRUMnet model."""
    tf.config.run_functions_eagerly(True)
    global dataset

    model = PaperRUMnet(
        num_products_features=2,
        num_customer_features=3,
        width_eps_x=3,
        depth_eps_x=2,
        heterogeneity_x=2,
        width_eps_z=3,
        depth_eps_z=2,
        heterogeneity_z=2,
        width_u=3,
        depth_u=1,
        tol=1e-5,
        optimizer="adam",
        lr=0.0005,
        epochs=10,
        batch_size=4,
    )
    model.instantiate()
    nll_a = model.evaluate(dataset)
    model.fit(dataset)
    nll_b = model.evaluate(dataset)
    assert nll_b < nll_a
    assert model.batch_predict(
        dataset.shared_features_by_choice[0],
        dataset.items_features_by_choice[0],
        np.ones((4, 3)),
        dataset.choices,
        None,
    )[1].shape == (4, 3)


def test_cpu_rumnet():
    """Tests the CPURUMNet model."""
    tf.config.run_functions_eagerly(True)
    global dataset

    model = CPURUMnet(
        num_products_features=2,
        num_customer_features=3,
        width_eps_x=4,
        depth_eps_x=3,
        heterogeneity_x=2,
        width_eps_z=4,
        depth_eps_z=3,
        heterogeneity_z=2,
        width_u=4,
        depth_u=3,
        tol=1e-5,
        optimizer="adam",
        lr=0.001,
        epochs=2,
    )
    model.instantiate()
    assert model.batch_predict(
        dataset.shared_features_by_choice[0],
        dataset.items_features_by_choice[0],
        np.ones((4, 3)),
        dataset.choices,
        None,
    )[1].shape == (4, 3)

    assert model.batch_predict(
        (dataset.shared_features_by_choice[0],),
        (dataset.items_features_by_choice[0],),
        np.ones((4, 3)),
        dataset.choices,
        None,
    )[1].shape == (4, 3)


def test_gpu_rumnet():
    """Tests the GPURUMNet model."""
    tf.config.run_functions_eagerly(True)
    global dataset

    model = GPURUMnet(
        num_products_features=2,
        num_customer_features=3,
        width_eps_x=4,
        depth_eps_x=3,
        heterogeneity_x=2,
        width_eps_z=4,
        depth_eps_z=3,
        heterogeneity_z=2,
        width_u=4,
        depth_u=3,
        tol=1e-5,
        optimizer="adam",
        lr=0.01,
        epochs=5,
    )
    model.instantiate()
    assert model.batch_predict(
        dataset.shared_features_by_choice[0],
        dataset.items_features_by_choice[0],
        np.ones((4, 3)),
        dataset.choices,
        None,
    )[1].shape == (4, 3)
    nll_a = model.evaluate(dataset)
    model.fit(dataset)
    nll_b = model.evaluate(dataset)
    assert nll_b < nll_a

    assert model.batch_predict(
        (dataset.shared_features_by_choice[0],),
        (dataset.items_features_by_choice[0],),
        np.ones((4, 3)),
        dataset.choices,
        None,
    )[1].shape == (4, 3)
