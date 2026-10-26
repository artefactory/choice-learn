"""Tests SimpleMNL."""

import tensorflow as tf

from choice_learn.datasets import load_swissmetro
from choice_learn.models import MultiNomialLogit

dataset = load_swissmetro()


def test_early_stopping():
    """Tests early stopping callback."""
    tf.config.run_functions_eagerly(True)
    global dataset

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="train_loss", patience=10, min_delta=1e-3, mode="min"
        )
    ]
    model = MultiNomialLogit(
        epochs=1_000, optimizer="adam", callbacks=callbacks, lr=1e-2, batch_size=-1
    )
    _ = model.fit(dataset[:64])
    assert model.stop_training


def test_reduce_lr_on_plateau():
    """Tests ReduceLROnPlateau callback."""
    tf.config.run_functions_eagerly(True)
    global dataset

    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor="train_loss", patience=5, mode="min")]
    model = MultiNomialLogit(
        epochs=1_00, optimizer="adam", callbacks=callbacks, lr=1e-1, batch_size=-1
    )
    _ = model.fit(dataset[:64])
    assert model.optimizer.learning_rate.numpy() < 1e-1
