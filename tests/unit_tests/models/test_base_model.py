"""Testing base ChoiceModel."""

import numpy as np
import pytest
import tensorflow as tf

from choice_learn.models.base_model import ChoiceModel


def test_adamax_optimizer():
    """Check that model uses Adamax optimizer when set as arg."""
    model = ChoiceModel(optimizer="Adamax")
    assert isinstance(model.optimizer, tf.keras.optimizers.Adamax)


def test_not_recognized_optimizer():
    """Check that model switches to Adam when an optimizer is not recognized."""
    model = ChoiceModel(optimizer="bad_optimizer")
    assert isinstance(model.optimizer, tf.keras.optimizers.Adam)


def test_regularizer_instantiation():
    """Checks that the model instantiates reg."""
    model = ChoiceModel(regularization="L1", regularization_strength=0.1)
    assert isinstance(model.regularizer, tf.keras.regularizers.L1)
    assert (model.regularizer.l1 == np.array(0.1, dtype=np.float32)).all()
    model = ChoiceModel(regularization="l1", regularization_strength=0.1)
    assert isinstance(model.regularizer, tf.keras.regularizers.L1)
    assert (model.regularizer.l1 == np.array(0.1, dtype=np.float32)).all()
    model = ChoiceModel(regularization="L2", regularization_strength=0.01)
    assert isinstance(model.regularizer, tf.keras.regularizers.L2)
    assert (model.regularizer.l2 == np.array(0.01, dtype=np.float32)).all()
    model = ChoiceModel(regularization="l2", regularization_strength=0.01)
    assert isinstance(model.regularizer, tf.keras.regularizers.L2)
    assert (model.regularizer.l2 == np.array(0.01, dtype=np.float32)).all()

    model = ChoiceModel(regularization="l1L2", regularization_strength=0.2)
    assert isinstance(model.regularizer, tf.keras.regularizers.L1L2)
    assert (model.regularizer.l1 == np.array(0.2, dtype=np.float32)).all()
    assert (model.regularizer.l2 == np.array(0.2, dtype=np.float32)).all()
    model = ChoiceModel(regularization="L1l2", regularization_strength=(0.2, 0.2))
    assert isinstance(model.regularizer, tf.keras.regularizers.L1L2)
    assert (model.regularizer.l1 == np.array(0.2, dtype=np.float32)).all()
    assert (model.regularizer.l2 == np.array(0.2, dtype=np.float32)).all()
    model = ChoiceModel(regularization="L1L2", regularization_strength=(0.2, 0.02))
    assert isinstance(model.regularizer, tf.keras.regularizers.L1L2)
    assert (model.regularizer.l1 == np.array(0.2, dtype=np.float32)).all()
    assert (model.regularizer.l2 == np.array(0.02, dtype=np.float32)).all()


def test_regularizer_instant_error():
    """Checks that the model instantiates reg."""
    with pytest.raises(ValueError):
        _ = ChoiceModel(regularization="L3", regularization_strength=0.1)
    with pytest.raises(ValueError):
        _ = ChoiceModel(regularization="L1", regularization_strength=0.0)
