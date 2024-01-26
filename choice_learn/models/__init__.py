"""Models classes and functions."""
import tensorflow as tf

from .conditional_mnl import ConditionalMNL

if len(tf.config.list_physical_devices("GPU")) > 0:
    print("GPU detected, importing GPU version of RUMnet.")
    from .rumnet import GPURUMnet as RUMnet
else:
    from .rumnet import CPURUMnet as RUMnet

    print("No GPU detected, importing CPU version of RUMnet.")

__all__ = ["ConditionalMNL", "RUMnet"]
