"""Models classes and functions."""

import logging

import tensorflow as tf

from .conditional_logit import ConditionalLogit
from .learning_mnl import LearningMNL
from .nested_logit import NestedLogit
from .reslogit import ResLogit
from .simple_mnl import SimpleMNL
from .tastenet import TasteNet

if len(tf.config.list_physical_devices("GPU")) > 0:
    logging.info("GPU detected, importing GPU version of RUMnet.")
    from .rumnet import GPURUMnet as RUMnet
else:
    from .rumnet import CPURUMnet as RUMnet

    logging.info("No GPU detected, importing CPU version of RUMnet.")

__all__ = [
    "ConditionalLogit",
    "RUMnet",
    "SimpleMNL",
    "TasteNet",
    "NestedLogit",
    "ResLogit",
    "LearningMNL",
]
