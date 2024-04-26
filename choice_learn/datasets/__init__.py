"""Init file for datasets module."""

from .base import (
    load_car_preferences,
    load_electricity,
    load_hc,
    load_heating,
    load_modecanada,
    load_swissmetro,
    load_train,
)
from .expedia import load_expedia
from .tafeng import load_tafeng

__all__ = [
    "load_modecanada",
    "load_swissmetro",
    "load_electricity",
    "load_heating",
    "load_train",
    "load_tafeng",
    "load_expedia",
    "load_car_preferences",
    "load_hc",
]
