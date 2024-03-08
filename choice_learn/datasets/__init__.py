"""Init file for datasets module."""

from .base import load_electricity, load_heating, load_modecanada, load_swissmetro, load_train

__all__ = ["load_modecanada", "load_swissmetro", "load_electricity", "load_heating", "load_train"]
