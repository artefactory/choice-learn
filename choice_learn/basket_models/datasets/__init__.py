"""Loaders for known datasets as TripDataset."""

from .badminton import BadmintonDataGenerator
from .bakery import load_bakery

__all__ = ["load_bakery", "BadmintonDataGenerator"]
