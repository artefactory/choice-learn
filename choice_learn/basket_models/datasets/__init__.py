"""Loaders for known datasets as TripDataset."""

from .bakery import load_bakery
from .badminton import BadmintonDataGenerator

__all__ = ["load_bakery", "BadmintonDataGenerator"]
