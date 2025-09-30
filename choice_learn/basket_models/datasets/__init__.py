"""Loaders for known datasets as TripDataset."""

from .bakery import load_bakery
from .synthetic_dataset import SyntheticDataGenerator

__all__ = ["load_bakery", "SyntheticDataGenerator"]
