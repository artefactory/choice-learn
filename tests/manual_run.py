"""Runs manually all test_* functions."""
import importlib
import os
import sys
from pathlib import Path


def run_module(file):
    """Run all functions in file file.

    Parameters
    ----------
    file : str
        .py file
    """
    sub_file = str(file).replace(".py", "").replace("/", ".")[6:]
    __import__(sub_file)
    module = importlib.import_module(sub_file, package="./")
    for i in dir(module):
        item = getattr(module, i)
        if callable(item) and item.__name__.startswith("test_"):
            item()


if __name__ == "__main__":
    directory = Path("./tests/")
    for file in directory.glob("./**/*.py"):
        if "manual_run" not in str(file):
            run_module(file)
