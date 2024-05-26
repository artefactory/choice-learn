"""Runs manually all test_* functions."""
import importlib
from pathlib import Path


def run_module(file):
    """Run all functions in file file.

    Parameters
    ----------
    file : str
        .py file
    """
    sub_file = file.replace(".py", "")
    module = importlib.import_module(sub_file, package="__name__")
    for i in dir(module):
        item = getattr(module, i)
        if callable(item) and item.startswith("test_"):
            item()


if __name__ == "__main__":
    for file in Path.glob("./**/*.py"):
        run_module(file)
