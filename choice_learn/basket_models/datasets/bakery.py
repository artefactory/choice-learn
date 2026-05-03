"""Base TripDataset loader for the Bakery dataset from Benson et al. (2018)."""

import os
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from choice_learn.datasets.base import download_from_url, get_path

from ..data.basket_dataset import Trip, TripDataset

DATA_MODULE = "choice_learn/datasets/data"


def load_bakery(as_frame=False, load_5_25_version=False):
    """Load the bakery dataset from uchoice-Bakery.txt.

    Parameters
    ----------
    as_frame : bool, optional
        Whether to return the dataset as pd.DataFrame. If not, returned as TripDataset,
        by default False.
    load_5_25_version : bool, optional
        Whether to return the 5-25 version of the dataset,
        by default False.
    """
    url = "https://drive.usercontent.google.com/u/0/uc?id=1qV8qmiHTq6y5fwgN0_hRXyKreNKrF72E&export=download"
    data_file_name = download_from_url(url)

    archive_path = get_path(data_file_name)
    # We put the extracted files in the data directory
    with tarfile.open(archive_path, "r:gz") as tar:
        # Here are the files we are downloading
        file_names = tar.getnames()

        # Security check for safe member paths (defense-in-depth for all Python versions)
        for member in tar.getmembers():
            member_path = os.path.normpath(member.name)
            if (
                Path(member_path).is_absolute()
                or member_path.startswith("..")
                or ".." in member_path.split(os.path.sep)
            ):
                raise ValueError(f"tar archive contains unsafe path: {member.name}")

        # We extract all the files
        if sys.version_info >= (3, 12):
            tar.extractall(path=archive_path.parent, filter="data")  # nosec
        else:
            tar.extractall(path=archive_path.parent)  # nosec

        # We want to read the uchoice-Bakery.txt file (second file in the archive)
    if load_5_25_version:
        csv_file_to_read = "uchoice-Bakery/uchoice-Bakery-5-25.txt"
    else:
        csv_file_to_read = "uchoice-Bakery/uchoice-Bakery.txt"
    if csv_file_to_read not in file_names:
        raise FileNotFoundError(f"'{csv_file_to_read}' not found in the archive.")

    noms_colonnes = [
        "article_1",
        "article_2",
        "article_3",
        "article_4",
        "article_5",
        "article_6",
        "article_7",
        "article_8",
    ]

    # likewise get_path function
    path = archive_path.parent / csv_file_to_read
    df = pd.read_csv(path, sep=r"\s+", header=None, names=noms_colonnes)

    if as_frame:
        return df

    n_item = int(df.max().max())

    # Apparently all items are available at each trip
    availability_matrix = np.array([[1] * n_item])

    list_purchases = [[int(item) - 1 for item in row if pd.notna(item)] for row in df.to_numpy()]

    # Dummy prices, all equal to 1
    prices = np.array([[1] * n_item])
    trips_list = [
        Trip(purchases=purchases, assortment=0, prices=prices) for purchases in list_purchases
    ]

    return TripDataset(trips=trips_list, available_items=availability_matrix)
