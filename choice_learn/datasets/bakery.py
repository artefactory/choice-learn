import os
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from choice_learn.basket_models.data.basket_dataset import Trip, TripDataset
from choice_learn.datasets.base import download_from_url, get_path

DATA_MODULE = "choice_learn/datasets/data"


def load_bakery(as_frame=False):
    """Load the bakery dataset from uchoice-Bakery-5-25.txt.

    Parameters
    ----------
    as_frame : bool, optional
        Whether to return the dataset as pd.DataFrame. If not, returned as TripDataset,
        by default False."""

    url = "https://drive.usercontent.google.com/u/0/uc?id=1qV8qmiHTq6y5fwgN0_hRXyKreNKrF72E&export=download"
    data_file_name = download_from_url(url)

    archive_path = get_path(data_file_name)

    # We put the extracted files in the data directory
    extract_path = "../../choice_learn/datasets/data/"
    with tarfile.open(archive_path, "r:gz") as tar:
        # Here are the files we are downloading
        file_names = tar.getnames()

        # We extract all the files
        tar.extractall(path=extract_path)

        # We want to read the uchoice-Bakery.txt file (second file in the archive)
        csv_file_to_read = file_names[1]

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
    path = Path(os.path.join("../../", DATA_MODULE)).resolve() / csv_file_to_read
    df = pd.read_csv(path, sep=r"\s+", header=None, names=noms_colonnes)

    if as_frame:
        return df

    n_item = int(df.max().max())

    # Apparently all items are available at each trip
    availability_matrix = np.array([[1] * n_item])

    list_purchases = [[int(item) - 1 for item in row if pd.notna(item)] for row in df.values]

    # Dummy prices, all equal to 1
    prices = np.array([[1] * n_item])
    trips_list = [
        Trip(purchases=purchases, assortment=0, prices=prices) for purchases in list_purchases
    ]

    return TripDataset(trips=trips_list, available_items=availability_matrix)
