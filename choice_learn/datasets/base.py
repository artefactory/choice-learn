"""Datasets loader."""
import csv
import gzip
from importlib import resources

import numpy as np

from choice_learn.data.choice_dataset import ChoiceDataset

DATA_MODULE = "choice_learn.datasets.data"


def load_csv(data_file_name, data_module=DATA_MODULE, encoding="utf-8"):
    """Base function to load csv files.

    Parameters
    ----------
    data_file_name : str
        name of the csv file to load
    data_module : str, optional
        path to directory containing the data file, by default DATA_MODULE
    encoding : str, optional
        encoding method of file, by default "utf-8"

    Returns:
    --------
    list
        list of column names
    np.ndarray
        data contained in the csv file
    """
    data_path = resources.files(data_module)
    with (data_path / data_file_name).open("r", encoding=encoding) as csv_file:
        data_file = csv.reader(csv_file)
        names = next(data_file)
        data = []

        for i, ir in enumerate(data_file):
            data.append(np.asarray(ir, dtype=np.float64))
    return names, np.stack(data)


def load_gzip(data_file_name, data_module=DATA_MODULE, encoding="utf-8"):
    """Base function to load zipped .csv.gz files.

    Parameters
    ----------
    data_file_name : str
        name of the csv.gz file to load
    data_module : str, optional
        path to directory containing the data file, by default DATA_MODULE
    encoding : str, optional
        encoding method of file, by default "utf-8"

    Returns:
    --------
    list
        list of column names
    np.ndarray
        data contained in the csv file
    """
    data_path = resources.files(data_module)
    with (data_path / data_file_name).open("rb") as compressed_file:
        compressed_file = gzip.open(compressed_file, mode="rt", encoding=encoding)
        names = next(compressed_file)
        names = names.replace("\n", "")
        data = np.loadtxt(compressed_file, delimiter=",", dtype=int)

    return names.split(","), data


def slice_from_names(array, slice_names, all_names):
    """Slicing on 2nd dimension function for numpy arrays.

    Slices array in the second dimension from column names.

    Parameters
    ----------
    array : np.ndarray
        array to be sliced
    slice_names : list
        names of columns to return
    all_names : list
        names of all columns

    Returns:
    --------
    np.ndarray
        sliced array
    """
    return array[:, [all_names.index(name) for name in slice_names]]


def load_swissmetro(
    one_hot_cat_data=False, add_items_one_hot=False, as_frame=False, return_desc=False
):
    """Load and return the SwissMetro dataset from Bierlaire et al. (2001).

    Parameters
    ----------
    one_hot_cat_data : bool, optional
        Whether to transform categorical data as OneHot, by default False
    add_items_one_hot : bool, optional
        Whether to add a OneHot encoding of items as items_features, by default False
    as_frame : bool, optional
        Whether to return the dataset as pd.DataFrame. If not, returned as ChoiceDataset,
        by default False
    return_desc : bool, optional
        Whether to return the description, by default False

    Returns:
    --------
    ChoiceDataset
        Loaded SwissMetro dataset
    """
    data_file_name = "swissmetro.csv.gz"
    names, data = load_gzip(data_file_name)
    items = ["TRAIN", "SM", "CAR"]
    items_features = []
    session_features = [
        "GROUP",
        "PURPOSE",
        "FIRST",
        "TICKET",
        "WHO",
        "LUGGAGE",
        "AGE",
        "MALE",
        "INCOME",
        "GA",
        "ORIGIN",
        "DEST",
    ]
    sessions_items_features = ["TT", "CO", "HE"]
    sessions_items_features = [
        [f"{item}_{feature}" for feature in sessions_items_features] for item in items
    ]
    sessions_items_availabilities = ["TRAIN_AV", "SM_AV", "CAR_AV"]
    choice_column = "CHOICE"

    if add_items_one_hot:
        items_features = np.eye(len(items), dtype=np.float64)
    else:
        items_features = None

    # Adding dummy CAR_HE feature as 0 for consistency
    names.append("CAR_HE")
    data = np.hstack([data, np.zeros((data.shape[0], 1))])

    session_features = slice_from_names(data, session_features, names)
    sessions_items_features = np.stack(
        [slice_from_names(data, features, names) for features in sessions_items_features], axis=-1
    )
    sessions_items_availabilities = slice_from_names(data, sessions_items_availabilities, names)
    choices = data[:, names.index(choice_column)]

    # Remove no choice
    choice_done = np.where(choices > 0)[0]
    session_features = session_features[choice_done]
    sessions_items_features = sessions_items_features[choice_done]
    sessions_items_availabilities = sessions_items_availabilities[choice_done]
    choices = choices[choice_done]

    # choices renormalization
    choices = choices - 1

    if return_desc:
        # TODO
        pass
    if one_hot_cat_data:
        # TODO
        pass
    if as_frame:
        # TODO
        pass

    return ChoiceDataset(
        items_features=items_features,
        sessions_features=session_features,
        sessions_items_features=sessions_items_features,
        sessions_items_availabilities=sessions_items_availabilities,
        choices=choices,
    )


def load_modecanada():
    """_summary_."""
    pass
