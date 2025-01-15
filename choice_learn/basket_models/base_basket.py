"""Datasets loader."""

import os
import sys
from importlib import resources
from typing import Union

import numpy as np
import pandas as pd
from trip_dataset import Trip, TripDataset

sys.path.append("../")

OS_DATA_MODULE = os.path.join(os.path.abspath(".."), "choice_learn", "basket_models", "data")
DATA_MODULE = "choice_learn.basket_models.data"

# When files are executed on a server, the path is different
# OS_DATA_MODULE = os.path.join(os.path.abspath(".."),
#                               "choice-learn", "choice_learn", "basket_models", "data")
# DATA_MODULE = "choice-learn.choice_learn.basket_models.data"


def csv_to_df(
    data_file_name: str, data_module: str = OS_DATA_MODULE, sep: str = ""
) -> pd.DataFrame:
    """Load and return the dataset.

    Parameters
    ----------
    data_file_name: str
        name of the csv file to load
    data_module: str, optional
        path to directory containing the data file, by default DATA_MODULE
    encoding: str, optional
        encoding method of file, by default "utf-8"
    sep: str, optional
        separator used in the csv file, by default ''

    Returns
    -------
    pd.DataFrame
        Loaded dataset
    """
    path = os.path.join(data_module, data_file_name)

    return pd.read_csv(path, sep=sep)


def map_indexes(df: pd.DataFrame, column_name: str, index_start: int) -> dict:
    """Map the values of a column to indexes.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing the column to map
    column_name: str
        Name of the column to map
    index_start: int
        Index to start the mapping from

    Returns
    -------
    dict
        Mapping from values to indexes
    """
    unique_values = df[column_name].unique()
    # Index the items id starting from index_start
    mapping = {value: index + index_start for index, value in enumerate(unique_values)}
    df[column_name] = df[column_name].map(mapping)

    return mapping


def from_csv(
    train_file: str,
    test_file: Union[str, None] = None,
    val_file: Union[str, None] = None,
    nrows: Union[int, None] = None,
    sep: str = None,
    user_id_col: str = "user_id",
    item_id_col: str = "item_id",
    session_id_col: str = "session_id",
    quantity_col: str = "quantity",
    week_id_col: str = "week_id",
    price_col: str = "price",
) -> tuple[TripDataset]:
    """Build a TripDataset from a csv file (with preprocessing).

    The csv file should contain the following columns:
    - user_id
    - item_id
    - session_id
    - quantity
    - week_id
    - price
    (not necessarily with these names).

    Parameters
    ----------
    train_file: str
        Name of the train dataset file
    test_file: str, optional
        Name of the test dataset file, by default None
    val_file: str, optional
        Name of the validation dataset file, by default None
    nrows: int, optional
        Number of rows to load, by default None
    user_id_col: str, optional
        Name of the user id column, by default "user_id"
    item_id_col: str, optional
        Name of the item id column, by default "item_id"
    session_id_col: str, optional
        Name of the session id column, by default "session_id"
    quantity_col: str, optional
        Name of the quantity column, by default "quantity"
    price_col: str, optional
        Name of the price column, by default "price"
    week_id_col: str, optional
        Name of the week id column, by default "week_id"

    Returns
    -------
    TripDataset
        Training TripDataset built from the csv files (with preprocessing)
    TripDataset
        Test TripDataset built from the csv files (with preprocessing)
    TripDataset
        Validation TripDataset built from the csv files (with preprocessing)
    """
    # Load the data and select the first nrows
    print(
        "Number of transactions in the training set: ",
        f"{csv_to_df(data_file_name=train_file, data_module=OS_DATA_MODULE, sep=sep).shape[0]}",
    )
    train_dataset = csv_to_df(data_file_name=train_file, data_module=OS_DATA_MODULE, sep=sep).iloc[
        :nrows
    ]
    print(
        "Number of transactions in the test set: ",
        f"{csv_to_df(data_file_name=test_file, data_module=OS_DATA_MODULE, sep=sep).shape[0]}",
    )
    test_dataset = (
        csv_to_df(data_file_name=test_file, data_module=OS_DATA_MODULE, sep=sep).iloc[:nrows]
        if test_file
        else None
    )
    print(
        "Number of transactions in the validation set: ",
        f"{csv_to_df(data_file_name=val_file, data_module=OS_DATA_MODULE, sep=sep).shape[0]}",
    )
    val_dataset = (
        csv_to_df(data_file_name=val_file, data_module=OS_DATA_MODULE, sep=sep).iloc[:nrows]
        if val_file
        else None
    )

    train_set_grouped_by_item = csv_to_df(
        data_file_name=train_file, data_module=OS_DATA_MODULE, sep=sep
    ).groupby(["item_id"])
    train_set_grouped_by_trip = csv_to_df(
        data_file_name=train_file, data_module=OS_DATA_MODULE, sep=sep
    ).groupby(["session_id", "user_id"])
    print(f"Nb of items in the total train set: {train_set_grouped_by_item.ngroups}")
    print(f"Nb of trips in the total train set: {train_set_grouped_by_trip.ngroups}")
    print(
        "Average number of items per trip in the total train set: ",
        f"{train_set_grouped_by_trip.size().mean()}",
    )
    print(
        "Min and max number of items per trip in the total train set: "
        f"{train_set_grouped_by_trip.size().min()}, {train_set_grouped_by_trip.size().max()}"
    )

    print(f"Before mapping the item ids: {train_dataset['item_id'].unique()=}\n")

    # Rename columns
    for dataset in [train_dataset, test_dataset, val_dataset]:
        dataset = dataset.rename(
            columns={
                user_id_col: "user_id",
                item_id_col: "item_id",
                session_id_col: "session_id",
                quantity_col: "quantity",
                price_col: "price",
                week_id_col: "week_id",
            }
        )

    # Map the indexes
    for dataset in [train_dataset, test_dataset, val_dataset]:
        for column in dataset.columns:
            if column[-3:] == "_id":
                print(f"Remapping {column}")
                if column == "item_id":
                    # 1-index mapping (the checkout item 0 is counted in n_items)
                    map_indexes(dataset, column, index_start=1)
                else:
                    # O-index mapping
                    map_indexes(dataset, column, index_start=0)
    # /!\ TODO: the same ids accross datasets are not necessarily mapped to the same index
    # --> The remapping should be done on the whole dataset after having merged the
    # different subsets?

    print(f"After mapping the item ids: {train_dataset['item_id'].unique()=}\n")

    # Normalize the raw prices (the price for a given trip is divided by the per-item mean price)
    for dataset in [train_dataset, test_dataset, val_dataset]:
        # Drop rows with NaN values in the price column
        dataset = dataset.dropna(subset=["price"])

        # Division by the mean of the prices of the given item in the different trips
        dataset["price"] = dataset["price"] / dataset.groupby("item_id")["price"].transform("mean")
        # Other possibility : division by the mean of the prices of the items in the given trip
        # dataset["price"] = dataset["price"] / dataset.groupby("session_id")["price"].transform(
        #     "mean"
        # )

    # Divide the data into trips
    dataset_trips = [[], [], []]
    for i, dataset in enumerate([train_dataset, test_dataset, val_dataset]):
        # The different datasets can have different values for n_items
        specific_n_items = dataset["item_id"].nunique() + 1  # +1 for the checkout item

        count = 0
        grouped_sessions = list(dataset.groupby("session_id"))
        for trip_idx, (trip_id, trip_data) in enumerate(dataset.groupby("session_id")):
            purchases = trip_data["item_id"].tolist()
            customer = trip_data["user_id"].tolist()
            # All the trips of a given session have the same week_id
            week = trip_data["week_id"].tolist()[0]

            if len(purchases) != len(set(purchases)) and i != 1:
                # Remove duplicates while preserving order
                purchases = list(dict.fromkeys(purchases))

            # Create price array with error handling
            # (Price of checkout item 0: 1 or another default value > 0)
            # (-1 means that the price has not already been set)
            prices = np.array([1] + [-1] * (specific_n_items - 1))

            # 1. Get the price of each item in the trip
            for item_id, session_id in zip(purchases, trip_data["session_id"]):
                try:
                    if isinstance(
                        dataset.set_index(["item_id", "session_id"]).loc[(item_id, session_id)][
                            "price"
                        ],
                        pd.Series,
                    ):
                        # Then the price is a Pandas series (same value repeated)
                        if (
                            (
                                dataset.set_index(["item_id", "session_id"])
                                .loc[(item_id, session_id)]["price"]
                                .to_numpy()[0]
                            )
                            == (
                                dataset.set_index(["item_id", "session_id"])
                                .loc[(item_id, session_id)]["price"]
                                .to_numpy()[0]
                            )
                        ):
                            # Ensure that the price is not NaN
                            # (The price is NaN when there is no item_id in session_id)
                            prices[item_id] = (
                                dataset.set_index(["item_id", "session_id"])
                                .loc[(item_id, session_id)]["price"]
                                .to_numpy()[0]
                            )
                        else:
                            prices[item_id] = 1  # Or another default value > 0
                    else:
                        # Then the price is a scalar
                        if (
                            dataset.set_index(["item_id", "session_id"]).loc[(item_id, session_id)][
                                "price"
                            ]
                            == dataset.set_index(["item_id", "session_id"]).loc[
                                (item_id, session_id)
                            ]["price"]
                        ):
                            # Ensure that the price is not NaN
                            # (The price is NaN when there is no item_id in session_id)
                            prices[item_id] = dataset.set_index(["item_id", "session_id"]).loc[
                                (item_id, session_id)
                            ]["price"]
                        else:
                            prices[item_id] = 1  # Or another default value > 0

                except KeyError:
                    prices[item_id] = 1  # Or another default value > 0

            # 2. Approximate the price of the items not in the trip with
            # the price of the same item in the previous or next trip
            for item_id in range(specific_n_items):
                if prices[item_id] == -1:
                    found_price = False
                    step = 1
                    while not found_price:
                        # Proceed step by step to find the price of the item
                        # in the k-th previous or the k-th next trip
                        prev_session_id, prev_session_data = None, None
                        next_session_id, next_session_data = None, None

                        if trip_idx - step >= 0:
                            prev_session_id, prev_session_data = grouped_sessions[trip_idx - step]
                        if trip_idx + step < len(grouped_sessions):
                            next_session_id, next_session_data = grouped_sessions[trip_idx + step]

                        if (
                            prev_session_data is not None
                            and item_id in prev_session_data["item_id"].tolist()
                        ):
                            # If item_id is in the previous trip, take the
                            # price of the item in the previous trip
                            if isinstance(
                                dataset.set_index(["item_id", "session_id"]).loc[
                                    (item_id, prev_session_id)
                                ]["price"],
                                pd.Series,
                            ):
                                # Then the price is a Pandas series (same value repeated)
                                prices[item_id] = (
                                    dataset.set_index(["item_id", "session_id"])
                                    .loc[(item_id, prev_session_id)]["price"]
                                    .to_numpy()[0]
                                )
                            else:
                                # Then the price is a scalar
                                prices[item_id] = dataset.set_index(["item_id", "session_id"]).loc[
                                    (item_id, prev_session_id)
                                ]["price"]
                            found_price = True

                        elif (
                            next_session_data is not None
                            and item_id in next_session_data["item_id"].tolist()
                        ):
                            # If item_id is in the next session, take the
                            # price of the item in the next trip
                            if isinstance(
                                dataset.set_index(["item_id", "session_id"]).loc[
                                    (item_id, next_session_id)
                                ]["price"],
                                pd.Series,
                            ):
                                # Then the price is a Pandas series (same value repeated)
                                prices[item_id] = (
                                    dataset.set_index(["item_id", "session_id"])
                                    .loc[(item_id, next_session_id)]["price"]
                                    .to_numpy()[0]
                                )
                            else:
                                # Then the price is a scalar
                                prices[item_id] = dataset.set_index(["item_id", "session_id"]).loc[
                                    (item_id, next_session_id)
                                ]["price"]
                            found_price = True

                        if trip_idx - step < 0 and trip_idx + step >= len(grouped_sessions):
                            # Then we have checked all possible previous and next trips
                            break

                        step += 1

                    if not found_price:
                        prices[item_id] = 1  # Or another default value > 0

            for customer_id in customer:
                purchases_customer = trip_data[trip_data["user_id"] == customer_id][
                    "item_id"
                ].tolist()
                dataset_trips[i].append(
                    Trip(
                        id=count,
                        purchases=purchases_customer + [0],  # Add the checkout item 0 at the end
                        customer=customer_id,
                        week=week,
                        prices=prices,
                    )
                )
                count += 1

    # Build the TripDatasets
    train_trip_dataset = TripDataset(trips=dataset_trips[0])
    test_trip_dataset = TripDataset(trips=dataset_trips[1])
    val_trip_dataset = TripDataset(trips=dataset_trips[2])

    n_items = train_dataset["item_id"].nunique() + 1  # +1 for the checkout item
    n_customers = train_dataset["user_id"].nunique()
    n_trips = {
        "train": len(train_trip_dataset),
        "test": len(test_trip_dataset),
        "val": len(val_trip_dataset),
    }

    print(
        "Training dataset:",
        f"n_items={train_dataset['item_id'].nunique()}, ",
        f"n_customers={train_dataset['user_id'].nunique()}",
    )
    print(
        "Validation dataset:",
        f"n_items={val_dataset['item_id'].nunique()}, ",
        f"n_customers={val_dataset['user_id'].nunique()}",
    )
    print(
        "Test dataset:",
        f"n_items={test_dataset['item_id'].nunique()}, ",
        f"n_customers={test_dataset['user_id'].nunique()}",
    )

    return train_trip_dataset, test_trip_dataset, val_trip_dataset, n_items, n_customers, n_trips


def get_path_bis(data_file_name: str, module: str = DATA_MODULE) -> str:
    """Get path toward data file.

    Specifically used to handled Python 3.8 and 3.9+ differences in importlib.resources handling.
    Modified version of https://github.com/artefactory/choice-learn/blob/main/choice_learn/datasets/base.py.

    Parameters
    ----------
    data_file_name: str
        name of the csv file to load
    module: str, optional
        path to directory containing the data file, by default DATA_MODULE

    Returns
    -------
    Path
        path to the data file
    """
    import sys

    if sys.version >= "3.9":
        return resources.files(module) / data_file_name

    with resources.path(module, data_file_name) as path:
        return path
