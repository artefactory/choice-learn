"""Datasets loader."""

import os
import sys
from typing import Union

import numpy as np
import pandas as pd

from pathlib import Path

from .basket_dataset import Trip, TripDataset

OS_DATA_MODULE = os.path.join(os.path.abspath(".."), "choice_learn", "datasets", "data")
DATA_MODULE = "../choice_learn/datasets/data"

# When files are executed on a server, the path is different
sys.path.append("../")
# OS_DATA_MODULE = os.path.join(os.path.abspath(".."),
#                               "choice-learn", "choice_learn", "datasets", "data")
# DATA_MODULE = "../choice-learn/choice_learn/datasets/data"


def csv_to_df(
    data_file_name: str, data_module: str = OS_DATA_MODULE, sep: str = ""
) -> pd.DataFrame:
    """Load and return the dataset.

    Parameters
    ----------
    data_file_name: str
        Name of the csv file to load
    data_module: str, optional
        Path to directory containing the data file, by default DATA_MODULE
    encoding: str, optional
        Encoding method of file, by default "utf-8"
    sep: str, optional
        Separator used in the csv file, by default ''

    Returns
    -------
    pd.DataFrame
        Loaded dataset
    """
    path = os.path.join(data_module, data_file_name)

    return pd.read_csv(path, sep=sep)


def map_indexes(df: pd.DataFrame, column_name: str, index_start: int) -> dict:
    """Create the mapping and map the values of a column to indexes.

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
    data_file_name: str,
    nrows: Union[int, None] = None,
    sep: str = None,
    store_id_col: str = "store_id",
    item_id_col: str = "item_id",
    session_id_col: str = "session_id",
    quantity_col: str = "quantity",
    week_id_col: str = "week_id",
    price_col: str = "price",
) -> tuple[TripDataset]:
    """Build a TripDataset from a csv file (with preprocessing).

    The csv file should contain the following columns:
    - store_id
    - item_id
    - session_id
    - quantity
    - week_id
    - price
    (not necessarily with these names).

    Parameters
    ----------
    data_file_name: str
        Name of the csv file to load
    nrows: int, optional
        Number of rows to load, by default None
    store_id_col: str, optional
        Name of the store id column, by default "store_id"
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
    trip_dataset: TripDataset
        TripDataset built from the csv files (with preprocessing)
    n_items: int
        Number of distinct items in the dataset
    n_stores: int
        Number of distinct stores in the dataset
    n_trips: int
        Number of distinct trips in the dataset

    trip_dataset, n_items, n_stores, n_trips
    """
    # Load the data and select the first nrows
    dataset = csv_to_df(data_file_name=data_file_name, data_module=OS_DATA_MODULE, sep=sep)
    print(
        "Number of transactions in the total dataset: ",
        f"{dataset.shape[0]}",
    )
    dataset = dataset.iloc[:nrows]

    # Print some statistics about the dataset
    dataset_grouped_by_item = csv_to_df(
        data_file_name=data_file_name, data_module=OS_DATA_MODULE, sep=sep
    ).groupby(["item_id"])
    dataset_grouped_by_trip = csv_to_df(
        data_file_name=data_file_name, data_module=OS_DATA_MODULE, sep=sep
    ).groupby(["session_id", "store_id"])
    print(f"Nb of items in the total dataset: {dataset_grouped_by_item.ngroups}")
    print(f"Nb of trips in the total dataset: {dataset_grouped_by_trip.ngroups}")
    print(
        "Average number of items per trip in the total dataset: ",
        f"{dataset_grouped_by_trip.size().mean()}",
    )
    print(
        "Min and max number of items per trip in the total dataset: "
        f"{dataset_grouped_by_trip.size().min()}, {dataset_grouped_by_trip.size().max()}"
    )

    # Rename columns
    dataset = dataset.rename(
        columns={
            store_id_col: "store_id",
            item_id_col: "item_id",
            session_id_col: "session_id",
            quantity_col: "quantity",
            price_col: "price",
            week_id_col: "week_id",
        }
    )

    print(f"Before mapping the item ids: {dataset['item_id'].unique()=}\n")

    # Map the indexes
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

    print(f"After mapping the item ids: {dataset['item_id'].unique()=}\n")

    # Normalize the raw prices (the price for a given trip is divided by the per-item mean price)
    # Drop rows with NaN values in the price column
    dataset = dataset.dropna(subset=["price"])

    # Division by the mean of the prices of the given item in the different trips
    dataset["price"] = dataset["price"] / dataset.groupby("item_id")["price"].transform("mean")
    # Other possibility : division by the mean of the prices of the items in the given trip
    # dataset["price"] = dataset["price"] / dataset.groupby("session_id")["price"].transform(
    #     "mean"
    # )

    n_items = dataset["item_id"].nunique() + 1  # +1 for the checkout item
    n_stores = dataset["store_id"].nunique()

    # Divide the data into trips
    dataset_trips = []

    grouped_sessions = list(dataset.groupby("session_id"))
    for trip_idx, (trip_id, trip_data) in enumerate(dataset.groupby("session_id")):
        purchases = trip_data["item_id"].tolist()
        store = trip_data["store_id"].tolist()
        # All the trips of a given session have the same week_id
        week = trip_data["week_id"].tolist()[0]

        if len(purchases) != len(set(purchases)):
            # Remove duplicates while preserving order
            purchases = list(dict.fromkeys(purchases))

        # Create price array with error handling
        # (Price of checkout item 0: 1 or another default value > 0)
        # (-1 means that the price has not already been set)
        prices = np.array([1] + [-1] * (n_items - 1))

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
                        == dataset.set_index(["item_id", "session_id"]).loc[(item_id, session_id)][
                            "price"
                        ]
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
        for item_id in range(n_items):
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

        for store_id in store:
            purchases_store = trip_data[trip_data["store_id"] == store_id]["item_id"].tolist()
            dataset_trips.append(
                Trip(
                    purchases=purchases_store + [0],  # Add the checkout item 0 at the end
                    store=store_id,
                    week=week,
                    prices=prices,
                    assortment=0,  # TODO: Add the assortment
                )
            )

    # Build the TripDatasets
    assortments = np.expand_dims(np.ones(n_items), axis=0)  # TODO: Add the assortments
    trip_dataset = TripDataset(trips=dataset_trips, assortments=assortments)

    n_trips = len(trip_dataset)

    print(f"{n_items=}, {n_stores=} and {n_trips=}")

    return trip_dataset, n_items, n_stores, n_trips




def load_bakery(as_frame=False):
    """Load the bakery dataset from uchoice-Bakery-5-25.txt.

    Parameters
    ----------
    as_frame : bool, optional
        Whether to return the dataset as pd.DataFrame. If not, returned as TripDataset,
        by default False."""
    
    noms_colonnes = ['article_1', 'article_2', 'article_3', 'article_4', 'article_5']

     # likewise get_path function
    path = Path(os.path.join("..", DATA_MODULE)).resolve() / 'uchoice-Bakery/uchoice-Bakery-5-25.txt'

    df = pd.read_csv(path, sep='\s+', header=None, names=noms_colonnes)

    if as_frame :
        return df
    
    num_purchases = df.shape[0]
    n_item = int(df.max().max())
    
    # Apparently all items are available at each trip 
    availability_matrix = np.ones((num_purchases, n_item), dtype=int) 
    
    list_purchases = [[int(item) for item in row if pd.notna(item)] for row in df.values]
    prices = [1]*n_item
    trips_list = [Trip(purchases=purchases, assortment=0, prices =prices) for purchases in list_purchases]

    return TripDataset(trips = trips_list, available_items = availability_matrix)
