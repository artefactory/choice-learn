"""ICDM 2013 Expedia dataset."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from choice_learn.data.choice_dataset import ChoiceDataset
from choice_learn.data.storage import OneHotStorage
from choice_learn.datasets.base import get_path

DATA_MODULE = "choice_learn.datasets.data"


def load_expedia(as_frame=False, preprocessing="rumnet"):
    """Load the Expedia dataset."""
    filename = "expedia.csv"
    data_path = get_path(filename, module=DATA_MODULE)
    if not Path.exists((data_path / filename)):
        print("In order to use the Expedia dataset, please download it from:")
        print("https://www.kaggle.com/c/expedia-personalized-sort")
        print("and save it in the following location:")
        print(os.path.join(DATA_MODULE, filename))
        print("The downloaded train.csv file should be named 'expedia.csv'")
        raise FileNotFoundError(
            f"File {filename} not found in {os.path.join(DATA_MODULE, filename)}"
        )

    expedia_df = pd.read_csv((data_path / filename))
    if as_frame:
        return expedia_df

    if preprocessing == "rumnet":
        expedia_df.date_time = pd.to_datetime(expedia_df.date_time, format="%Y-%m-%d %H:%M:%S")
        expedia_df.loc[:, "day_of_week"] = expedia_df.loc[:, "date_time"].dt.dayofweek
        expedia_df.loc[:, "month"] = expedia_df.loc[:, "date_time"].dt.month
        expedia_df.loc[:, "hour"] = expedia_df.loc[:, "date_time"].dt.hour

        for id_col in [
            "site_id",
            "visitor_location_country_id",
            "prop_country_id",
            "srch_destination_id",
        ]:
            value_counts = expedia_df[["srch_id", id_col]].drop_duplicates()[id_col].value_counts()
            kept_ids = value_counts.index[value_counts.gt(1000)]
            for id_ in expedia_df[id_col].unique():
                if id_ not in kept_ids:
                    expedia_df.loc[expedia_df[id_col] == id_, id_col] = -1

        # Filtering
        expedia_df = expedia_df[expedia_df.price_usd <= 1000]
        expedia_df = expedia_df[expedia_df.price_usd >= 10]
        expedia_df["log_price"] = expedia_df.price_usd.apply(np.log)
        expedia_df = expedia_df[expedia_df.srch_length_of_stay <= 14]
        expedia_df = expedia_df[expedia_df.srch_booking_window <= 365]
        expedia_df["booking_window"] = np.log(expedia_df["srch_booking_window"] + 1)
        expedia_df = expedia_df.fillna(-1)

        order_cols = [
            "srch_id",
            "prop_id",
            "prop_starrating",
            "prop_review_score",
            "prop_brand_bool",
            "prop_location_score1",
            "prop_location_score2",
            "prop_log_historical_price",
            "position",
            "promotion_flag",
            "srch_length_of_stay",
            "srch_adults_count",
            "srch_children_count",
            "srch_room_count",
            "srch_saturday_night_bool",
            "orig_destination_distance",
            "random_bool",
            "day_of_week",
            "month",
            "hour",
            "log_price",
            "booking_window",
            "site_id",
            "visitor_location_country_id",
            "prop_country_id",
            "srch_destination_id",
            "click_bool",
            "booking_bool",
        ]
        expedia_df = expedia_df[order_cols]

        expedia_df["av"] = 1
        asst_size = 38  # Fixed number of items in the assortment

        # Loop to fill the data frame with dummy products
        # next loop creates the dummy products
        for _ in range(asst_size):
            dum = (
                expedia_df.groupby("srch_id")
                .filter(lambda x: len(x) < asst_size)
                .groupby("srch_id")
                .max()
                .reset_index(drop=False)
            )
            dum.loc[:, "booking_bool"] = 0
            dum.loc[:, "av"] = 0
            expedia_df = pd.concat([expedia_df, dum])

        # getting rid of search & prop_id and the clickbool and bookingbool
        # adding no_purchase fixed effect
        expedia_df["is_no_purchase"] = 0

        # adding the no_purchase option to the data
        df1 = (
            expedia_df.groupby("srch_id")
            .filter(lambda x: x.booking_bool.sum() == 1)
            .groupby("srch_id")
            .max()
            .reset_index(drop=False)
        )
        df1.loc[:, "is_no_purchase"] = 1
        df1.loc[:, "log_price"] = 0
        df1.loc[:, "booking_bool"] = 0

        df2 = (
            expedia_df.groupby("srch_id")
            .filter(lambda x: x.booking_bool.sum() == 0)
            .groupby("srch_id")
            .max()
            .reset_index(drop=False)
        )
        df2.loc[:, "is_no_purchase"] = 1
        df2.loc[:, "log_price"] = 0
        df2.loc[:, "booking_bool"] = 1
        expedia_df = pd.concat([expedia_df, df1, df2])

        expedia_df = expedia_df.sort_values("srch_id")
        choices = expedia_df.groupby("srch_id").apply(lambda x: x.booking_bool.argmax())

        fixed_items_features = None
        site_id_storage = OneHotStorage(ids=expedia_df.site_id.unique(), name="site_id")
        visitor_location_country_id_storage = OneHotStorage(
            ids=expedia_df.visitor_location_country_id.unique(), name="visitor_location_country_id"
        )
        srch_destination_id_storage = OneHotStorage(
            ids=expedia_df.srch_destination_id.unique(), name="srch_destination_id"
        )
        prop_country_id_storage = OneHotStorage(
            ids=expedia_df.prop_country_id.unique(), name="prop_country_id"
        )

        contexts_features_names = [
            "srch_id",
            "srch_length_of_stay",
            "srch_adults_count",
            "srch_children_count",
            "srch_room_count",
            "srch_saturday_night_bool",
            "booking_window",
            "random_bool",
            "day_of_week",
            "month",
            "hour",
            "site_id",
            "visitor_location_country_id",
            "srch_destination_id",
        ]

        contexts_features = expedia_df[contexts_features_names].drop_duplicates()
        contexts_features = contexts_features.set_index("srch_id")
        contexts_features = contexts_features.to_numpy()

        contexts_items_features_names = [
            "srch_id",
            "prop_starrating",
            "prop_review_score",
            "prop_brand_bool",
            "prop_location_score1",
            "prop_location_score2",
            "prop_log_historical_price",
            "position",
            "promotion_flag",
            "orig_destination_distance",
            "log_price",
            "prop_country_id",
        ]

        contexts_items_features = (
            expedia_df[contexts_items_features_names]
            .groupby("srch_id")
            .apply(lambda x: x[contexts_items_features_names[1:]].to_numpy())
        )
        contexts_items_availabilities = (
            expedia_df[["srch_id", "av"]].groupby("srch_id").apply(lambda x: x["av"].to_numpy())
        )
        contexts_items_features = np.stack(contexts_items_features)

        return ChoiceDataset(
            fixed_items_features=fixed_items_features,
            contexts_features=contexts_features,
            contexts_items_features=contexts_items_features,
            features_by_ids=[
                site_id_storage,
                visitor_location_country_id_storage,
                srch_destination_id_storage,
                prop_country_id_storage,
            ],
            choices=choices.to_numpy(),
            contexts_features_names=contexts_features_names[1:],
            contexts_items_features_names=contexts_items_features_names[1:],
            contexts_items_availabilities=np.stack(contexts_items_availabilities.to_numpy()),
        )

    raise ValueError(
        f"Preprocessing {preprocessing} not recognized, only 'rumnet' currently available"
    )
