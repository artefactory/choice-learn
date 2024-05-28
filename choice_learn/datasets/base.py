"""Datasets loader."""
import csv
import gzip
import os
from importlib import resources

import numpy as np
import pandas as pd

from choice_learn.data.choice_dataset import ChoiceDataset

OS_DATA_MODULE = os.path.join(os.path.abspath(".."), "choice_learn", "datasets", "data")
DATA_MODULE = "choice_learn.datasets.data"


def get_path(data_file_name, module=DATA_MODULE):
    """Get path toward data file.

    Specifically used to handled Python 3.8 and 3.9+ differences in importlib.resources handling.

    Parameters
    ----------
    module : str, optional
        path to directory containing the data file, by default DATA_MODULE
    data_file_name : str
        name of the csv file to load

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


def load_csv(data_file_name, data_module=OS_DATA_MODULE, encoding="utf-8"):
    """Load csv files.

    Parameters
    ----------
    data_file_name : str
        name of the csv file to load
    data_module : str, optional
        path to directory containing the data file, by default DATA_MODULE
    encoding : str, optional
        encoding method of file, by default "utf-8"

    Returns
    -------
    list
        list of column names
    np.ndarray
        data contained in the csv file
    """
    with open(os.path.join(data_module, data_file_name), "r", encoding=encoding) as csv_file:
        data_file = csv.reader(csv_file)
        names = next(data_file)
        data = []

        for i, ir in enumerate(data_file):
            data.append(np.asarray(ir, dtype=np.float64))
    return names, np.stack(data)


def load_gzip(data_file_name, data_module=OS_DATA_MODULE, encoding="utf-8"):
    """Load zipped .csv.gz files.

    Parameters
    ----------
    data_file_name : str
        name of the csv.gz file to load
    data_module : str, optional
        path to directory containing the data file, by default DATA_MODULE
    encoding : str, optional
        encoding method of file, by default "utf-8"

    Returns
    -------
    list
        list of column names
    np.ndarray
        data contained in the csv file
    """
    with open(os.path.join(data_module, data_file_name), "rb") as compressed_file:
        compressed_file = gzip.open(compressed_file, mode="rt", encoding=encoding)
        names = next(compressed_file)
        names = names.replace("\n", "")
        data = np.loadtxt(compressed_file, delimiter=",", dtype=object)

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

    Returns
    -------
    np.ndarray
        sliced array
    """
    return array[:, [all_names.index(name) for name in slice_names]]


def load_swissmetro(add_items_one_hot=False, as_frame=False, return_desc=False, preprocessing=None):
    """Load and return the SwissMetro dataset from Bierlaire et al. (2001).

    Parameters
    ----------
    add_items_one_hot : bool, optional
        Whether to add a OneHot encoding of items as items_features, by default False
    as_frame : bool, optional
        Whether to return the dataset as pd.DataFrame. If not, returned as ChoiceDataset,
        by default False
    return_desc : bool, optional
        Whether to return the description, by default False
    preprocessing : str, optional
        Preprocessing to apply to the dataset, by default None

    Returns
    -------
    ChoiceDataset
        Loaded SwissMetro dataset
    """
    description = """This dataset consists of survey data collected on the trains between St.Gallen
     and Geneva, Switzerland, during March 1998. The respondents provided information in order to
     analyze the impact of the modal innovation intransportation, represented by the Swissmetro,
     a revolutionary mag-lev under ground system, against the usual transport modes represented by
     car and train.

    Bierlaire, M., Axhausen, K. and Abay, G. (2001), The acceptance of modal innovation:
    The case of Swissmetro, in ‘Proceedings of the Swiss Transport Research Conference’,
    Ascona, Switzerland."""

    data_file_name = "swissmetro.csv.gz"
    full_path = get_path(data_file_name, module=DATA_MODULE)
    swiss_df = pd.read_csv(full_path)
    swiss_df["CAR_HE"] = 0.0

    items = ["TRAIN", "SM", "CAR"]
    shared_features_by_choice_names = [
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
    items_features_by_choice_names = ["CO", "TT", "HE", "SEATS"]
    choice_column = "CHOICE"
    availabilities_column = "AV"

    if add_items_one_hot:
        items_features_by_choice_names += [f"oh_{item}" for item in items]
        for item in items:
            for item2 in items:
                if item == item2:
                    swiss_df[f"{item}_oh_{item}"] = 1
                else:
                    swiss_df[f"{item2}_oh_{item}"] = 0

    if return_desc:
        return description

    swiss_df = swiss_df.loc[swiss_df.CHOICE != 0]
    swiss_df.CHOICE = swiss_df.CHOICE - 1

    if preprocessing == "long_format":
        long = []
        for n_row, row in swiss_df.iterrows():
            df_dict = {
                "PURPOSE": [],
                "AGE": [],
                "item_id": [],
                "TT": [],
                "CO": [],
                "CHOICE": [],
                "choice_id": [],
            }

            for item_index, item_id in enumerate(["TRAIN", "SM", "CAR"]):
                if row[f"{item_id}_AV"] > 0:
                    if item_index == row.CHOICE:
                        df_dict["CHOICE"].append(1)
                    else:
                        df_dict["CHOICE"].append(0)

                    df_dict["item_id"].append(item_id)
                    df_dict["TT"].append(row[f"{item_id}_TT"])
                    df_dict["CO"].append(row[f"{item_id}_CO"])

                    df_dict["PURPOSE"].append(row["PURPOSE"])
                    df_dict["AGE"].append(row["AGE"])
                    df_dict["choice_id"].append(n_row)
            long.append(pd.DataFrame(df_dict))
        swiss_df = pd.concat(long, axis=0)
        swiss_df = swiss_df.reset_index(drop=True)
        as_frame = True
    if as_frame:
        return swiss_df

    if preprocessing == "tastenet":
        swiss_df = swiss_df.loc[swiss_df.AGE != 6]
        swiss_df["TRAIN_ASC_TRAIN"] = 1.0
        swiss_df["SM_ASC_TRAIN"] = 0.0
        swiss_df["CAR_ASC_TRAIN"] = 0.0

        swiss_df["TRAIN_ASC_SM"] = 0.0
        swiss_df["SM_ASC_SM"] = 1.0
        swiss_df["CAR_ASC_SM"] = 0.0

        swiss_df["TRAIN_ASC_CAR"] = 0.0
        swiss_df["SM_ASC_CAR"] = 0.0
        swiss_df["CAR_ASC_CAR"] = 1.0

        swiss_df["FEMALE"] = 1 - swiss_df["MALE"]
        shared_features_by_choice_names = ["MALE", "FEMALE"]
        swiss_df["NOT_FIRST"] = 1 - swiss_df["FIRST"]
        shared_features_by_choice_names += ["FIRST", "NOT_FIRST"]
        swiss_df["NOT_GA"] = 1 - swiss_df["GA"]
        shared_features_by_choice_names += ["GA", "NOT_GA"]

        age_dummy = pd.get_dummies(swiss_df.AGE, prefix="AGE").astype(int)
        swiss_df = pd.concat([swiss_df, age_dummy], axis=1)
        shared_features_by_choice_names += age_dummy.columns.to_list()

        swiss_df.INCOME = swiss_df.apply(lambda row: 1 if row.INCOME == 0 else row.INCOME, axis=1)
        income_dummy = pd.get_dummies(swiss_df.INCOME, prefix="INCOME").astype(int)
        swiss_df = pd.concat([swiss_df, income_dummy], axis=1)
        shared_features_by_choice_names += income_dummy.columns.to_list()

        swiss_df.WHO = swiss_df.apply(lambda row: 1 if row.WHO == 0 else row.WHO, axis=1)
        who_dummy = pd.get_dummies(swiss_df.WHO, prefix="WHO").astype(int)
        swiss_df = pd.concat([swiss_df, who_dummy], axis=1)
        shared_features_by_choice_names += who_dummy.columns.to_list()

        swiss_df = swiss_df.loc[swiss_df.PURPOSE != 9]
        purpose_dict = {
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 1,
            6: 2,
            7: 3,
            8: 4,
        }
        swiss_df.PURPOSE = swiss_df.apply(lambda row: purpose_dict[row.PURPOSE], axis=1)
        purpose_dummy = pd.get_dummies(swiss_df.PURPOSE, prefix="PURPOSE").astype(int)
        swiss_df = pd.concat([swiss_df, purpose_dummy], axis=1)
        shared_features_by_choice_names += purpose_dummy.columns.to_list()

        luggage_dummy = pd.get_dummies(swiss_df.LUGGAGE, prefix="LUGGAGE").astype(int)
        swiss_df = pd.concat([swiss_df, luggage_dummy], axis=1)
        shared_features_by_choice_names += luggage_dummy.columns.to_list()

        swiss_df["SM_CO"] = swiss_df["SM_CO"] * (swiss_df["GA"] == 0)
        swiss_df["TRAIN_CO"] = swiss_df["TRAIN_CO"] * (swiss_df["GA"] == 0)

        for col in [
            "TRAIN_TT",
            "TRAIN_HE",
            "TRAIN_CO",
            "SM_TT",
            "SM_HE",
            "SM_CO",
            "CAR_TT",
            "CAR_CO",
        ]:
            swiss_df[col] = swiss_df[col] / 100

        return swiss_df.ID.to_numpy(), ChoiceDataset.from_single_wide_df(
            df=swiss_df,
            items_id=items,
            shared_features_columns=shared_features_by_choice_names,
            items_features_suffixes=items_features_by_choice_names
            + ["ASC_TRAIN", "ASC_SM", "ASC_CAR"],
            available_items_suffix=availabilities_column,
            choices_column=choice_column,
            choice_format="item_index",
        )

    if preprocessing == "tutorial":
        # swiss_df = pd.DataFrame(data, columns=names)
        # Removing unknown choices
        # Keep only commute an dbusiness trips
        swiss_df = swiss_df.loc[swiss_df.PURPOSE.isin([1, 3])]

        # Normalizing values
        swiss_df[["TRAIN_TT", "SM_TT", "CAR_TT"]] = swiss_df[["TRAIN_TT", "SM_TT", "CAR_TT"]] / 60.0
        swiss_df[["TRAIN_HE", "SM_HE"]] = swiss_df[["TRAIN_HE", "SM_HE"]] / 60.0

        swiss_df["train_free_ticket"] = swiss_df.apply(
            lambda row: ((row["GA"] == 1 or row["WHO"] == 2) > 0).astype(int), axis=1
        )
        swiss_df["sm_free_ticket"] = swiss_df.apply(
            lambda row: ((row["GA"] == 1 or row["WHO"] == 2) > 0).astype(int), axis=1
        )
        swiss_df["car_free_ticket"] = 0

        swiss_df["train_travel_cost"] = swiss_df.apply(
            lambda row: (row["TRAIN_CO"] * (1 - row["train_free_ticket"])) / 100, axis=1
        )
        swiss_df["sm_travel_cost"] = swiss_df.apply(
            lambda row: (row["SM_CO"] * (1 - row["sm_free_ticket"])) / 100, axis=1
        )
        swiss_df["car_travel_cost"] = swiss_df.apply(lambda row: row["CAR_CO"] / 100, axis=1)

        swiss_df["single_luggage_piece"] = swiss_df.apply(
            lambda row: (row["LUGGAGE"] == 1).astype(int), axis=1
        )
        swiss_df["multiple_luggage_piece"] = swiss_df.apply(
            lambda row: (row["LUGGAGE"] == 3).astype(int), axis=1
        )
        swiss_df["regular_class"] = swiss_df.apply(lambda row: 1 - row["FIRST"], axis=1)
        swiss_df["train_survey"] = swiss_df.apply(lambda row: 1 - row["SURVEY"], axis=1)

        shared_features_by_choice = swiss_df[
            ["train_survey", "regular_class", "single_luggage_piece", "multiple_luggage_piece"]
        ].to_numpy()
        train_features = swiss_df[["train_travel_cost", "TRAIN_TT", "TRAIN_HE"]].to_numpy()
        sm_features = swiss_df[["sm_travel_cost", "SM_TT", "SM_HE", "SM_SEATS"]].to_numpy()
        car_features = swiss_df[["car_travel_cost", "CAR_TT"]].to_numpy()

        # We need to have the same number of features for each item, we create dummy ones:
        car_features = np.concatenate([car_features, np.zeros((len(car_features), 2))], axis=1)
        train_features = np.concatenate(
            [train_features, np.zeros((len(train_features), 1))], axis=1
        )
        items_features_by_choice = np.stack([train_features, sm_features, car_features], axis=1)

        available_items_by_choice = swiss_df[["TRAIN_AV", "SM_AV", "CAR_AV"]].to_numpy()
        # Re-Indexing choices from 1 to 3 to 0 to 2
        choices = swiss_df.CHOICE.to_numpy()

        return ChoiceDataset(
            shared_features_by_choice=shared_features_by_choice,
            items_features_by_choice=items_features_by_choice,
            available_items_by_choice=available_items_by_choice,
            shared_features_by_choice_names=[
                "train_survey",
                "regular_class",
                "single_luggage_piece",
                "multiple_luggage_piece",
            ],
            items_features_by_choice_names=["cost", "travel_time", "headway", "seats"],
            choices=choices,
        )
    if preprocessing == "biogeme_nested":
        # Keep only commute an dbusiness trips
        swiss_df = swiss_df.loc[swiss_df.PURPOSE.isin([1, 3])]

        # Normalizing values by 100
        swiss_df[["TRAIN_TT", "SM_TT", "CAR_TT"]] = (
            swiss_df[["TRAIN_TT", "SM_TT", "CAR_TT"]] / 100.0
        )

        swiss_df["train_free_ticket"] = swiss_df.apply(
            lambda row: (row["GA"] == 1).astype(int), axis=1
        )
        swiss_df["sm_free_ticket"] = swiss_df.apply(
            lambda row: (row["GA"] == 1).astype(int), axis=1
        )

        swiss_df["train_travel_cost"] = swiss_df.apply(
            lambda row: (row["TRAIN_CO"] * (1 - row["train_free_ticket"])) / 100, axis=1
        )
        swiss_df["sm_travel_cost"] = swiss_df.apply(
            lambda row: (row["SM_CO"] * (1 - row["sm_free_ticket"])) / 100, axis=1
        )
        swiss_df["car_travel_cost"] = swiss_df.apply(lambda row: row["CAR_CO"] / 100, axis=1)

        train_features = swiss_df[["train_travel_cost", "TRAIN_TT"]].to_numpy()
        sm_features = swiss_df[["sm_travel_cost", "SM_TT"]].to_numpy()
        car_features = swiss_df[["car_travel_cost", "CAR_TT"]].to_numpy()

        items_features_by_choice = np.stack([train_features, sm_features, car_features], axis=1)

        available_items_by_choice = swiss_df[["TRAIN_AV", "SM_AV", "CAR_AV"]].to_numpy()
        # Re-Indexing choices from 1 to 3 to 0 to 2
        choices = swiss_df.CHOICE.to_numpy()

        return ChoiceDataset(
            shared_features_by_choice=None,
            items_features_by_choice=items_features_by_choice,
            available_items_by_choice=available_items_by_choice,
            shared_features_by_choice_names=None,
            items_features_by_choice_names=["cost", "travel_time"],
            choices=choices,
        )
    if preprocessing == "rumnet":
        swiss_df["One"] = 1.0
        swiss_df["Zero"] = 0.0

        available_items_by_choice = swiss_df[["TRAIN_AV", "SM_AV", "CAR_AV"]].to_numpy()
        items_features_by_choice = np.stack(
            [
                swiss_df[["One", "Zero", "Zero", "TRAIN_TT", "TRAIN_CO", "TRAIN_HE"]].to_numpy(),
                swiss_df[["Zero", "One", "Zero", "SM_TT", "SM_CO", "SM_HE"]].to_numpy(),
                swiss_df[["Zero", "Zero", "One", "CAR_TT", "CAR_CO", "CAR_HE"]].to_numpy(),
            ],
            axis=1,
        )

        items_features_by_choice[:, :, 0] = items_features_by_choice[:, :, 0] / 1000
        items_features_by_choice[:, :, 1] = items_features_by_choice[:, :, 1] / 5000
        items_features_by_choice[:, :, 2] = items_features_by_choice[:, :, 2] / 100

        long_data = pd.get_dummies(
            swiss_df,
            columns=[
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
            ],
            drop_first=False,
        )

        # Transorming the category data into OneHot
        shared_features_by_choice = []
        for col in long_data.columns:
            if col.startswith("GROUP"):
                shared_features_by_choice.append(col)
            if col.startswith("PURPOSE"):
                shared_features_by_choice.append(col)
            if col.startswith("FIRST"):
                shared_features_by_choice.append(col)
            if col.startswith("TICKET"):
                shared_features_by_choice.append(col)
            if col.startswith("WHO"):
                shared_features_by_choice.append(col)
            if col.startswith("LUGGAGE"):
                shared_features_by_choice.append(col)
            if col.startswith("AGE"):
                shared_features_by_choice.append(col)
            if col.startswith("MALE"):
                shared_features_by_choice.append(col)
            if col.startswith("INCOME"):
                shared_features_by_choice.append(col)
            if col.startswith("GA"):
                shared_features_by_choice.append(col)
            if col.startswith("ORIGIN"):
                shared_features_by_choice.append(col)
            if col.startswith("DEST"):
                shared_features_by_choice.append(col)

        shared_features_by_choice = long_data[shared_features_by_choice].to_numpy()
        choices = swiss_df.CHOICE.to_numpy()

        return ChoiceDataset(
            shared_features_by_choice=shared_features_by_choice.astype("float32"),
            items_features_by_choice=items_features_by_choice.astype("float32"),
            available_items_by_choice=available_items_by_choice,
            choices=choices,
        )

    return ChoiceDataset.from_single_wide_df(
        df=swiss_df,
        items_id=items,
        shared_features_columns=shared_features_by_choice_names,
        items_features_suffixes=items_features_by_choice_names,
        available_items_suffix=availabilities_column,
        choices_column=choice_column,
        choice_format="item_index",
    )


def load_modecanada(
    add_items_one_hot=False,
    add_is_public=False,
    as_frame=False,
    return_desc=False,
    choice_format="one_zero",
    split_features=False,
    to_wide=False,
    preprocessing=None,
):
    """Load and return the ModeCanada dataset from Koppleman et al. (1993).

    Parameters
    ----------
    one_hot_cat_data : bool, optional
        Whether to transform categorical data as OneHot, by default False.
    add_is_public : bool, optional
        Whether to add the is_public feature, by default False.
    add_items_one_hot : bool, optional
        Whether to add a OneHot encoding of items as items_features, by default False
    as_frame : bool, optional
        Whether to return the dataset as pd.DataFrame. If not, returned as ChoiceDataset,
        by default False.
    return_desc : bool, optional
        Whether to return the description, by default False.
    choice_format : str, optional, among ["one_zero", "items_id"]
        format indicating how the choice is encoded, by default "one_zero".
    split_features : bool, optional
        Whether to split features by type in different dataframes, by default False.
    to_wide : bool, optional
        Whether to return the dataset in wide format,
        by default False (an thus returned in long format).
    preprocessing : str, optional
        Preprocessing to apply to the dataset, by default None

    Returns
    -------
    ChoiceDataset
        Loaded ModeCanada dataset
    """
    desc = """The dataset was assembled in 1989 by VIA Rail (the Canadian national rail carrier) to
     estimate the demand for high-speed rail in the Toronto-Montreal corridor. The main information
     source was a Passenger Review administered to business travelers augmented by information about
     each trip. The observations consist of a choice between four modes of transportation (train,
     air, bus, car) with information about the travel mode and about the passenger. The posted
     dataset has been balanced to only include cases where all four travel modes are recorded.

     Christophier V. Forinash and Frank S. Koppelman (1993) “Application and interpretation of
     nested logit models of intercity mode choice,” Transportation Research Record 1413, 98-106. """
    _ = to_wide
    data_file_name = "ModeCanada.csv.gz"
    # names, data = load_gzip(data_file_name)
    # names = [name.replace('"', "") for name in names]
    # canada_df = pd.DataFrame(data[:, 1:], index=data[:, 0].astype(int), columns=names[1:])

    full_path = get_path(data_file_name, module=DATA_MODULE)
    canada_df = pd.read_csv(full_path)
    canada_df["alt"] = canada_df.apply(lambda row: row.alt.replace('"', ""), axis=1)
    # Just some typing
    canada_df.income = canada_df.income.astype("float32")

    items = ["air", "bus", "car", "train"]
    shared_features = ["income", "dist", "urban"]
    items_features = ["cost", "freq", "ovt", "ivt"]
    choice_column = "choice"

    if add_items_one_hot:
        canada_df["oh_air"] = canada_df.apply(
            lambda row: 1.0 if row.alt == items[0] else 0.0, axis=1
        )
        canada_df["oh_bus"] = canada_df.apply(
            lambda row: 1.0 if row.alt == items[1] else 0.0, axis=1
        )
        canada_df["oh_car"] = canada_df.apply(
            lambda row: 1.0 if row.alt == items[2] else 0.0, axis=1
        )
        canada_df["oh_train"] = canada_df.apply(
            lambda row: 1.0 if row.alt == items[3] else 0.0, axis=1
        )
        items_features = ["oh_air", "oh_bus", "oh_car", "oh_train"] + items_features

    if add_is_public:
        canada_df["is_public"] = canada_df.apply(
            lambda row: 0.0 if row.alt == "car" else 1.0, axis=1
        )
        items_features.append("is_public")

    if return_desc:
        return desc

    for col in canada_df.columns:
        canada_df[col] = pd.to_numeric(canada_df[col], errors="ignore")

    if choice_format == "items_id":
        # We need to transform how the choice is encoded to add the chosen item id
        named_choice = [0] * len(canada_df)
        for n_row, row in canada_df.iterrows():
            if row.choice == 0:
                sub_df = canada_df[canada_df.case == row.case]
                choice = sub_df.loc[sub_df.choice == 1].alt.to_numpy()[0]
                named_choice[n_row - 1] = choice

        for n_row, row in canada_df.iterrows():
            if row.choice == 1:
                named_choice[n_row - 1] = row.alt

        canada_df["choice"] = named_choice

    if as_frame:
        if split_features:
            shared_features_by_choice = canada_df[["case"] + shared_features].drop_duplicates()
            shared_features_by_choice = shared_features_by_choice.rename(
                columns={"case": "choice_id"}
            )

            items_features_by_choice = canada_df[["case", "alt"] + items_features]
            items_features_by_choice = items_features_by_choice.rename(
                columns={"case": "choice_id", "alt": "item_id"}
            )

            choices = canada_df.loc[canada_df.choice == 1][["case", "alt"]]
            choices = choices.rename(columns={"case": "choice_id", "alt": "choice"})

            return (
                shared_features_by_choice,
                items_features_by_choice,
                choices,
            )
        return canada_df

    if split_features:
        # Order of item_id is alphabetical: air, bus, car, train
        shared_features_by_choice = (
            canada_df[["case"] + shared_features].drop_duplicates()[shared_features].to_numpy()
        )

        cif = []
        ci_av = []
        for context in canada_df.case.unique():
            context_df = canada_df.loc[canada_df.case == context]
            # Order of item_id is alphabetical: air, bus, car, train
            cf = []
            cav = []
            for item in ["air", "bus", "car", "train"]:
                if item in context_df.alt.unique():
                    cf.append(context_df.loc[context_df.alt == item][items_features].to_numpy()[0])
                    cav.append(1)
                else:
                    cf.append([0.0, 0.0, 0.0, 0.0])
                    cav.append(0)
            cif.append(cf)
            ci_av.append(cav)
        items_features_by_choice = np.array(cif)
        available_items_by_choice = np.array(ci_av)

        choices = np.squeeze(canada_df.loc[canada_df.choice == 1]["alt"].to_numpy())
        choices = np.array([["air", "bus", "car", "train"].index(c) for c in choices])

        return (
            shared_features_by_choice,
            items_features_by_choice,
            available_items_by_choice,
            choices,
        )

    if len(items_features) == 0:
        items_features = None

    if preprocessing == "tutorial":
        # Following torch-choice guide:
        canada_df = canada_df.loc[canada_df.noalt == 4]
        if add_items_one_hot:
            preprocessing_items_features = ["oh_air", "oh_car", "oh_bus", "oh_train"] + [
                "cost",
                "freq",
                "ovt",
                "ivt",
            ]
        else:
            preprocessing_items_features = ["cost", "freq", "ovt", "ivt"]

        items = ["air", "bus", "car", "train"]
        canada_df = canada_df.astype({"income": "float32"})
        return ChoiceDataset.from_single_long_df(
            df=canada_df,
            shared_features_columns=["income"],
            items_features_columns=preprocessing_items_features,
            items_id_column="alt",
            choices_id_column="case",
            choices_column="choice",
            choice_format="one_zero",
        )

    return ChoiceDataset.from_single_long_df(
        df=canada_df,
        shared_features_columns=shared_features,
        items_features_columns=items_features,
        items_id_column="alt",
        choices_id_column="case",
        choices_column=choice_column,
        choice_format="one_zero",
    )


def load_heating(
    as_frame=False,
    to_wide=False,
    return_desc=False,
):
    """Load and return the Heating dataset from Kenneth Train.

    Parameters
    ----------
    as_frame : bool, optional
        Whether to return the dataset as pd.DataFrame. If not, returned as ChoiceDataset,
        by default False.
    return_desc : bool, optional
        Whether to return the description, by default False.
    to_wide : bool, optional
        Whether to return the dataset in wide format,
        by default False (an thus returned in long format).

    Returns
    -------
    ChoiceDataset
        Loaded Heating dataset
    """
    desc = """Kenneth Train's dataset containing data on choice of heating system in California
    houses.
    Description can be found at: https://rdrr.io/cran/mlogitBMA/man/heating.html

    Train, K.E. (2003) Discrete Choice Methods with Simulation. Cambridge University Press."""
    _ = to_wide
    data_file_name = "heating_data.csv.gz"

    full_path = get_path(data_file_name, module=DATA_MODULE)
    heating_df = pd.read_csv(full_path)

    if return_desc:
        return desc

    if as_frame:
        return heating_df

    shared_features_by_choice = ["income", "agehed", "rooms", "region"]
    choice = ["depvar"]
    items_features_by_choice = ["ic.", "oc."]
    items = ["gc", "gr", "ec", "er", "hp"]

    choices = np.array([items.index(val) for val in heating_df[choice].to_numpy().ravel()])
    contexts = heating_df[shared_features_by_choice].to_numpy()
    contexts_items = np.stack(
        [
            heating_df[[feat + item for feat in items_features_by_choice]].to_numpy()
            for item in items
        ],
        axis=1,
    )
    return ChoiceDataset(
        shared_features_by_choice=contexts,
        items_features_by_choice=contexts_items,
        choices=choices,
        shared_features_by_choice_names=shared_features_by_choice,
        items_features_by_choice_names=items_features_by_choice,
    )


def load_electricity(
    as_frame=False,
    to_wide=False,
    return_desc=False,
):
    """Load and return the Electricity dataset from Kenneth Train.

    Parameters
    ----------
    as_frame : bool, optional
        Whether to return the dataset as pd.DataFrame. If not, returned as ChoiceDataset,
        by default False.
    to_wide : bool, optional
        Whether to return the dataset in wide format,
        by default False (an thus returned in long format).
    return_desc : bool, optional
        Whether to return the description, by default False.

    Returns
    -------
    ChoiceDataset
        Loaded Electricity dataset
    """
    _ = to_wide
    data_file_name = "electricity.csv.gz"

    description = """A sample of 2308 households in the United States.
    - choice: the choice of the individual, one of 1, 2, 3, 4,
    - id: the individual index,
    - pfi: fixed price at a stated cents per kWh, with the price varying over suppliers and
        experiments, for scenario i=(1, 2, 3, 4),
    - cli: the length of contract that the supplier offered, in years (such as 1 year or 5 years.)
        During this contract period, the supplier guaranteed the prices and the buyer would have to
        pay a penalty if he/she switched to another supplier. The supplier could offer no
        contractin which case either side could stop the agreement at any time. This is recorded
        as a contract length of 0,
    - loci: is the supplier a local company,
    - wki: is the supplier a well-known company,
    - todi: a time-of-day rate under which the price is 11 cents per kWh from 8am to 8pm and 5 cents
        per kWh from 8pm to 8am. These TOD prices did not vary over suppliers or experiments:
        whenever the supplier was said to offer TOD, the prices were stated as above.
    - seasi: a seasonal rate under which the price is 10 cents per kWh in the summer, 8 cents per
        kWh in the winter, and 6 cents per kWh in the spring and fall. Like TOD rates, these prices
        did not vary. Note that the price is for the electricity only, not transmission and
        distribution, which is supplied by the local regulated utility.

    Train, K.E. (2003) Discrete Choice Methods with Simulation. Cambridge University Press.
    """

    full_path = get_path(data_file_name, module=DATA_MODULE)
    elec_df = pd.read_csv(full_path)
    elec_df.choice = elec_df.choice.astype(int)
    elec_df[["pf", "cl", "loc", "wk", "tod", "seas"]] = elec_df[
        ["pf", "cl", "loc", "wk", "tod", "seas"]
    ].astype(float)

    if as_frame:
        return elec_df
    if return_desc:
        return description

    return ChoiceDataset.from_single_long_df(
        df=elec_df,
        items_features_columns=["pf", "cl", "loc", "wk", "tod", "seas"],
        items_id_column="alt",
        choices_id_column="chid",
        choice_format="one_zero",
    )


def load_train(
    as_frame=False,
    to_wide=False,
    return_desc=False,
):
    """Load and return the Train dataset from Koppleman et al. (1993).

    Parameters
    ----------
    as_frame : bool, optional
        Whether to return the dataset as pd.DataFrame. If not, returned as ChoiceDataset,
        by default False.
    to_wide : bool, optional
        Whether to return the dataset in wide format,
        by default False (an thus returned in long format).
    return_desc : bool, optional
        Whether to return the description, by default False.

    Returns
    -------
    ChoiceDataset
        Loaded Train dataset
    """
    desc = "A sample of 235  Dutchindividuals facing 2929 choice situations."
    desc += """Ben-Akiva M, Bolduc D, Bradley M(1993).
    “Estimation of Travel Choice Models with Randomly Distributed Values of Time.
    ”Papers 9303, Laval-Recherche en Energie. https://ideas.repec.org/p/fth/lavaen/9303.html."""
    _ = to_wide
    data_file_name = "train_data.csv.gz"

    full_path = get_path(data_file_name, module=DATA_MODULE)
    train_df = pd.read_csv(full_path)

    if return_desc:
        return desc

    if as_frame:
        return train_df
    train_df["choice"] = train_df.apply(lambda row: row.choice[-1], axis=1)

    return ChoiceDataset.from_single_wide_df(
        df=train_df,
        items_id=["1", "2"],
        shared_features_columns=["id"],
        items_features_prefixes=["price", "time", "change", "comfort"],
        delimiter="",
        available_items_suffix=None,
        choices_column="choice",
        choice_format="items_id",
    )


def load_car_preferences(
    as_frame=False,
    return_desc=False,
):
    """Load and return the Car dataset from  McFadden, Daniel and Kenneth Train (2000).

    “Mixed MNL models for discrete response”, Journal of Applied Econometrics, 15(5), 447–470.

    Parameters
    ----------
    as_frame : bool, optional
        Whether to return the dataset as pd.DataFrame. If not, returned as ChoiceDataset,
        by default False.
    return_desc : bool, optional
        Whether to return the description, by default False.

    Returns
    -------
    ChoiceDataset
        Loaded Train dataset
    """
    desc = "Stated Preferences for Car Choice."
    desc += """McFadden, Daniel and Kenneth Train (2000)
    “Mixed MNL models for discrete response”, Journal of Applied Econometrics, 15(5), 447–470."""

    data_file_name = "car.csv.gz"

    full_path = get_path(data_file_name, module=DATA_MODULE)
    cars_df = pd.read_csv(full_path)

    if return_desc:
        return desc

    if as_frame:
        return cars_df

    cars_df["choice"] = cars_df.apply(lambda row: row.choice[-1], axis=1)
    shared_features = ["college", "hsg2", "coml5"]
    items_features = [
        "type",
        "fuel",
        "price",
        "range",
        "acc",
        "speed",
        "pollution",
        "size",
        "space",
        "cost",
        "station",
    ]
    items_id = [f"{i}" for i in range(1, 7)]

    return ChoiceDataset.from_single_wide_df(
        df=cars_df,
        items_id=items_id,
        shared_features_columns=shared_features,
        items_features_prefixes=items_features,
        delimiter="",
        choices_column="choice",
        choice_format="items_id",
    )


def load_hc(
    as_frame=False,
    return_desc=False,
):
    """Load and return the HC dataset from Kenneth Train.

    Parameters
    ----------
    as_frame : bool, optional
        Whether to return the dataset as pd.DataFrame. If not, returned as ChoiceDataset,
        by default False.
    return_desc : bool, optional
        Whether to return the description, by default False.

    Returns
    -------
    ChoiceDataset
        Loaded Train dataset
    """
    desc = """HC contains data on the choice of heating and central cooling system for 250
    single-family, newly built houses in California.

    The alternatives are:

    Gas central heat with cooling gcc,
    Electric central resistence heat with cooling ecc,
    Electric room resistence heat with cooling erc,
    Electric heat pump, which provides cooling also hpc,
    Gas central heat without cooling gc,
    Electric central resistence heat without cooling ec,
    Electric room resistence heat without cooling er.
    Heat pumps necessarily provide both heating and cooling such that heat pump without cooling is
    not an alternative.

    The variables are:

    depvar gives the name of the chosen alternative,
    ich.alt are the installation cost for the heating portion of the system,
    icca is the installation cost for cooling
    och.alt are the operating cost for the heating portion of the system
    occa is the operating cost for cooling
    income is the annual income of the household
    Note that the full installation cost of alternative gcc is ich.gcc+icca, and similarly for the
    operating cost and for the other alternatives with cooling.
    """

    data_file_name = "HC.csv.gz"

    full_path = get_path(data_file_name, module=DATA_MODULE)
    hc_df = pd.read_csv(full_path)

    if return_desc:
        return desc

    if as_frame:
        return hc_df

    items_id = ["gcc", "ecc", "erc", "hpc", "gc", "ec", "er"]
    return ChoiceDataset.from_single_wide_df(
        df=hc_df,
        shared_features_columns=["income"],
        items_features_prefixes=["ich", "och", "occa", "icca"],
        delimiter=".",
        items_id=items_id,
        choices_column="depvar",
        choice_format="items_id",
    )
