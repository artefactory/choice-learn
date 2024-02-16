"""Datasets loader."""
import csv
import gzip
from importlib import resources

import numpy as np
import pandas as pd

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
    data = data.astype(int)

    items = ["TRAIN", "SM", "CAR"]
    items_features_names = []
    session_features_names = [
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
    sessions_items_features_names = ["TT", "CO", "HE"]
    sessions_items_features_names = [
        [f"{item}_{feature}" for feature in sessions_items_features_names] for item in items
    ]
    sessions_items_availabilities = ["TRAIN_AV", "SM_AV", "CAR_AV"]
    choice_column = "CHOICE"

    if add_items_one_hot:
        items_features = np.eye(len(items), dtype=np.float64)
        items_features_names = [f"oh_{item}" for item in items]
    else:
        items_features = None
        items_features_names = None

    # Adding dummy CAR_HE feature as 0 for consistency
    names.append("CAR_HE")
    data = np.hstack([data, np.zeros((data.shape[0], 1))])

    session_features = slice_from_names(data, session_features_names, names)
    sessions_items_features = np.stack(
        [slice_from_names(data, features, names) for features in sessions_items_features_names],
        axis=-1,
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
        return pd.DataFrame(data, columns=names)

    return ChoiceDataset(
        fixed_items_features=items_features,
        contexts_features=session_features,
        contexts_items_features=sessions_items_features,
        contexts_items_availabilities=sessions_items_availabilities,
        choices=choices,
        fixed_items_features_names=items_features_names,
        contexts_features_names=session_features_names,
        contexts_items_features_names=sessions_items_features_names,
    )


def load_modecanada(
    add_items_one_hot=False,
    add_is_public=False,
    as_frame=False,
    return_desc=False,
    choice_mode="one_zero",
    split_features=False,
    to_wide=False,
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
    choice_mode : str, optional, among ["one_zero", "items_id"]
        mode indicating how the choice is encoded, by default "one_zero".
    split_features : bool, optional
        Whether to split features by type in different dataframes, by default False.
    to_wide : bool, optional
        Whether to return the dataset in wide format,
        by default False (an thus retuned in long format).

    Returns:
    --------
    ChoiceDataset
        Loaded ModeCanada dataset
    """
    _ = to_wide
    data_file_name = "ModeCanada.csv.gz"
    names, data = load_gzip(data_file_name)
    names = [name.replace('"', "") for name in names]
    canada_df = pd.DataFrame(data[:, 1:], index=data[:, 0].astype(int), columns=names[1:])
    canada_df["alt"] = canada_df.apply(lambda row: row.alt.replace('"', ""), axis=1)
    # Just some typing
    canada_df.income = canada_df.income.astype("float32")

    items = ["air", "bus", "car", "train"]
    items_features = []
    session_features = ["income", "dist", "urban"]
    sessions_items_features = ["cost", "freq", "ovt", "ivt"]
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
        items_features = ["oh_air", "oh_bus", "oh_car", "oh_train"]

    if add_is_public:
        canada_df["is_public"] = canada_df.apply(
            lambda row: 0.0 if row.alt == "car" else 1.0, axis=1
        )
        items_features.append("is_public")

    if return_desc:
        # TODO
        pass

    for col in canada_df.columns:
        canada_df[col] = pd.to_numeric(canada_df[col], errors="ignore")

    if choice_mode == "items_id":
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
            if add_is_public:
                fixed_items_features = pd.DataFrame(
                    {"item_id": ["car", "train", "bus", "air"], "is_public": [0, 1, 1, 1]}
                )
            else:
                fixed_items_features = None
            contexts_features = canada_df[["case", "income", "dist", "urban"]].drop_duplicates()
            contexts_features = contexts_features.rename(columns={"case": "context_id"})

            contexts_items_features = canada_df[["case", "alt", "freq", "cost", "ivt", "ovt"]]
            contexts_items_features = contexts_items_features.rename(
                columns={"case": "context_id", "alt": "item_id"}
            )

            choices = canada_df.loc[canada_df.choice == 1][["case", "alt"]]
            choices = choices.rename(columns={"case": "context_id", "alt": "choice"})

            return fixed_items_features, contexts_features, contexts_items_features, choices
        return canada_df

    if split_features:
        # Order of item_id is alphabetical: air, bus, car, train
        if add_is_public:
            fixed_items_features = np.array([[1.0], [1.0], [0.0], [1.0]])
        else:
            fixed_items_features = None
        contexts_features = (
            canada_df[["case", "income", "dist", "urban"]]
            .drop_duplicates()[["income", "dist", "urban"]]
            .to_numpy()
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
                    cf.append(
                        context_df.loc[context_df.alt == item][
                            ["freq", "cost", "ivt", "ovt"]
                        ].to_numpy()[0]
                    )
                    cav.append(1)
                else:
                    cf.append([0.0, 0.0, 0.0, 0.0])
                    cav.append(0)
            cif.append(cf)
            ci_av.append(cav)
        contexts_items_features = np.array(cif)
        contexts_items_availabilities = np.array(ci_av)

        choices = np.squeeze(canada_df.loc[canada_df.choice == 1]["alt"].to_numpy())
        choices = np.array([["air", "bus", "car", "train"].index(c) for c in choices])

        return (
            fixed_items_features,
            contexts_features,
            contexts_items_features,
            contexts_items_availabilities,
            choices,
        )

    if len(items_features) == 0:
        items_features = None

    return ChoiceDataset.from_single_long_df(
        df=canada_df,
        fixed_items_features_columns=items_features,
        contexts_features_columns=session_features,
        contexts_items_features_columns=sessions_items_features,
        items_id_column="alt",
        contexts_id_column="case",
        choices_column=choice_column,
        choice_mode="one_zero",
    )


def load_heating(
    as_frame=False,
    return_desc=False,
    to_wide=False,
):
    """Load and return the ModeCanada dataset from Koppleman et al. (1993).

    Parameters
    ----------
    as_frame : bool, optional
        Whether to return the dataset as pd.DataFrame. If not, returned as ChoiceDataset,
        by default False.
    return_desc : bool, optional
        Whether to return the description, by default False.
    to_wide : bool, optional
        Whether to return the dataset in wide format,
        by default False (an thus retuned in long format).

    Returns:
    --------
    ChoiceDataset
        Loaded ModeCanada dataset
    """
    _ = to_wide
    data_file_name = "heating_data.csv.gz"
    names, data = load_gzip(data_file_name)

    heating_df = pd.read_csv(resources.files(DATA_MODULE) / "heating_data.csv.gz")

    if return_desc:
        # TODO
        pass

    if as_frame:
        return heating_df

    contexts_features = ["income", "agehed", "rooms", "region"]
    choice = ["depvar"]
    contexts_items_features = ["ic.", "oc."]
    items = ["gc", "gr", "ec", "er", "hp"]

    choices = np.array([items.index(val) for val in heating_df[choice].to_numpy().ravel()])
    contexts = heating_df[contexts_features].to_numpy()
    contexts_items = np.stack(
        [
            heating_df[[feat + item for feat in contexts_items_features]].to_numpy()
            for item in items
        ],
        axis=1,
    )
    return ChoiceDataset(
        contexts_features=contexts, contexts_items_features=contexts_items, choices=choices
    )
