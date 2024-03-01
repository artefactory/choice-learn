"""Some datasets used for personal examples."""
import os

import numpy as np
import pandas as pd

from choice_learn.data.choice_dataset import ChoiceDataset

DATA_MODULE = os.path.join(os.path.abspath(".."), "choice_learn", "datasets", "data")


def load_tafeng(as_frame=False, preprocessing=None):
    """Function to load the TaFeng dataset.

    Orginal file and informations can be found here:
    https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset/

    Parameters
    ----------
    as_frame : bool, optional
        Whether to return the original file as pd.DF, by default False
    preprocessing : str, optional
        predefined pre-processing to apply, by default None

    Returns:
    --------
    pd.DF or ChoiceDataset
        TaFeng Grocery Dataset.
    """
    filepath = os.path.join(DATA_MODULE, "ta_feng.csv.zip")
    # url = "https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset/download?datasetVersionNumber=1"
    # if not os.path.exists(filepath):
    #     with urllib.request.urlopen(url) as f:
    #         file = f.read().decode("utf-8")

    tafeng_df = pd.read_csv(filepath)
    if as_frame:
        return tafeng_df

    if preprocessing == "assort_example":
        subdf = tafeng_df.loc[tafeng_df.PRODUCT_SUBCLASS == 100505]
        prods = subdf.PRODUCT_ID.value_counts().index[
            (subdf.PRODUCT_ID.value_counts() > 20).to_numpy()
        ]
        subdf = tafeng_df.loc[tafeng_df.PRODUCT_ID.isin(prods)]
        subdf = subdf.dropna()
        subdf = subdf.reset_index(drop=True)

        # Create Prices
        items = list(subdf.PRODUCT_ID.unique())
        init_prices = []
        for item in items:
            first_price = subdf.loc[subdf.PRODUCT_ID == item].SALES_PRICE.to_numpy()[0]
            init_prices.append(first_price)

        # Encode Age Groups
        age_groups = {}
        for i, j in enumerate(subdf.AGE_GROUP.unique()):
            age_groups[j] = i
        age_groups = {
            "<25": 0,
            "25-29": 0,
            "30-34": 0,
            "35-39": 1,
            "40-44": 1,
            "45-49": 1,
            "50-54": 2,
            "55-59": 2,
            "60-64": 2,
            ">65": 2,
        }
        age_groups = {
            "<25": [1, 0, 0],
            "25-29": [0, 1, 0],
            "30-34": [0, 1, 0],
            "35-39": [0, 1, 0],
            "40-44": [0, 1, 0],
            "45-49": [0, 1, 0],
            "50-54": [0, 0, 1],
            "55-59": [0, 0, 1],
            "60-64": [0, 0, 1],
            ">65": [0, 0, 1],
        }

        all_prices = []
        customer_features = []
        choices = []

        curr_prices = [i for i in init_prices]

        for n_row, row in subdf.iterrows():
            for _ in range(int(row.AMOUNT)):
                item = row.PRODUCT_ID
                price = row.SALES_PRICE / row.AMOUNT
                age = row.AGE_GROUP

                item_index = items.index(item)

                # customer_features.append([age_groups[age]])
                customer_features.append(age_groups[age])
                choices.append(item_index)
                curr_prices[item_index] = price
                all_prices.append([i for i in curr_prices])

        all_prices = np.expand_dims(np.array(all_prices), axis=-1)
        customer_features = np.array(customer_features).astype("float32")
        choices = np.array(choices)

        # Create Dataset
        return ChoiceDataset(
            contexts_features=customer_features,
            choices=choices,
            contexts_items_features=all_prices,
            contexts_items_availabilities=np.ones((len(choices), 25)).astype("float32"),
        )

    return tafeng_df
