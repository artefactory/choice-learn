"""Some datasets used for personal examples."""

import numpy as np
import pandas as pd

from choice_learn.data.choice_dataset import ChoiceDataset
from choice_learn.datasets.base import get_path

from pathlib import Path
import tarfile
import os

from choice_learn.datasets.base import download_from_url
from choice_learn.basket_models.data.basket_dataset import Trip, TripDataset
DATA_MODULE = "choice_learn/datasets/data"


def load_tafeng(as_frame=False, return_desc=False, preprocessing=None):
    """Load the TaFeng dataset.

    Orginal file and informations can be found here:
    https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset/

    Parameters
    ----------
    as_frame : bool, optional
        Whether to return the original file as pd.DF, by default False
    preprocessing : str, optional
        predefined pre-processing to apply, by default None
    return_desc : bool, optional
        Whether to return the description of the dataset, by default False

    Returns
    -------
    pd.DF or ChoiceDataset
        TaFeng Grocery Dataset.
    """
    filename = "ta_feng.csv.zip"

    filepath = get_path(filename, module=DATA_MODULE)
    # url = "https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset/download?datasetVersionNumber=1"
    # if not os.path.exists(filepath):
    #     with urllib.request.urlopen(url) as f:
    #         file = f.read().decode("utf-8")

    description = """The dataset contains a Chinese grocery store transaction data from November
    2000 to February 2001.
    Details and files can be found at:
    https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset/download?datasetVersionNumber=1
    """

    tafeng_df = pd.read_csv(filepath)
    if as_frame:
        return tafeng_df
    if return_desc:
        return description

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
            shared_features_by_choice=customer_features,
            choices=choices,
            items_features_by_choice=all_prices,
            available_items_by_choice=np.ones((len(choices), 25)).astype("float32"),
        )

    return load_tafeng(as_frame=False, preprocessing="assort_example")



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
    extract_path = '../../choice_learn/datasets/data/'
    with tarfile.open(archive_path, "r:gz") as tar:

        # Here are the files we are downloading
        file_names = tar.getnames()
        print(f"Files : {file_names}")
        
        # We extract all the files 
        tar.extractall(path=extract_path)
        
        # We want to read the uchoice-Bakery.txt file (second file in the archive)
        csv_file_to_read = file_names[1] 
    
    noms_colonnes = ['article_1', 'article_2', 'article_3', 'article_4', 'article_5', 'article_6', 'article_7','article_8']

    # likewise get_path function
    path = Path(os.path.join("../../", DATA_MODULE)).resolve() / csv_file_to_read
    df = pd.read_csv(path, sep='\s+', header=None, names=noms_colonnes)

    if as_frame :
        return df
    
    n_item = int(df.max().max())
    
    # Apparently all items are available at each trip 
    availability_matrix = np.array([[1]*n_item]) 
    
    list_purchases = [[int(item)-1 for item in row if pd.notna(item)] for row in df.values]

    # Dummy prices, all equal to 1
    prices = np.array([[1]*n_item])
    trips_list = [Trip(purchases=purchases, assortment=0, prices =prices) for purchases in list_purchases]

    return TripDataset(trips = trips_list, available_items = availability_matrix)
