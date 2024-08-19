"""Unit testing for included Open Source datasets loaders."""

import pandas as pd

from choice_learn.data import ChoiceDataset
from choice_learn.datasets import (
    load_car_preferences,
    load_electricity,
    load_hc,
    load_heating,
    load_modecanada,
    load_swissmetro,
    load_tafeng,
    load_train,
)


def test_swissmetro_loader():
    """Test loading the Swissmetro dataset."""
    swissmetro = load_swissmetro(as_frame=True)
    assert isinstance(swissmetro, pd.DataFrame)
    assert swissmetro.shape == (10719, 29)

    swissmetro = load_swissmetro()
    assert isinstance(swissmetro, ChoiceDataset)
    swissmetro = load_swissmetro(add_items_one_hot=True)
    assert isinstance(swissmetro, ChoiceDataset)


def test_modecanada_loader():
    """Test loading the Canada dataset."""
    canada = load_modecanada(as_frame=True)
    assert isinstance(canada, pd.DataFrame)
    assert canada.shape == (15520, 12)

    canada = load_modecanada()
    assert isinstance(canada, ChoiceDataset)


def test_electricity_loader():
    """Test loading the Electricity dataset."""
    electricity = load_electricity(as_frame=True)
    assert isinstance(electricity, pd.DataFrame)
    assert electricity.shape == (17232, 10)

    electricity = load_electricity()
    assert isinstance(electricity, ChoiceDataset)


def test_train_loader():
    """Test loading the Train dataset."""
    train = load_train(as_frame=True)
    assert isinstance(train, pd.DataFrame)
    assert train.shape == (2929, 11)

    train = load_train()
    assert isinstance(train, ChoiceDataset)


def test_tafeng_loader():
    """Test loading the TaFeng dataset."""
    tafeng = load_tafeng(as_frame=True)
    assert isinstance(tafeng, pd.DataFrame)
    assert tafeng.shape == (817741, 9)

    tafeng = load_tafeng()
    assert isinstance(tafeng, ChoiceDataset)


def test_heating_loader():
    """Test loading the heating dataset."""
    heating = load_heating(as_frame=True)
    assert isinstance(heating, pd.DataFrame)
    assert heating.shape == (900, 16)

    heating = load_heating()
    assert isinstance(heating, ChoiceDataset)


def test_car_preferences_loader():
    """Test loading the car preferences dataset."""
    cars = load_car_preferences(as_frame=True)
    assert isinstance(cars, pd.DataFrame)
    assert cars.shape == (4654, 71)

    cars = load_car_preferences()
    assert isinstance(cars, ChoiceDataset)


def test_hc_loader():
    """Test loading the car preferences dataset."""
    hc = load_hc(as_frame=True)
    assert isinstance(hc, pd.DataFrame)
    assert hc.shape == (250, 19)

    hc = load_hc(as_frame=False)
    assert isinstance(hc, ChoiceDataset)
