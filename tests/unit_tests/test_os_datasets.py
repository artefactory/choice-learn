"""Unit testing for included Open Source datasets loaders."""

import numpy as np
import pandas as pd

from choice_learn.data import ChoiceDataset
from choice_learn.datasets import (
    load_car_preferences,
    load_electricity,
    load_hc,
    load_heating,
    load_londonpassenger,
    load_modecanada,
    load_swissmetro,
    load_tafeng,
    load_train,
)
from choice_learn.datasets.base import load_csv, load_gzip, slice_from_names


def test_swissmetro_loader():
    """Test loading the Swissmetro dataset."""
    swissmetro = load_swissmetro(as_frame=True)
    assert isinstance(swissmetro, pd.DataFrame)
    assert swissmetro.shape == (10719, 29)

    swissmetro = load_swissmetro()
    assert isinstance(swissmetro, ChoiceDataset)
    swissmetro = load_swissmetro(add_items_one_hot=True)
    assert isinstance(swissmetro, ChoiceDataset)


def test_swissmetro_long_format():
    """Test loading the Swissmetro dataset in long format."""
    swissmetro = load_swissmetro(as_frame=True, preprocessing="long_format")
    assert isinstance(swissmetro, pd.DataFrame)
    assert swissmetro.shape == (30474, 7)


def test_swissmetro_tastenet():
    """Test TasteNet preprocessing of dataset."""
    _ = load_swissmetro(preprocessing="tastenet")


def test_swissmetro_tutorial():
    """Test tutorial preprocessing of dataset."""
    _ = load_swissmetro(preprocessing="tutorial")


def test_biogeme_nested_tutorial():
    """Test biogeme_nested preprocessing of dataset."""
    _ = load_swissmetro(preprocessing="biogeme_nested")


def test_rumnet_tutorial():
    """Test rumnet preprocessing of dataset."""
    _ = load_swissmetro(preprocessing="rumnet")


def test_modecanada_loader():
    """Test loading the Canada dataset."""
    canada = load_modecanada(as_frame=True, choice_format="items_id")
    assert isinstance(canada, pd.DataFrame)
    assert canada.shape == (15520, 11)

    canada = load_modecanada()
    assert isinstance(canada, ChoiceDataset)

    ca, na, da = load_modecanada(
        as_frame=True,
        add_items_one_hot=True,
        add_is_public=True,
        choice_format="items_id",
        split_features=True,
    )
    assert ca.shape == (4324, 4)
    assert na.shape == (15520, 11)
    assert da.shape == (4324, 2)


def test_modecanada_features_split():
    """Test that features are split well."""
    (
        o,
        ca,
        na,
        da,
    ) = load_modecanada(add_items_one_hot=True, add_is_public=True, split_features=True)
    assert o.shape == (4324, 3)
    assert ca.shape == (4324, 4, 9)
    assert na.shape == (4324, 4)
    assert da.shape == (4324,)


def test_modecanada_loader_2():
    """Test loading the Canada dataset w/ preprocessing."""
    canada = load_modecanada(preprocessing="tutorial", add_items_one_hot=True)
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


def test_londonpassenger_loader():
    """Test loading the London Passenger Mode Choice dataset."""
    londonpassenger = load_londonpassenger(as_frame=True)
    assert isinstance(londonpassenger, pd.DataFrame)
    expected_columns = [ 
        'trip_id', 
        'household_id', 
        'person_n', 
        'trip_n', 
        'travel_mode',
        'fueltype', 
        'faretype', 
        'bus_scale', 
        'survey_year', 
        'travel_year',
        'travel_month', 
        'travel_date', 
        'day_of_week',
        'start_time', 
        'age', 
        'female',
        'driving_license', 
        'car_ownership', 
        'distance', 
        'dur_walking',
        'dur_cycling', 
        'dur_pt_access',
        'dur_pt_rail', 
        'dur_pt_bus',
        'dur_pt_int',
        'pt_interchanges', 
        'dur_driving', 
        'cost_transit',
        'cost_driving_fuel',
        'cost_driving_ccharge',
        'driving_traffic_percent', 
        'week_end', 
        'purpose_home_to_work',
        'purpose_home_to_school', 
        'purpose_home_to_other',
        'purpose_work_travel', 
        'purpose_other', 
    ]
    assert londonpassenger.columns.equals(pd.Index(expected_columns))
    assert londonpassenger.shape == (81086, 37)

    londonpassenger = load_londonpassenger()
    assert isinstance(londonpassenger, ChoiceDataset)
    assert np.shape(londonpassenger.items_features_by_choice) == (1, 81086, 4, 8)
    assert np.shape(londonpassenger.shared_features_by_choice) == (1, 81086, 19)
    expected_items_features_names = [
       'fueltype',
       'faretype',
       'bus_scale',
       'dur_pt',
       'interchanges',
       'cost_pt',
       'cost_driving',
       'driving_traffic_percent'
    ]
    assert londonpassenger.items_features_by_choice_names[0] == expected_items_features_names
    expected_shared_features_names = [
        'household_id',
        'person_n',
        'trip_n',
        'purpose_home_to_work',
        'purpose_home_to_school',
        'purpose_home_to_other',
        'purpose_work_travel',
        'purpose_other',
        'survey_year',
        'travel_year',
        'travel_month',
        'travel_date',
        'week_end',
        'start_time',
        'age',
        'female',
        'driving_license',
        'car_ownership',
        'distance'
    ]
    assert londonpassenger.shared_features_by_choice_names[0] == expected_shared_features_names

    londonpassenger = load_londonpassenger(add_items_one_hot=True)
    assert isinstance(londonpassenger, ChoiceDataset)
    assert np.shape(londonpassenger.items_features_by_choice) == (
        1,
        81086,
        4,
        12,
    )  # last dimension +4
    assert np.shape(londonpassenger.shared_features_by_choice) == (1, 81086, 19)
    expected_items_features_names = [
        'fueltype',
        'faretype',
         'bus_scale',
         'dur_pt',
        'interchanges',
         'cost_pt',
        'cost_driving',
        'driving_traffic_percent',
         'oh_walking',
         'oh_cycling',
        'oh_pt',
         'oh_driving'
    ]
    assert londonpassenger.items_features_by_choice_names[0] == expected_items_features_names
    expected_shared_features_names = [
        "household_id",
        "person_n",
        "trip_n",
        'purpose_home_to_work',
        'purpose_home_to_school',
        'purpose_home_to_other',
        'purpose_work_travel',
        'purpose_other',
        "survey_year",
        "travel_year",
        "travel_month",
        "travel_date",
        "week_end",
        "start_time",
        "age",
        "female",
        "driving_license",
        "car_ownership",
        "distance",
    ]
    assert londonpassenger.shared_features_by_choice_names[0] == expected_shared_features_names

    londonpassenger = load_londonpassenger(as_frame=True, preprocessing="summation")
    assert isinstance(londonpassenger, pd.DataFrame)
    expected_columns = [
        'trip_id', 'household_id', 'person_n', 'trip_n', 'travel_mode',
       'fueltype', 'faretype', 'bus_scale', 'survey_year', 'travel_year',
       'travel_month', 'travel_date', 'start_time', 'age', 'female',
       'driving_license', 'car_ownership', 'distance', 'dur_walking',
       'dur_cycling', 'pt_interchanges', 'dur_driving', 'cost_pt',
       'driving_traffic_percent', 'week_end', 'purpose_home_to_work',
       'purpose_home_to_school', 'purpose_home_to_other',
       'purpose_work_travel', 'purpose_other', 'dur_pt', 'cost_driving'
    ]
    assert londonpassenger.columns.equals(pd.Index(expected_columns))
    assert londonpassenger.shape == (81086, 32)

    londonpassenger = load_londonpassenger(preprocessing="summation")
    assert isinstance(londonpassenger, ChoiceDataset)
    assert np.shape(londonpassenger.items_features_by_choice) == (1, 81086, 4, 8)
    assert np.shape(londonpassenger.shared_features_by_choice) == (1, 81086, 19)
    expected_items_features_names = [
        'fueltype',
        'faretype',
        'bus_scale',
        'dur_pt',
        'interchanges',
        'cost_pt',
        'cost_driving',
        'driving_traffic_percent'
    ]
    assert londonpassenger.items_features_by_choice_names[0] == expected_items_features_names
    expected_shared_features_names = ['household_id',
         'person_n',
         'trip_n',
         'purpose_home_to_work',
         'purpose_home_to_school',
         'purpose_home_to_other',
         'purpose_work_travel',
         'purpose_other',
         'survey_year',
         'travel_year',
         'travel_month',
         'travel_date',
         'week_end',
         'start_time',
         'age',
         'female',
         'driving_license',
         'car_ownership',
         'distance']
    assert londonpassenger.shared_features_by_choice_names[0] == expected_shared_features_names

    londonpassenger = load_londonpassenger(add_items_one_hot=True, preprocessing="summation")
    assert isinstance(londonpassenger, ChoiceDataset)
    assert np.shape(londonpassenger.items_features_by_choice) == (
        1,
        81086,
        4,
        12,
    )  # last dimension +4
    assert np.shape(londonpassenger.shared_features_by_choice) == (1, 81086, 19)
    expected_items_features_names = [
        'fueltype',
        'faretype',
        'bus_scale',
        'dur_pt',
        'interchanges',
        'cost_pt',
        'cost_driving',
        'driving_traffic_percent',
        'oh_walking',
        'oh_cycling',
        'oh_pt',
        'oh_driving'
    ]
    assert londonpassenger.items_features_by_choice_names[0] == expected_items_features_names
    expected_shared_features_names = [
        'household_id',
        'person_n',
        'trip_n',
        'purpose_home_to_work',
        'purpose_home_to_school',
        'purpose_home_to_other',
        'purpose_work_travel',
        'purpose_other',
        'survey_year',
        'travel_year',
        'travel_month',
        'travel_date',
        'week_end',
        'start_time',
        'age',
        'female',
        'driving_license',
        'car_ownership',
        'distance'
    ]
    assert londonpassenger.shared_features_by_choice_names[0] == expected_shared_features_names


def test_description():
    """Test getting description."""
    _ = load_swissmetro(return_desc=True)
    _ = load_modecanada(return_desc=True)
    _ = load_heating(return_desc=True)
    _ = load_electricity(return_desc=True)
    _ = load_train(return_desc=True)
    _ = load_car_preferences(return_desc=True)
    _ = load_hc(return_desc=True)
    _ = load_londonpassenger(return_desc=True)
    _ = load_tafeng(return_desc=True)


def test_load_csv():
    """Test csv file loader."""
    _ = load_csv(data_file_name="test_data.csv", data_module="tests/data")
    names, data = load_gzip("swissmetro.csv.gz", data_module="choice_learn/datasets/data")
    _ = slice_from_names(data, names[:4], names)
