"""Tests specific config of NestedLogit."""

import tensorflow as tf

from choice_learn.data import ChoiceDataset
from choice_learn.datasets import load_hc
from choice_learn.models import NestedLogit

hc_df = load_hc(as_frame=True)
items_id = ["gcc", "ecc", "erc", "hpc", "gc", "ec", "er"]
cooling_modes = ["gcc", "ecc", "erc", "hpc"]
room_modes = ["erc", "er"]

for mode in items_id:
    if mode in cooling_modes:
        hc_df[f"icca.{mode}"] = hc_df["icca"]
        hc_df[f"occa.{mode}"] = hc_df["occa"]
    else:
        hc_df[f"icca.{mode}"] = 0.0
        hc_df[f"occa.{mode}"] = 0.0

for item in items_id:
    if item in cooling_modes:
        hc_df[f"int_cooling.{item}"] = 1.0
        hc_df[f"inc_cooling.{item}"] = hc_df.income
    else:
        hc_df[f"int_cooling.{item}"] = 0.0
        hc_df[f"inc_cooling.{item}"] = 0.0
    if item in room_modes:
        hc_df[f"inc_room.{item}"] = hc_df.income
    else:
        hc_df[f"inc_room.{item}"] = 0

dataset = ChoiceDataset.from_single_wide_df(
    df=hc_df,
    shared_features_columns=["income"],
    items_features_prefixes=[
        "ich",
        "och",
        "occa",
        "icca",
        "int_cooling",
        "inc_cooling",
        "inc_room",
    ],
    delimiter=".",
    items_id=items_id,
    choices_column="depvar",
    choice_format="items_id",
)


def test_fit_hc_formul_1():
    """Tests specific config of NestedLogit on HC dataset."""
    tf.config.run_functions_eagerly(True)
    global dataset

    spec = {
        "ich": "constant",
        "och": "constant",
        "occa": "constant",
        "icca": "constant",
        "int_cooling": "constant",
        "inc_cooling": "constant",
        "inc_room": "constant",
    }
    model = NestedLogit(
        coefficients=spec,
        items_nests=[[0, 1, 2, 3], [4, 5, 6]],
        optimizer="lbfgs",
        shared_gammas_over_nests=True,
    )

    _ = model.fit(dataset, get_report=True, verbose=1)

    nll = model.evaluate(dataset) * len(dataset)
    assert nll < 180.0
    assert model.report.shape == (8, 5)


def test_fit_hc_formul_2():
    """Test config with add_coefficient formulation."""
    tf.config.run_functions_eagerly(True)
    global dataset

    model = NestedLogit(
        items_nests=[[0, 1, 2, 3], [4, 5, 6]], optimizer="lbfgs", shared_gammas_over_nests=False
    )
    # Coefficients that are for all the alternatives
    model.add_shared_coefficient(feature_name="ich", items_indexes=[0, 1, 2, 3, 4, 5, 6])
    model.add_shared_coefficient(feature_name="och", items_indexes=[0, 1, 2, 3, 4, 5, 6])
    model.add_shared_coefficient(feature_name="icca", items_indexes=[0, 1, 2, 3, 4, 5, 6])
    model.add_shared_coefficient(feature_name="occa", items_indexes=[0, 1, 2, 3, 4, 5, 6])

    # The coefficients concerning the income are split into two groups of alternatives:
    model.add_shared_coefficient(
        feature_name="income", items_indexes=[0, 1, 2, 3], coefficient_name="income_cooling"
    )
    model.add_shared_coefficient(
        feature_name="income", items_indexes=[2, 6], coefficient_name="income_room"
    )

    # Finally only one nest has an intercept
    model.add_shared_coefficient(feature_name="intercept", items_indexes=[0, 1, 2, 3])
    _ = model.fit(dataset, get_report=False, verbose=2)

    assert model.evaluate(dataset) < 180.0
