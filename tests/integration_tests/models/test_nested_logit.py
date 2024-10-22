"""Tests specific config of NestedLogit."""

import tensorflow as tf

from choice_learn.data import ChoiceDataset
from choice_learn.datasets import load_hc
from choice_learn.models import NestedLogit


def test_fit_hc_formul_1():
    """Tests specific config of NestedLogit on HC dataset."""
    tf.config.run_functions_eagerly(True)
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
