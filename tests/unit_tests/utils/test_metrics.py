"""Test the metrics functions and classes."""

import numpy as np

from choice_learn.utils.metrics import (
    MRR,
    HitRate,
    MeanRank,
    NegativeLogLikeliHood,
)


def test_custom_categorical_crossentropy():
    """Test the NegativeLogLikeliHood metric class."""
    met = NegativeLogLikeliHood()
    met.update_state(
        y_true=[[0.0, 1.0]],
        y_pred=[[0.0, 1.0]],
    )
    assert met.result() == 0.0
    met.reset_state()
    met.update_state(
        y_true=[[0.0, 1.0]],
        y_pred=[[1.0, 0.0]],
    )
    assert met.result() > 20.0

    met = NegativeLogLikeliHood(sparse=True)
    met.update_state(
        y_true=[1],
        y_pred=[[0.0, 1.0]],
    )
    assert met.result() == 0.0
    met.reset_state()
    met.update_state(
        y_true=[1],
        y_pred=[[1.0, 0.0]],
    )
    assert met.result() > 20.0

    met = NegativeLogLikeliHood(sparse=True, from_logits=True)
    met.update_state(
        y_true=[1],
        y_pred=[[0.0, 21.0]],
    )
    assert met.result() < 1e-10
    met.reset_state()
    met.update_state(
        y_true=[1],
        y_pred=[[21.0, 0.0]],
    )
    assert met.result() > 20.0

    met.reset_state()
    met.update_state(
        y_true=[1],
        y_pred=[[2.0, 4.0]],
    )
    met.update_state(
        y_true=[0],
        y_pred=[[4.0, 2.0]],
    )
    assert met.result() > 0.1269
    assert met.result() < 0.1270

    met.reset_state()
    met.update_state(
        y_true=[0],
        y_pred=[[2.0, 4.0]],
    )
    met.update_state(
        y_true=[1],
        y_pred=[[4.0, 2.0]],
    )
    assert met.result() > 2.1269
    assert met.result() < 2.1270


def test_sample_weights():
    """Test sample_weight use for NLL metric."""
    exact_nll = NegativeLogLikeliHood(
        from_logits=False,
        sparse=False,
        axis=-1,
        epsilon=1e-35,
        name="exact_categorical_crossentropy",
    )

    exact_nll.update_state(
        y_true=[[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]], y_pred=[[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]]
    )
    value_ref = exact_nll.result()
    exact_nll.reset_state()
    exact_nll.update_state(
        y_true=[[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
        y_pred=[[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]],
        sample_weight=[1.0, 1.0, 1.0],
    )
    value_1 = exact_nll.result()
    exact_nll.reset_state()

    exact_nll.update_state(
        y_true=[[0.0, 1.0], [1.0, 0.0]], y_pred=[[0.2, 0.8], [0.2, 0.8]], sample_weight=[1.0, 2.0]
    )
    value_2 = exact_nll.result()
    exact_nll.reset_state()

    exact_nll.update_state(
        y_true=[[0.0, 1.0], [1.0, 0.0]], y_pred=[[0.2, 0.8], [0.2, 0.8]], sample_weight=[0.5, 1.0]
    )
    value_3 = exact_nll.result()
    exact_nll.reset_state()

    assert np.allclose(value_ref, value_1)
    assert np.allclose(value_ref, value_2)
    assert np.allclose(value_ref, value_3)


def test_mrr_metric():
    """Test the Mean Reciprocal Rank (MRR) metric."""
    met = MRR()
    met.update_state(
        y_true=[1, 0],
        y_pred=np.array(
            [
                [0.1, 0.9, 0.0],
                [0.4, 0.5, 0.1],
            ]
        ),
    )
    assert np.allclose(met.result().numpy(), 0.75)


def test_mean_rank_metric():
    """Test the MeanRank metric."""
    met = MeanRank()
    met.update_state(
        y_true=[2, 0],
        y_pred=np.array(
            [
                [0.1, 0.2, 0.7],
                [0.1, 0.6, 0.3],
            ]
        ),
    )
    assert np.allclose(met.result().numpy(), 2.0)


def test_hit_rate_metric():
    """Test the HitRate metric."""
    # Test Hit Rate @ 2
    met = HitRate(top_k=2)
    met.update_state(
        y_true=[1, 0],
        y_pred=[
            [0.6, 0.3, 0.1],
            [0.1, 0.5, 0.4],
        ],
    )
    assert np.allclose(met.result().numpy(), 0.5)


def test_average_on_trip():
    """Test the average_on_trip logic using batch clustering for metrics."""
    met = MeanRank(average_on_trip=True)
    met.update_state(
        y_true=[2, 0, 1],
        y_pred=np.array(
            [
                [0.1, 0.2, 0.7],
                [0.1, 0.6, 0.3],
                [0.3, 0.5, 0.2],
            ]
        ),
        batch=[0, 0, 1],
    )
    assert np.allclose(met.result().numpy(), 1.5)
