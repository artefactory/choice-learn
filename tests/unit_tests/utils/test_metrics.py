"""Test the metrics functions and classes."""

from choice_learn.utils.metrics import NegativeLogLikeliHood


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
    assert met.result() > 2.1270


# def test_sample_weights():
#     """Test sample_weight parametrization of CCE loss."""
#     exact_nll = CustomCategoricalCrossEntropy(
#         from_logits=False,
#         label_smoothing=0.0,
#         sparse=False,
#         axis=-1,
#         epsilon=1e-35,
#         name="exact_categorical_crossentropy",
#         reduction="sum_over_batch_size",
#     )

#     value_ref = exact_nll(
#         y_true=[[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]], y_pred=[[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]]
#     )
#     value_1 = exact_nll(
#         y_true=[[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
#         y_pred=[[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]],
#         sample_weight=[1.0, 1.0, 1.0],
#     )
#     value_2 = exact_nll(
#         y_true=[[0.0, 1.0], [1.0, 0.0]], y_pred=[[0.2, 0.8], [0.2, 0.8]], sample_weight=[1.0, 2.0]
#     )
#     value_3 = exact_nll(
#         y_true=[[0.0, 1.0], [1.0, 0.0]], y_pred=[[0.2, 0.8], [0.2, 0.8]], sample_weight=[0.5, 1.0]
#     )

#     assert value_ref == value_1
#     assert value_ref == value_2 * 2 / 3
#     assert value_ref == value_3 * 4 / 3
