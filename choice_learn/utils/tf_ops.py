"""Diverse Tensorflow operations used in the ChoiceLearn library."""

import tensorflow as tf


def softmax_with_availabilities(
    items_logit_by_choice, available_items_by_choice, axis=-1, normalize_exit=False, eps=1e-5
):
    """Compute softmax probabilities from utilities.

    Takes into account availabilities (1 if the product is available, 0 otherwise) to set
    probabilities to 0 for unavailable products and to renormalize the probabilities of
    available products.

    Parameters
    ----------
    items_logit_by_choice : np.ndarray (n_choices, n_items)
        Utilities / Logits on which to compute the softmax
    available_items_by_choice : np.ndarray (n_choices, n_items)
        Matrix indicating the availability (1) or not (0) of the products
    axis : int, optional
        Axis of items_logit_by_choice on which to apply the softmax, by default -1
    normalize_exit : bool, optional
        Whether to normalize the probabilities of available products with an exit choice of
        utility 1, by default False
    eps : float, optional
        Value to avoid division by 0 when a product with probability almost 1 is unavailable,
        by default 1e-5

    Returns
    -------
    tf.Tensor (n_choices, n_items)
        Probabilities of each product for each choice computed from Logits
    """
    # Substract max utility to avoid overflow
    large_negative = tf.constant(-1e9, dtype=items_logit_by_choice.dtype)
    masked_logits = tf.where(available_items_by_choice > 0.5, items_logit_by_choice, large_negative)

    normalizer = tf.reduce_max(masked_logits, axis=axis, keepdims=True)
    numerator = tf.exp(masked_logits - normalizer)

    # Set unavailable products utility to 0
    numerator = tf.multiply(numerator, available_items_by_choice)
    # Sum of total available utilities
    denominator = tf.reduce_sum(numerator, axis=axis, keepdims=True)

    # Add 1 to the denominator to take into account the exit choice
    if normalize_exit:
        denominator += tf.exp(-normalizer)
    # Avoid division by 0 when only unavailable items have highest utilities
    elif eps:
        denominator += eps

    # Compute softmax
    return numerator / denominator


# @tf.function
def exponomial_probabilities_with_availabilities(
    items_logit_by_choice: tf.Tensor,
    available_items_by_choice: tf.Tensor | None = None,
    lam: float = 1.0,
) -> tf.Tensor:
    """Compute Exponomial choice probabilities with varying product availability.

    Parameters
    ----------
    items_logit_by_choice: tf.Tensor
        Tensor of shape (..., n) containing deterministic utilities.
    available_items_by_choice: tf.Tensor
        Optional boolean (or 0/1) Tensor of shape (..., n) indicating
        which products are available. If None, all products are available.
    lam: float
        Positive rate parameter (lam > 0).

    Returns
    -------
        Tensor of shape (..., n) of choice probabilities (0 for unavailable items).
    """
    utilities = tf.convert_to_tensor(items_logit_by_choice)
    dtype = utilities.dtype
    lam_t = tf.cast(lam, dtype)

    u_shape = tf.shape(utilities)
    n_items = u_shape[-1]

    if available_items_by_choice is None:
        mask = tf.ones_like(utilities, dtype=tf.bool)
    else:
        mask = tf.cast(available_items_by_choice, tf.bool)

    # Flatten leading batch dimensions to [B, n]
    u_flat = tf.reshape(utilities, [-1, n_items])
    mask_flat = tf.reshape(mask, [-1, n_items])
    mask_f = tf.cast(mask_flat, dtype)

    # 1. Push unavailable items to the end during sorting by setting utility to -inf
    neg_inf = (
        tf.constant(-1e30, dtype=dtype) if dtype != tf.float64 else tf.constant(-1e300, dtype=dtype)
    )
    u_for_sort = tf.where(mask_flat, u_flat, neg_inf)

    sort_idx = tf.argsort(u_for_sort, axis=-1, direction="DESCENDING")
    u_sorted = tf.gather(u_flat, sort_idx, batch_dims=1)
    m_sorted = tf.gather(mask_f, sort_idx, batch_dims=1)

    # 2. Compute effective ranks (1, 2, ..., |S|) and cumulative utility differences
    # Unavailable items have m_sorted == 0, so they don't advance the rank or sum
    eff_ranks = tf.cumsum(m_sorted, axis=-1)
    safe_ranks = tf.maximum(eff_ranks, tf.cast(1.0, dtype))  # avoid div-by-zero

    cum_u = tf.cumsum(u_sorted * m_sorted, axis=-1)
    exponent = -lam_t * (cum_u - eff_ranks * u_sorted)

    # Zero out G_i for unavailable items so they don't enter the tail sum
    g_values = tf.where(
        tf.cast(m_sorted, tf.bool),
        tf.exp(exponent) / safe_ranks,
        tf.cast(0.0, dtype),
    )

    # 3. Compute P_i = G_i - sum_{j > i} G_j / (eff_rank_j - 1)
    if n_items > 1:
        # Only items with eff_rank >= 2 contribute to the tail sum of higher-ranked items
        denom = tf.maximum(eff_ranks[:, 1:] - tf.cast(1.0, dtype), tf.cast(1.0, dtype))
        weighted_g_values = g_values[:, 1:] / denom
        rev_cumsum = tf.cumsum(weighted_g_values, axis=-1, reverse=True)
        tail_terms = tf.pad(rev_cumsum, paddings=[[0, 0], [0, 1]])
    else:
        tail_terms = tf.zeros_like(g_values)

    p_sorted = (g_values - tail_terms) * m_sorted

    # Clean up floating-point noise and renormalize over available items
    p_sorted = tf.maximum(p_sorted, tf.cast(0.0, dtype))
    row_sums = tf.reduce_sum(p_sorted, axis=-1, keepdims=True)
    # Safe division in case an entire row has 0 available products
    p_sorted = tf.math.divide_no_nan(p_sorted, row_sums)

    # 4. Restore original ordering
    inv_idx = tf.argsort(sort_idx, axis=-1, direction="ASCENDING")
    probs_flat = tf.gather(p_sorted, inv_idx, batch_dims=1)

    return tf.reshape(probs_flat, u_shape)
