"""Estimation of probabilities from utilities with a normal noise."""

import tensorflow as tf
import tensorflow_probability as tfp


def one_hot_argmax(values, axis=-1):
    """Return the argmax, one-hot encoded with TensorFlow based functions.

    Parameters
    ----------
    values : np.ndarray, tf.Tensor
        values from which to compute argmax
    axis : int, optional
        axis on which to compute the argmax, by default -1

    Returns
    -------
    tf.Tensor
        one-hot encode argmax
    """
    return tf.one_hot(tf.math.argmax(values, axis=axis), depth=values.shape[1])


def noisy_argmax(
    utilities, n_noise_samples=128, axis=-1, noise_distribution="normal", binary=False
):
    """Argmax function with noise added to sample a probabilitic argmax.

    Parameters
    ----------
    utilities : np.ndarray, tf.Tensor
        values from which to compute argmax
    n_noise_samples: int, optional
        Number of noise samples to be drawn, by default 128
    axis : int, optional
        axis on which to compute the argmax, by default -1
    noise_distribution: str, optional
        distribution law for the noise, by default "normal"

    Returns
    -------
    tf.Tensor
        Probabilistic argmax
    """
    if binary and utilities.shape[1] > 2:
        raise ValueError(
            f"""Binary Classification is possible only if two utilities/probit
            values are give, got {utilities.shape[1]}"""
        )
    # Sampling z
    if noise_distribution == "normal":
        if binary:
            noise = tfp.distributions.Normal(0, 1).sample(
                (n_noise_samples, utilities.shape[0], utilities.shape[1] - 1)
            )
            noise = tf.concat([tf.zeros_like(noise), noise], axis=axis)
        else:
            noise = tfp.distributions.Normal(0, 1).sample(
                (n_noise_samples, utilities.shape[0], utilities.shape[1])
            )
        exp_noise = noise
    elif noise_distribution == "gumbel":
        noise = tfp.distributions.Gumbel(0, 1).sample(
            (n_noise_samples, utilities.shape[0], utilities.shape[1])
        )
        exp_noise = tf.ones((n_noise_samples, utilities.shape[0], utilities.shape[1])) - tf.exp(
            -noise
        )
    else:
        raise ValueError(f"Noise Distribution {noise_distribution} unknown.")

    # Experimental computation of E(y*(theta+Z)) with 1/N sum(y*(theta+z))
    noisy_baskets = tf.stack(
        [one_hot_argmax(noise[i] + utilities, axis=axis) for i in range(n_noise_samples)]
    )

    # Estimation of the jacobian from Expectation and nu(Z)
    jacobian_estimated = tf.stack(
        [
            tf.reduce_mean(
                tf.einsum(
                    "ki, kj -> kij",
                    tf.gather(noisy_baskets, k, axis=1),
                    tf.gather(exp_noise, k, axis=1),
                ),
                axis=0,
            )
            for k in range(utilities.shape[0])
        ]
    )
    # jacobian_estimated = tf.reduce_mean(tf.einsum("bki, bkj -> bkij",
    # noisy_baskets, exp_noise), axis=0)
    return tf.reduce_mean(noisy_baskets, axis=0), jacobian_estimated


@tf.custom_gradient
def normalomax_with_availabilities(
    utilities,
    available_items_by_choice=None,
    n_noise_samples=1024 * 16,
    noise_distribution="normal",
    axis=-1,
    binary=False,
):
    """Argmax function with noise added to sample a probabilitic argmax with availabilities.

    Operation can be used as a TF operation with defined gradients.

    Parameters
    ----------
    utilities : np.ndarray, tf.Tensor [batch_size, n_vals]
        values from which to compute argmax
    available_items_by_choice : np.ndarray, tf.Tensor [batch_size, n_vals]
        Matrix indicating the availability (1) or not (0) of the products
    n_noise_samples: int, optional
        Number of noise samples to be drawn, by default 1024*16
    axis : int, optional
        axis on which to compute the argmax, by default -1
    noise_distribution: str, optional
        distribution law for the noise, by default "normal"

    Returns
    -------
    tf.Tensor
        Probabilistic argmax
    tf.Tensor
        Corresponding gradients
    """
    if available_items_by_choice is not None:
        minimizer = (tf.reduce_max(utilities) - tf.reduce_min(utilities) + 1) * (
            tf.ones_like(available_items_by_choice) - available_items_by_choice
        )
        utilities = utilities - minimizer
    output, jacobian = noisy_argmax(
        utilities,
        n_noise_samples=n_noise_samples,
        noise_distribution=noise_distribution,
        axis=axis,
        binary=binary,
    )

    def grad(upstream):
        return tf.cast(tf.einsum("bij,bj->bi", jacobian, upstream), tf.float32), tf.zeros_like(
            available_items_by_choice
        )
        # return tf.cast(tf.einsum("bij,bj->bi", J, upstream), tf.float32)

    return output, grad
