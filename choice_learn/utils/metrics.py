"""Diverse Tensorflow customied metrics used in the ChoiceLearn library."""

import tensorflow as tf


class NegativeLogLikeliHood(tf.keras.metrics.Metric):
    """Compute Negative Loglikelihood.

    Parameters
    ----------
    tf : _type_
        _description_
    """

    def __init__(
        self,
        from_logits=False,
        sparse=False,
        epsilon=1e-10,
        name="negative_log_likelihood",
        axis=-1,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.nll = self.add_variable(shape=(), initializer="zeros", name="neg_ll")
        self.n_evals = self.add_variable(shape=(), initializer="zeros", name="n_evals")
        self.from_logits = from_logits
        self.sparse = sparse
        self.epsilon = epsilon
        self.axis = axis

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulate statistics for the metric.

        Parameters
        ----------
        y_true : _type_
            _description_
        y_pred : _type_
            _description_
        sample_weight : _type_, optional
            _description_, by default None
        """
        if self.axis == -1:
            if not len(tf.shape(y_pred)) == 2:
                raise ValueError(f"y_pred must be of shape size 2, is {len(tf.shape(y_pred))}.")
        if self.from_logits:  # Apply softmax if utilities are given
            y_pred = tf.nn.softmax(y_pred, axis=self.axis)
        else:
            y_pred = tf.convert_to_tensor(y_pred)
        if self.sparse:  # Create OneHot labels if sparse labels are given
            y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[self.axis])
        else:
            y_true = tf.cast(y_true, y_pred.dtype)

        # Apply label clipping to avoid log(0) and such issues
        y_pred = tf.clip_by_value(y_pred, self.epsilon, 1.0)
        if sample_weight is not None:
            pass
        self.nll.assign(self.nll - tf.reduce_sum(y_true * tf.math.log(y_pred)))
        self.n_evals.assign(self.n_evals + tf.shape(y_true)[0])

    def result(self):
        """Compute the current metric value.

        Returns
        -------
        _type_
            _description_
        """
        return self.nll / self.n_evals
