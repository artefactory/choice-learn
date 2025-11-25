"""Diverse Tensorflow customied metrics used in the ChoiceLearn library."""

import tensorflow as tf


class NegativeLogLikeliHood(tf.keras.metrics.Metric):
    """Compute Negative Loglikelihood.

    Parameters
    ----------
    from_logits : bool, optional
        Whether provided values are logits or probabilities, by default False
    sparse : bool, optional
        Whether y_true is given as an index or a one-hot, by default False
    epsilon : float, optional
        Lower bound for log(.), by default 1e-10
    name : str, optional
        Name of operation, by default "negative_log_likelihood"
    axis : int, optional
        axis on which to apply the metric, by default -1
    """

    def __init__(
        self,
        from_logits=False,
        sparse=False,
        average_on_batch=False,
        epsilon=1e-10,
        name="negative_log_likelihood",
        axis=-1,
        **kwargs,
    ):
        """Initialize metric.

        Parameters
        ----------
        from_logits : bool, optional
            Whether provided values are logits or probabilities, by default False
        sparse : bool, optional
            Whether y_true is given as an index or a one-hot, by default False
        epsilon : float, optional
            Lower bound for log(.), by default 1e-10
        average_on_batch: bool, optional
            Whether the metric should be averaged over each batch. Typically used to
            get metrics averaged by Trip, by default False
        name : str, optional
            Name of operation, by default "negative_log_likelihood"
        axis : int, optional
            axis on which to apply the metric, by default -1
        """
        super().__init__(name=name, **kwargs)
        self.nll = self.add_variable(shape=(), initializer="zeros", name="neg_ll")
        self.n_evals = self.add_variable(shape=(), initializer="zeros", name="n_evals")
        self.from_logits = from_logits
        self.sparse = sparse
        self.average_on_batch = average_on_batch
        self.epsilon = epsilon
        self.axis = axis

    def update_state(self, y_true, y_pred, batch=None, sample_weight=None):
        """Accumulate statistics for the metric.

        Parameters
        ----------
        y_true : np.ndarray
            Ground Truth value
        y_pred : np.ndarray
            Predicted values
        sample_weight : np.ndarray, optional
            sample wise weight, by default None
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
        if sample_weight is None:
            nll_value = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=self.axis)
        else:
            nll_value = -tf.reduce_sum(
                y_true * tf.math.log(y_pred) * tf.expand_dims(sample_weight, axis=-1),
                axis=self.axis,
            )

        if batch is not None and self.average_on_batch:
            for _, idx in zip(*tf.unique(batch)):
                self.nll.assign(self.nll + tf.reduce_mean(nll_value[idx]))
                self.n_evals.assign(self.n_evals + 1)
        else:
            self.nll.assign(self.nll + tf.reduce_sum(nll_value))
            if sample_weight is None:
                self.n_evals.assign(self.n_evals + tf.shape(y_true)[0])
            else:
                self.n_evals.assign(self.n_evals + tf.reduce_sum(sample_weight))

    def result(self):
        """Compute the current metric value.

        Returns
        -------
        float
            Negative Log Likelihood value
        """
        return tf.math.divide_no_nan(self.nll, self.n_evals)
