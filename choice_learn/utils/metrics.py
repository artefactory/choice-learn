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
        average_on_trip=False,
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
        average_on_trip: bool, optional
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
        self.average_on_trip = average_on_trip
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
        if batch is not None and self.average_on_trip:
            unique_trips, segment_ids = tf.unique(batch)
            trip_nlls = tf.math.unsorted_segment_mean(
                nll_value, segment_ids, tf.shape(unique_trips)[0]
            )
            self.nll.assign_add(tf.reduce_sum(trip_nlls))
            self.n_evals.assign_add(tf.cast(tf.shape(unique_trips)[0], self.n_evals.dtype))
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


class MRR(tf.keras.metrics.Metric):
    """Compute Mean Reciprocal Rank."""

    def __init__(
        self,
        average_on_trip=False,
        name="mean_reciprocal_rank",
        axis=-1,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.mrr = self.add_variable(shape=(), initializer="zeros", name="mrr")
        self.n_evals = self.add_variable(shape=(), initializer="zeros", name="n_evals")
        self.average_on_trip = average_on_trip
        self.axis = axis

    def update_state(
        self,
        y_true,
        y_pred,
        batch=None,
    ):
        """Accumulate statistics for the metric.

        Parameters
        ----------
        y_true : np.ndarray
            Ground Truth value
        y_pred : np.ndarray
            Predicted values
        """
        if self.axis == -1:
            if not len(tf.shape(y_pred)) == 2:
                raise ValueError(f"y_pred must be of shape size 2, is {len(tf.shape(y_pred))}.")
        else:
            y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, dtype=tf.int32)

        ranks = tf.argsort(tf.argsort(-y_pred, axis=1), axis=1) + 1  # Shape: (batch_size, n_items)
        item_batch_indices = tf.stack(
            [tf.range(len(y_true)), y_true], axis=1
        )  # Shape: (batch_size, 2)
        item_ranks = tf.gather_nd(ranks, item_batch_indices)  # Shape: (batch_size,)
        # mean_rank = tf.reduce_sum(tf.cast(1 / item_ranks, dtype=tf.float32), axis=self.axis)
        if batch is not None and self.average_on_trip:
            unique_trips, segment_ids = tf.unique(batch)
            reciprocal_ranks = 1.0 / tf.cast(item_ranks, dtype=tf.float32)
            trip_mrrs = tf.math.unsorted_segment_mean(
                reciprocal_ranks, segment_ids, tf.shape(unique_trips)[0]
            )
            self.mrr.assign_add(tf.reduce_sum(trip_mrrs))
            self.n_evals.assign_add(tf.cast(tf.shape(unique_trips)[0], self.n_evals.dtype))
        else:
            self.mrr.assign(self.mrr + tf.reduce_sum(tf.cast(1 / item_ranks, dtype=tf.float32)))
            self.n_evals.assign(self.n_evals + tf.shape(y_true)[0])

    def result(self):
        """Compute the current metric value.

        Returns
        -------
        float
            Negative Log Likelihood value
        """
        return tf.math.divide_no_nan(self.mrr, self.n_evals)


class HitRate(tf.keras.metrics.Metric):
    """Compute Hit Rate at k."""

    def __init__(
        self,
        average_on_trip=False,
        top_k: int = 10,
        name=None,
        axis=-1,
        **kwargs,
    ):
        if name is None:
            name = f"hit_rate_at_{top_k}"
        super().__init__(name=name, **kwargs)
        self.top_k = top_k
        self.hit_rate = self.add_variable(
            shape=(), initializer="zeros", name=f"hit_rate_at_{self.top_k}"
        )
        self.n_evals = self.add_variable(shape=(), initializer="zeros", name="n_evals")
        self.average_on_trip = average_on_trip
        self.axis = axis

    def update_state(self, y_true, y_pred, batch=None):
        """Accumulate statistics for the metric.

        Parameters
        ----------
        y_true : np.ndarray
            Ground Truth value
        y_pred : np.ndarray
            Predicted values
        """
        if self.axis == -1:
            if not len(tf.shape(y_pred)) == 2:
                raise ValueError(f"y_pred must be of shape size 2, is {len(tf.shape(y_pred))}.")
        else:
            y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, dtype=tf.int32)

        top_k_indices = tf.math.top_k(y_pred, k=self.top_k).indices  # Shape: (batch_size, top_k)
        hits_per_batch = tf.reduce_any(
            tf.equal(
                tf.cast(top_k_indices, tf.int32),
                tf.cast(tf.expand_dims(y_true, axis=1), tf.int32),
            ),
            axis=1,
        )
        hits = tf.cast(hits_per_batch, tf.float32)
        if batch is not None and self.average_on_trip:
            unique_trips, segment_ids = tf.unique(batch)
            trip_means = tf.math.unsorted_segment_mean(hits, segment_ids, tf.shape(unique_trips)[0])
            self.hit_rate.assign_add(tf.reduce_sum(trip_means))
            self.n_evals.assign_add(tf.cast(tf.shape(unique_trips)[0], self.n_evals.dtype))
        else:
            self.hit_rate.assign(self.hit_rate + tf.reduce_sum(hits))
            self.n_evals.assign(self.n_evals + tf.shape(y_true)[0])

    def result(self):
        """Compute the current metric value.

        Returns
        -------
        float
            Negative Log Likelihood value
        """
        return tf.math.divide_no_nan(self.hit_rate, self.n_evals)
