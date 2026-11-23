"""Losses implemented using Tensorflow that are used in the Choice-Learn library."""

import tensorflow as tf


class CustomCategoricalCrossEntropy(tf.keras.losses.Loss):
    """Custom Categorical Cross Entropy Loss. Handles all options in one place.

    Follows structure of tf.keras.losses.CategoricalCrossentropy and its different possibilities.

    Parameters
    ----------
    from_logits : bool, optional
        Whether to compute the softmax from logits or probabilities, by default False
    sparse : bool, optional
        Whether the choice labels are integers(True) or one-hot encoded(False), by default False
    label_smoothing : float, optional
        Value of smoothing to apply to labels, by default 0.0
        Smoothing applied is 1.0 - label_smoothing for chosen item and
        label_smoothing / (num_items - 1) for all other items
    axis : int, optional
        Axis on which to compute the softmax. Used only if from_logits is True, by default -1
    epsilon : float, optional
        Value to apply to avoid computation issues in log, by default 1e-10
    """

    def __init__(
        self,
        from_logits=False,
        sparse=False,
        label_smoothing=0.0,
        axis=-1,
        epsilon=1e-10,
        name="eps_categorical_crossentropy",
        reduction="sum_over_batch_size",
    ):
        """Initialize function.

        Follows structure of tf.keras.losses.CategoricalCrossentropy.

        Parameters
        ----------
        from_logits : bool, optional
            Whether to compute the softmax from logits or probabilities, by default False
        sparse : bool, optional
            Whether the choice labels are integers(True) or one-hot encoded(False), by default False
        label_smoothing : float, optional
            Value of smoothing to apply to labels, by default 0.0
            Smoothing applied is 1.0 - label_smoothing for chosen item and
            label_smoothing / (num_items - 1) for all other items
        axis : int, optional
            Axis on which to compute the softmax. Used only if from_logits is True, by default -1
        epsilon : float, optional
            Value to apply to avoid computation issues in log, by default 1e-10
        name: str
            Name of the loss function - here to follow tf.keras.losses.Loss signature
        reduction:
            Reduction function - here to follow tf.keras.losses.Loss signature
        """
        super().__init__(reduction=reduction, name=name)
        self.label_smoothing = label_smoothing
        self.from_logits = from_logits
        self.sparse = sparse
        self.axis = axis
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        """Compute the cross-entropy loss.

        Parameters
        ----------
        y_true : np.ndarray | tf.Tenosr
            Ground truth labels
        y_pred : np.ndarray | tf.Tenosr
            Predicted labels

        Returns
        -------
        tf.Tensor
            Average Cross-Entropy loss
        """
        if self.from_logits:  # Apply softmax if utilities are given
            y_pred = tf.nn.softmax(y_pred, axis=self.axis)
        else:
            y_pred = tf.convert_to_tensor(y_pred)
        if self.sparse:  # Create OneHot labels if sparse labels are given
            y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[self.axis])
        else:
            y_true = tf.cast(y_true, y_pred.dtype)

        # Smooth labels
        if self.label_smoothing > 0:
            label_smoothing = tf.convert_to_tensor(self.label_smoothing, dtype=y_pred.dtype)
            num_classes = tf.cast(tf.shape(y_true)[self.axis], y_pred.dtype)
            y_true = y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

        # Apply label clipping to avoid log(0) and such issues
        y_pred = tf.clip_by_value(y_pred, self.epsilon, 1.0 - self.epsilon)
        return -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=self.axis)


class ExactCategoricalCrossEntropy(tf.keras.losses.Loss):
    """Custom Categorical Cross Entropy Loss. Handles all options in one place.

    Follows structure of tf.keras.losses.CategoricalCrossentropy and its different possibilities.

    Parameters
    ----------
    from_logits : bool, optional
        Whether to compute the softmax from logits or probabilities, by default False
    sparse : bool, optional
        Whether the choice labels are integers(True) or one-hot encoded(False), by default False
    label_smoothing : float, optional
        Value of smoothing to apply to labels, by default 0.0
        Smoothing applied is 1.0 - label_smoothing for chosen item and
        label_smoothing / (num_items - 1)for all other items
    axis : int, optional
        Axis on which to compute the softmax. Used only if from_logits is True, by default -1
    epsilon : float, optional
        Value to apply to avoid computation issues in log, by default 1e-10
    """

    def __init__(
        self,
        from_logits=False,
        sparse=False,
        axis=-1,
        epsilon=1e-35,
        name="exact_categorical_crossentropy",
        reduction="sum_over_batch_size",
    ):
        """Initialize function.

        Follows structure of tf.keras.losses.CategoricalCrossentropy.

        Parameters
        ----------
        from_logits : bool, optional
            Whether to compute the softmax from logits or probabilities, by default False
        sparse : bool, optional
            Whether the choice labels are integers(True) or one-hot encoded(False), by default False
        axis : int, optional
            Axis on which to compute the softmax. Used only if from_logits is True, by default -1
        epsilon : float, optional
            Value to apply to avoid computation issues in log, by default 1e-10
        name: str
            Name of the loss function - here to follow tf.keras.losses.Loss signature
        reduction:
            Reduction function - here to follow tf.keras.losses.Loss signature
        """
        super().__init__(reduction=reduction, name=name)
        self.from_logits = from_logits
        self.sparse = sparse
        self.axis = axis
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        """Compute the cross-entropy loss.

        Parameters
        ----------
        y_true : np.ndarray | tf.Tensor
            Ground truth labels
        y_pred : np.ndarray | tf.Tensor
            Predicted labels

        Returns
        -------
        tf.Tensor
            Average Cross-Entropy loss
        """
        if self.from_logits:  # Apply softmax if utilities are given
            y_pred = tf.nn.softmax(y_pred, axis=self.axis)
        else:
            y_pred = tf.convert_to_tensor(y_pred)
        if self.sparse:  # Create OneHot labels if sparse labels are given
            y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[self.axis])
        else:
            y_true = tf.cast(y_true, y_pred.dtype)

        # Apply label clipping to avoid log(0) and such issues
        y_pred = tf.clip_by_value(y_pred, self.epsilon, 1.0 - self.epsilon)
        return -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=self.axis)


class NoiseConstrastiveEstimation(tf.keras.losses.Loss):
    """Noise Contrastive Estimation Loss."""

    def __init__(
        self,
        epsilon=1e-10,
        name="noise_contrastive_estimation",
        reduction="sum_over_batch_size",
    ):
        """Initialize function.

        Follows structure of tf.keras.losses.CategoricalCrossentropy.

        Parameters
        ----------
        epsilon: float
            safety net for logarithm computation
        name: str
            Name of the loss function - here to follow tf.keras.losses.Loss signature
        reduction:
            Reduction function - here to follow tf.keras.losses.Loss signature
        """
        self.epsilon = epsilon
        super().__init__(reduction=reduction, name=name)

    def __call__(self, logit_true, logit_negative, freq_true, freq_negative, sample_weight=None):
        """Overide of the tf.keras __call__ function.

        Parameters
        ----------
        logit_true : np.ndarray | tf.Tensor
            Ground truth samplelogits
        logit_negative : np.ndarray | tf.Tensor
            Negative samples logits
        freq_true : np.ndarray | tf.Tensor
            Frequency / probabilty in data of true sample
        freq_negative : np.ndarray | tf.Tensor
            Frequency / probabilty in data of negative samples
        sample_weight : np.ndarray | tf.Tensor
            samplewise weights

        Returns
        -------
        tf.Tensor
            Average Cross-Entropy loss
        """
        losses = self.call(logit_true, logit_negative, freq_true, freq_negative)
        if sample_weight is not None:
            losses = losses * sample_weight
        return tf.reduce_sum(losses)

    def call(self, logit_true, logit_negative, freq_true, freq_negative):
        """Compute NCE loss.

        Parameters
        ----------
        logit_true : np.ndarray | tf.Tensor
            Ground truth samples logit
        logit_negative : np.ndarray | tf.Tensor
            Negative samples logit
        freq_true : np.ndarray | tf.Tensor
            Frequency / probabilty in data of true sample
        freq_negative : np.ndarray | tf.Tensor
            Frequency / probabilty in data of negative samples

        Returns
        -------
        tf.Tensor
            Average Cross-Entropy loss
        """
        n_negative_samples = tf.cast(tf.shape(logit_negative)[1], tf.float32)

        true_contribution = 1 / (1 + n_negative_samples * freq_true * tf.exp(-logit_true))
        noise_contribution = 1 - (
            1 / (1 + n_negative_samples * freq_negative * tf.exp(-logit_negative))
        )
        return -tf.math.log(true_contribution + self.epsilon) - tf.reduce_sum(
            tf.math.log(noise_contribution + self.epsilon), axis=-1
        )
