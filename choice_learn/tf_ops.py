"""Diverse Tensorflow operations used in the ChoiceLearn library."""

import tensorflow as tf


def softmax_with_availabilities(
    items_logit_by_choice, available_items_by_choice, axis=-1, normalize_exit=False, eps=1e-5
):
    """Compute softmax probabilities from utilities.

    Takes into account availabilties (1 if the product is available, 0 otherwise) to set
    probabilities to 0 for unavailable products and to renormalize the probabilities of
    available products.

    Parameters
    ----------
    items_logit_by_choice : np.ndarray (n_choices, n_items)
        Utilities / Logits on which to compute the softmax
    available_items_by_choice : np.ndarray (n_choices, n_items)
        Matrix indicating the availabitily (1) or not (0) of the products
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
    tf.Tensor (n_chocies, n_items)
        Probabilities of each product for each choice computed from Logits
    """
    # Substract max utility to avoid overflow
    numerator = tf.exp(
        items_logit_by_choice - tf.reduce_max(items_logit_by_choice, axis=axis, keepdims=True)
    )
    # Set unavailable products utility to 0
    numerator = tf.multiply(numerator, available_items_by_choice)
    # Sum of total available utilities
    denominator = tf.reduce_sum(numerator, axis=axis, keepdims=True)
    # Add 1 to the denominator to take into account the exit choice
    if normalize_exit:
        denominator += 1
    # Avoir division by 0 when only unavailable items have highest utilities
    elif eps:
        denominator += eps

    # Compute softmax
    return numerator / denominator


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

        # Apply label clipping to avoid log(0) and such issues
        y_pred = tf.clip_by_value(y_pred, self.epsilon, 1.0 - self.epsilon)
        return -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=self.axis)
