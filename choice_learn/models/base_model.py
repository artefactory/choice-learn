"""Base class for choice models."""
import json
import os
import time
from abc import abstractmethod
from pathlib import Path

import numpy as np
import tensorflow as tf
import tqdm

from choice_learn.tf_ops import (
    CustomCategoricalCrossEntropy,
    custom_softmax,
)


class ChoiceModel(object):
    """Base class for choice models."""

    def __init__(
        self,
        label_smoothing=0.0,
        normalize_non_buy=False,
        optimizer="Adam",
        callbacks=None,
        lr=0.001,
    ):
        """Instantiates the ChoiceModel.

        Parameters
        ----------
        label_smoothing : float, optional
            Whether (then is ]O, 1[ value) or not (then can be None or 0) to use label smoothing,
        during training, by default 0.0
            by default None. Label smoothing is applied to LogLikelihood loss.
        normalize_non_buy : bool, optional
            Whether or not to add a normalization (then U=1) with the exit option in probabilites
            normalization,by default True
        callbacks : list of tf.kera callbacks, optional
            List of callbacks to add to model.fit, by default None and only add History
        """
        self.is_fitted = False
        self.normalize_non_buy = normalize_non_buy
        self.label_smoothing = label_smoothing
        self.stop_training = False

        # self.loss = tf.keras.losses.CategoricalCrossentropy(
        #     from_logits=False, label_smoothing=self.label_smoothing
        # )
        self.loss = CustomCategoricalCrossEntropy(
            from_logits=False, label_smoothing=self.label_smoothing
        )
        self.callbacks = tf.keras.callbacks.CallbackList(callbacks, add_history=True, model=None)
        self.callbacks.set_model(self)

        # Was originally in BaseMNL, moved here.
        if optimizer.lower() == "adam":
            self.optimizer = tf.keras.optimizers.Adam(lr)
        elif optimizer.lower() == "sgd":
            self.optimizer = tf.keras.optimizers.SGD(lr)
        elif optimizer.lower() == "adamax":
            self.optimizer = tf.keras.optimizers.Adamax(lr)
        elif optimizer.lower() == "lbfgs" or optimizer.lower() == "l-bfgs":
            print("Using L-BFGS optimizer, setting up .fit() function")
            self.fit = self._fit_with_lbfgs
        else:
            print(f"Optimizer {optimizer} not implemnted, switching for default Adam")
            self.optimizer = tf.keras.optimizers.Adam(lr)

    @abstractmethod
    def compute_utility(
        self, items_batch, sessions_batch, sessions_items_batch, availabilities_batch, choices_batch
    ):
        """Method that defines how the model computes the utility of a product.

        MUST be implemented in children classe
        For simpler use-cases this is the only method to be user-defined.

        Parameters
        ----------
        items_batch : tuple of np.ndarray (items_features)
            Fixed-Item-Features: formatting from ChoiceDataset: a matrix representing the products
            constant/fixed features.
            Shape must be (n_items, n_items_features)
        sessions_batch : tuple of np.ndarray (sessions_features)
            Time-Features
            Shape must be (n_sessions, n_sessions_features)
        sessions_items_batch : tuple of np.ndarray (sessions_items_features)
            Time-Item-Features
            Shape must be (n_sessions, n_sessions_items_features)
        availabilities_batch : np.ndarray
            Availabilities (sessions_items_availabilities)
            Shape must be (n_sessions, n_items)
        choices_batch : np.ndarray
            Choices
            Shape must be (n_sessions, )

        Returns:
        --------
        np.ndarray
            Utility of each product for each session.
            Shape must be (n_sessions, n_items)
        """
        # To be implemented in children classes
        # Can be numpy or tensorflow based
        return

    @tf.function
    def train_step(
        self,
        items_batch,
        sessions_batch,
        sessions_items_batch,
        availabilities_batch,
        choices_batch,
        sample_weight=None,
    ):
        """Function that represents one training step (= one gradient descent step) of the model.

        Parameters
        ----------
        items_batch : tuple of np.ndarray (items_features)
            Fixed-Item-Features: formatting from ChoiceDataset: a matrix representing the products
            constant/fixed features.
        sessions_batch : tuple of np.ndarray (sessions_features)
            Time-Features
        sessions_items_batch : tuple of np.ndarray (sessions_items_features)
            Time-Item-Features
        availabilities_batch : np.ndarray
            Availabilities (sessions_items_availabilities)
        choices_batch : np.ndarray
            Choices
        sample_weight : np.ndarray, optional
            List samples weights to apply during the gradient descent to the batch elements,
            by default None

        Returns:
        --------
        tf.Tensor
            Value of NegativeLogLikelihood loss for the batch
        """
        with tf.GradientTape() as tape:
            all_u = self.compute_utility(
                items_batch,
                sessions_batch,
                sessions_items_batch,
                availabilities_batch,
                choices_batch,
            )
            """
            all_u = tf.math.exp(all_u)

            # Assortment(t) Utility
            norms = tf.reduce_sum(tf.multiply(all_u, ia_batch), axis=1)
            if self.normalize_non_buy:
                norms += 1
            # Probabilities
            final_utilities = tf.divide(
                all_u,
                tf.repeat(tf.expand_dims(norms, 1), fif_batch[0].shape[0], axis=1),
            )
            # Probabilities of selected product
            available_utilities = tf.gather_nd(indices=choices_nd, params=final_utilities)
            """
            # probabilities = availability_softmax(all_u, availabilities_batch, axis=-1)
            probabilities = custom_softmax(
                all_u, availabilities_batch, normalize_exit=self.normalize_non_buy, axis=-1
            )
            # Negative Log-Likelihood
            neg_loglikelihood = self.loss(
                y_pred=probabilities,
                y_true=tf.one_hot(choices_batch, depth=probabilities.shape[1]),
                sample_weight=sample_weight,
            )
            """
            if sample_weight is not None:
                neg_loglikelihood = -tf.reduce_sum(
                    tf.math.log(available_utilities + 1e-10) * sample_weight
                )
            else:
                neg_loglikelihood = -tf.reduce_sum(tf.math.log(available_utilities + 1e-10))
            """
        grads = tape.gradient(neg_loglikelihood, self.weights)
        self.optimizer.apply_gradients(zip(grads, self.weights))
        return neg_loglikelihood

    def fit(
        self, choice_dataset, n_epochs, batch_size, sample_weight=None, val_dataset=None, verbose=0
    ):
        """Method to train the model with a ChoiceDataset.

        Parameters
        ----------
        choice_dataset : ChoiceDataset
            _description_
        n_epochs : int
            Number of epochs
        batch_size : int
            Batch size
        sample_weight : np.ndarray, optional
            Sample weights to apply, by default None
        val_dataset : ChoiceDataset, optional
            Test ChoiceDataset to evaluate performances on test at each epoch, by default None
        verbose : int, optional
            print level, for debugging, by default 0

        Returns:
        --------
        dict:
            Different metrics values over epochs.
        """
        losses_history = {"train_loss": []}
        t_range = tqdm.trange(n_epochs, position=0)

        self.callbacks.on_train_begin()

        # Iterate of epochs
        for epoch_nb in t_range:
            self.callbacks.on_epoch_begin(epoch_nb)
            t_start = time.time()
            train_logs = {"train_loss": []}
            val_logs = {"val_loss": []}
            epoch_losses = []

            if sample_weight is not None:
                if verbose > 0:
                    inner_range = tqdm.tqdm(
                        choice_dataset.iter_batch(
                            shuffle=True, sample_weight=sample_weight, batch_size=batch_size
                        ),
                        total=int(len(choice_dataset) / np.max([1, batch_size])),
                        position=1,
                        leave=False,
                    )
                else:
                    inner_range = choice_dataset.iter_batch(
                        shuffle=True, sample_weight=sample_weight, batch_size=batch_size
                    )

                for batch_nb, (
                    (
                        items_batch,
                        sessions_batch,
                        sessions_items_batch,
                        availabilities_batch,
                        choices_batch,
                    ),
                    weight_batch,
                ) in enumerate(inner_range):
                    self.callbacks.on_train_batch_begin(batch_nb)

                    neg_loglikelihood = self.train_step(
                        items_batch,
                        sessions_batch,
                        sessions_items_batch,
                        availabilities_batch,
                        choices_batch,
                        sample_weight=weight_batch,
                    )

                    train_logs["train_loss"].append(neg_loglikelihood)
                    temps_logs = {k: tf.reduce_mean(v) for k, v in train_logs.items()}
                    self.callbacks.on_train_batch_end(batch_nb, logs=temps_logs)

                    # Optimization Steps
                    epoch_losses.append(neg_loglikelihood)

            # In this case we do not need to batch the sample_weights
            else:
                if verbose > 0:
                    inner_range = tqdm.tqdm(
                        choice_dataset.iter_batch(shuffle=True, batch_size=batch_size),
                        total=int(len(choice_dataset) / np.max([batch_size, 1])),
                        position=1,
                        leave=False,
                    )
                else:
                    inner_range = choice_dataset.iter_batch(shuffle=True, batch_size=batch_size)
                for batch_nb, (
                    items_batch,
                    sessions_batch,
                    sessions_items_batch,
                    availabilities_batch,
                    choices_batch,
                ) in enumerate(inner_range):
                    self.callbacks.on_train_batch_begin(batch_nb)
                    neg_loglikelihood = self.train_step(
                        items_batch,
                        sessions_batch,
                        sessions_items_batch,
                        availabilities_batch,
                        choices_batch,
                    )
                    train_logs["train_loss"].append(neg_loglikelihood)
                    temps_logs = {k: tf.reduce_mean(v) for k, v in train_logs.items()}
                    self.callbacks.on_train_batch_end(batch_nb, logs=temps_logs)

                    # Optimization Steps
                    epoch_losses.append(neg_loglikelihood)

            # Take into account last batch that may have a differnt length into account for
            # the computation of the epoch loss.
            if batch_size != -1:
                last_batch_size = availabilities_batch.shape[0]
                coefficients = tf.concat(
                    [tf.ones(len(epoch_losses) - 1) * batch_size, [last_batch_size]], axis=0
                )
                epoch_lossses = tf.multiply(epoch_losses, coefficients)
                epoch_loss = tf.reduce_sum(epoch_lossses) / len(choice_dataset)
            else:
                epoch_loss = tf.reduce_mean(epoch_losses)
            losses_history["train_loss"].append(epoch_loss)
            desc = f"Epoch {epoch_nb} Train Loss {losses_history['train_loss'][-1].numpy()}"
            if verbose > 1:
                print(
                    f"Loop {epoch_nb} Time",
                    time.time() - t_start,
                    "Loss:",
                    tf.reduce_sum(epoch_losses).numpy(),
                )

            # Test on val_dataset if provided
            if val_dataset is not None:
                test_losses = []
                for batch_nb, (
                    items_batch,
                    sessions_batch,
                    sessions_items_batch,
                    availabilities_batch,
                    choices_batch,
                ) in enumerate(val_dataset.iter_batch(shuffle=False, batch_size=batch_size)):
                    self.callbacks.on_batch_begin(batch_nb)
                    self.callbacks.on_test_batch_begin(batch_nb)
                    test_losses.append(
                        self.batch_predict(
                            items_batch,
                            sessions_batch,
                            sessions_items_batch,
                            availabilities_batch,
                            choices_batch,
                        )[0]
                    )
                    val_logs["val_loss"].append(test_losses[-1])
                    temps_logs = {k: tf.reduce_mean(v) for k, v in val_logs.items()}
                    self.callbacks.on_test_batch_end(batch_nb, logs=temps_logs)
                test_loss = tf.reduce_mean(test_losses)
                if verbose > 1:
                    print("Test Negative-LogLikelihood:", test_loss.numpy())
                    desc += f", Test Loss {test_loss.numpy()}"
                losses_history["test_loss"] = losses_history.get("test_loss", []) + [
                    test_loss.numpy()
                ]
                train_logs = {**train_logs, **val_logs}

            temps_logs = {k: tf.reduce_mean(v) for k, v in train_logs.items()}
            self.callbacks.on_epoch_end(epoch_nb, logs=temps_logs)
            if self.stop_training:
                print("Early Stopping taking effect")
                break
            if verbose > 0:
                t_range.set_description(desc)
                t_range.refresh()

        temps_logs = {k: tf.reduce_mean(v) for k, v in train_logs.items()}
        self.callbacks.on_train_end(logs=temps_logs)
        return losses_history

    @tf.function
    def batch_predict(
        self,
        items_batch,
        sessions_batch,
        sessions_items_batch,
        availabilities_batch,
        choices_batch,
        sample_weight=None,
    ):
        """Function that represents one prediction (Probas + Loss) for one batch of a ChoiceDataset.

        Parameters
        ----------
        items_batch : tuple of np.ndarray (items_features)
            Fixed-Item-Features: formatting from ChoiceDataset: a matrix representing the products
            constant features.
        sessions_batch : tuple of np.ndarray (sessions_features)
            Time-Features
        sessions_items_batch : tuple of np.ndarray (sessions_items_features)
            Time-Item-Features
        availabilities_batch : np.ndarray
            Availabilities (sessions_items_availabilities)
        choices_batch : np.ndarray
            Choices
        sample_weight : np.ndarray, optional
            List samples weights to apply during the gradient descent to the batch elements,
            by default None

        Returns:
        --------
        tf.Tensor (1, )
            Value of NegativeLogLikelihood loss for the batch
        tf.Tensor (batch_size, n_items)
            Probabilities for each product to be chosen for each session
        """
        # Compute utilities from features
        utilities = self.compute_utility(
            items_batch, sessions_batch, sessions_items_batch, availabilities_batch, choices_batch
        )
        # Compute probabilities from utilities & availabilties
        # probabilities = availability_softmax(utilities, availabilities_batch, axis=-1)
        probabilities = custom_softmax(
            utilities, availabilities_batch, normalize_exit=self.normalize_non_buy, axis=-1
        )

        # Compute loss from probabilities & actual choices
        # batch_loss = self.loss(probabilities, c_batch, sample_weight=sample_weight)
        batch_loss = self.loss(
            y_pred=probabilities,
            y_true=tf.one_hot(choices_batch, depth=probabilities.shape[1]),
            sample_weight=sample_weight,
        )
        return batch_loss, probabilities

    def save_model(self, path):
        """Method to save the different models on disk.

        Parameters
        ----------
        path : str
            path to the folder where to save the model
        """
        if not os.exists(path):
            Path(path).mkdir(parents=True)

        for i, weight in enumerate(self.weights):
            tf.keras.savedmodel.save(Path(path) / f"weight_{i}")

        # To improve for non-string attributes
        params = self.__dict__
        json.dump(Path(path) / "params.json", params)

        # Save optimizer state

    @classmethod
    def load_model(cls, path):
        """Method to load a ChoiceModel previously saved with save_model().

        Parameters
        ----------
        path : str
            path to the folder where the saved model files are

        Returns:
        --------
        ChoiceModel
            Loaded ChoiceModel
        """
        obj = cls()
        obj.weights = []
        i = 0
        weight_path = f"weight_{i}"
        while weight_path in os.listdir(path):
            obj.weights.append(tf.keras.load_model.load(Path(path) / weight_path))
            i += 1
            weight_path = f"weight_{i}"

        # To improve for non string attributes
        params = json.load(Path(path) / "params.json")
        for k, v in params.items():
            setattr(obj, k, v)

        # Load optimizer step
        return cls

    def predict_probas(self, choice_dataset):
        """Predicts the choice probabilities for each session and each product of a ChoiceDataset.

        Parameters
        ----------
        choice_dataset : ChoiceDataset
            Dataset on which to apply to prediction

        Returns:
        --------
        np.ndarray (n_sessions, n_items)
            Choice probabilties for each session and each product
        """
        stacked_probabilities = []
        for (
            items_batch,
            sessions_batch,
            sessions_items_batch,
            availabilities_batch,
            choices_batch,
        ) in choice_dataset.iter_batch():
            _, probabilities = self.batch_predict(
                items_batch,
                sessions_batch,
                sessions_items_batch,
                availabilities_batch,
                choices_batch,
            )
            stacked_probabilities.append(probabilities)

        return tf.concat(stacked_probabilities, axis=0)

    def evaluate(self, choice_dataset, batch_size=None):
        """Evaluates the model for each session and each product of a ChoiceDataset.

        Predicts the probabilities according to the model and computes the Negative-Log-Likelihood
        loss from the actual choices.

        Parameters
        ----------
        choice_dataset : ChoiceDataset
            Dataset on which to apply to prediction

        Returns:
        --------
        np.ndarray (n_sessions, n_items)
            Choice probabilties for each session and each product
        """
        if batch_size is None:
            batch_size = choice_dataset.batch_size
        batch_losses = []
        for (
            items_batch,
            sessions_batch,
            sessions_items_batch,
            availabilities_batch,
            choices_batch,
        ) in choice_dataset.iter_batch(batch_size=batch_size):
            loss, _ = self.batch_predict(
                items_batch,
                sessions_batch,
                sessions_items_batch,
                availabilities_batch,
                choices_batch,
            )
            batch_losses.append(loss)
        if batch_size != -1:
            last_batch_size = availabilities_batch.shape[0]
            coefficients = tf.concat(
                [tf.ones(len(batch_losses) - 1) * batch_size, [last_batch_size]], axis=0
            )
            batch_losses = tf.multiply(batch_losses, coefficients)
            batch_loss = tf.reduce_sum(batch_losses) / len(choice_dataset)
        else:
            batch_loss = tf.reduce_mean(batch_losses)
        return batch_loss

    def _lbfgs_train_step(self, dataset):
        """A factory to create a function required by tfp.optimizer.lbfgs_minimize.

        Parameters
        ----------
        dataset: ChoiceDataset
            Dataset on which to estimate the paramters.

        Returns:
        --------
        function
            with the signature:
                loss_value, gradients = f(model_parameters).
        """
        # obtain the shapes of all trainable parameters in the model
        shapes = tf.shape_n(self.weights)
        n_tensors = len(shapes)

        # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
        # prepare required information first
        count = 0
        idx = []  # stitch indices
        part = []  # partition indices

        for i, shape in enumerate(shapes):
            n = np.product(shape)
            idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
            part.extend([i] * n)
            count += n

        part = tf.constant(part)

        @tf.function
        def assign_new_model_parameters(params_1d):
            """A function updating the model's parameters with a 1D tf.Tensor.

            Pararmeters
            -----------
            params_1d: tf.Tensor
                a 1D tf.Tensor representing the model's trainable parameters.
            """
            params = tf.dynamic_partition(params_1d, part, n_tensors)
            for i, (shape, param) in enumerate(zip(shapes, params)):
                self.weights[i].assign(tf.reshape(param, shape))

        # now create a function that will be returned by this factory
        @tf.function
        def f(params_1d):
            """A function that can be used by tfp.optimizer.lbfgs_minimize.

            This function is created by function_factory.

            Parameters
            ----------
            params_1d: tf.Tensor
                a 1D tf.Tensor.

            Returns:
            --------
            tf.Tensor
                A scalar loss and the gradients w.r.t. the `params_1d`.
            tf.Tensor
                A 1D tf.Tensor representing the gradients w.r.t. the `params_1d`.
            """
            # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
            with tf.GradientTape() as tape:
                # update the parameters in the model
                assign_new_model_parameters(params_1d)
                # calculate the loss
                loss_value = self.evaluate(dataset, batch_size=-1)

            # calculate gradients and convert to 1D tf.Tensor
            grads = tape.gradient(loss_value, self.weights)
            grads = tf.dynamic_stitch(idx, grads)

            # print out iteration & loss
            f.iter.assign_add(1)

            # store loss value so we can retrieve later
            tf.py_function(f.history.append, inp=[loss_value], Tout=[])

            return loss_value, grads

        # store these information as members so we can use them outside the scope
        f.iter = tf.Variable(0)
        f.idx = idx
        f.part = part
        f.shapes = shapes
        f.assign_new_model_parameters = assign_new_model_parameters
        f.history = []
        return f

    def _fit_with_lbfgs(self, dataset, n_epochs, tolerance=1e-8):
        """Fit function for L-BFGS optimizer.

        Replaces the .fit method when the optimizer is set to L-BFGS.

        Parameters
        ----------
        dataset : ChoiceDataset
            Dataset to be used for coefficients estimations
        n_epochs : int
            Maximum number of epochs allowed to reach minimum
        tolerance : float, optional
            Maximum tolerance accepted, by default 1e-8

        Returns:
        --------
        dict
            Fit history
        """
        # Only import tensorflow_probability if LBFGS optimizer is used, avoid unnecessary
        # dependency
        import tensorflow_probability as tfp

        func = self._lbfgs_train_step(dataset)

        # convert initial model parameters to a 1D tf.Tensor
        init_params = tf.dynamic_stitch(func.idx, self.weights)

        # train the model with L-BFGS solver
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=func,
            initial_position=init_params,
            max_iterations=n_epochs,
            tolerance=tolerance,
            f_absolute_tolerance=-1,
            f_relative_tolerance=-1,
        )

        # after training, the final optimized parameters are still in results.position
        # so we have to manually put them back to the model
        func.assign_new_model_parameters(results.position)
        print("L-BFGS Opimization finished:")
        print("---------------------------------------------------------------")
        print("Number of iterations:", results[2].numpy())
        print("Algorithm converged before reaching max iterations:", results[0].numpy())
        return func.history


class RandomChoiceModel(ChoiceModel):
    """Dumb model that randomly attributes utilities to products."""

    def __init__(self, **kwargs):
        """Initialization of the model."""
        super().__init__(**kwargs)

    def compute_utility(
        self, items_batch, sessions_batch, sessions_items_batch, availabilities_batch, choices_batch
    ):
        """Computes the random utility for each product of each session.

        Parameters
        ----------
        items_batch : tuple of np.ndarray (items_features)
            Fixed-Item-Features: formatting from ChoiceDataset: a matrix representing the products
            constant/fixed features.
        sessions_batch : tuple of np.ndarray (sessions_features)
            Time-Features
        sessions_items_batch : tuple of np.ndarray (sessions_items_features)
            Time-Item-Features
        availabilities_batch : np.ndarray
            Availabilities (sessions_items_availabilities)
        choices_batch : np.ndarray
            Choices

        Returns:
        --------
        tf.Tensor
            (n_sessions, n_items) matrix of random utilities
        """
        # In order to avoid unused arguments warnings
        del items_batch, sessions_batch, availabilities_batch, choices_batch
        return np.squeeze(np.random.uniform(shape=(sessions_items_batch.shape), minval=0, maxval=1))

    def fit(**kwargs):
        """Make sure that nothing happens during .fit."""
        del kwargs
        return {}


class DistribMimickingModel(ChoiceModel):
    """Dumb class model that mimicks the probabilities.

    It stores the encountered in the train datasets and always returns them
    """

    def __init__(self, **kwargs):
        """Initialization of the model."""
        super().__init__(**kwargs)
        self.weights = []

    def fit(self, choice_dataset, **kwargs):
        """Computes the choice frequency of each product and defines it as choice probabilities."""
        del kwargs
        choices = choice_dataset.choices
        for i in range(choice_dataset.get_num_items()):
            self.weights.append(tf.reduce_sum(tf.cast(choices == i, tf.float32)))
        self.weights = tf.stack(self.weights) / len(choices)

    def compute_utility(
        self, items_batch, sessions_batch, sessions_items_batch, availabilities_batch, choices_batch
    ):
        """Returns utility that is fixed. U = log(P).

        Parameters
        ----------
        items_batch : tuple of np.ndarray (items_features)
            Fixed-Item-Features: formatting from ChoiceDataset: a matrix representing the products
            constant/fixed features.
        sessions_batch : tuple of np.ndarray (sessions_features)
            Time-Features
        sessions_items_batch : tuple of np.ndarray (sessions_items_features)
            Time-Item-Features
        availabilities_batch : np.ndarray
            Availabilities (sessions_items_availabilities)
        choices_batch : np.ndarray
            Choices

        Returns:
        --------
        np.ndarray (n_sessions, n_items)
            Utilities

        Raises:
        -------
        ValueError
            If the model has not been fitted cannot evaluate the utility
        """
        # In order to avoid unused arguments warnings
        del items_batch, sessions_batch, sessions_items_batch, availabilities_batch
        if self.weights is None:
            raise ValueError("Model not fitted")
        return np.stack([np.log(self.weights.numpy())] * len(choices_batch), axis=0)
