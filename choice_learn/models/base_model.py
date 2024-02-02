"""Base class for choice models."""
import json
import os
import time
from abc import abstractmethod
from pathlib import Path

import numpy as np
import tensorflow as tf
import tqdm

import choice_learn.tf_ops as tf_ops


class ChoiceModel(object):
    """Base class for choice models."""

    def __init__(
        self,
        label_smoothing=0.0,
        normalize_non_buy=False,
        optimizer="Adam",
        callbacks=None,
        lr=0.001,
        epochs=1,
        batch_size=32,
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

        # Loss function wrapping tf.keras.losses.CategoricalCrossEntropy
        # with smoothing and normalization options
        self.loss = tf_ops.CustomCategoricalCrossEntropy(
            from_logits=False, label_smoothing=self.label_smoothing
        )
        self.callbacks = tf.keras.callbacks.CallbackList(callbacks, add_history=True, model=None)
        self.callbacks.set_model(self)

        # Was originally in BaseMNL, moved here.
        self.optimizer_name = optimizer
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

        self.epochs = epochs
        self.batch_size = batch_size

    @abstractmethod
    def compute_batch_utility(
        self,
        fixed_items_features,
        contexts_features,
        contexts_items_features,
        contexts_items_availabilities,
        choices,
    ):
        """Method that defines how the model computes the utility of a product.

        MUST be implemented in children classe
        For simpler use-cases this is the only method to be user-defined.

        Parameters
        ----------
        fixed_items_features : tuple of np.ndarray
            Fixed-Item-Features: formatting from ChoiceDataset: a matrix representing the products
            constant/fixed features.
            Shape must be (n_items, n_items_features)
        contexts_features : tuple of np.ndarray (contexts_features)
            a batch of contexts features
            Shape must be (n_contexts, n_contexts_features)
        contexts_items_features : tuple of np.ndarray (contexts_items_features)
            a batch of contexts items features
            Shape must be (n_contexts, n_contexts_items_features)
        contexts_items_availabilities : np.ndarray
            A batch of contexts items availabilities
            Shape must be (n_contexts, n_items)
        choices_batch : np.ndarray
            Choices
            Shape must be (n_contexts, )

        Returns:
        --------
        np.ndarray
            Utility of each product for each context.
            Shape must be (n_contexts, n_items)
        """
        # To be implemented in children classes
        # Can be numpy or tensorflow based
        return

    @tf.function
    def train_step(
        self,
        fixed_items_features,
        contexts_features,
        contexts_items_features,
        contexts_items_availabilities,
        choices,
        sample_weight=None,
    ):
        """Function that represents one training step (= one gradient descent step) of the model.

        Parameters
        ----------
        fixed_items_features : tuple of np.ndarray
            Fixed-Item-Features: formatting from ChoiceDataset: a matrix representing the products
            constant/fixed features.
            Shape must be (n_items, n_items_features)
        contexts_features : tuple of np.ndarray (contexts_features)
            a batch of contexts features
            Shape must be (n_contexts, n_contexts_features)
        contexts_items_features : tuple of np.ndarray (contexts_items_features)
            a batch of contexts items features
            Shape must be (n_contexts, n_contexts_items_features)
        contexts_items_availabilities : np.ndarray
            A batch of contexts items availabilities
            Shape must be (n_contexts, n_items)
        choices_batch : np.ndarray
            Choices
            Shape must be (n_contexts, )
        sample_weight : np.ndarray, optional
            List samples weights to apply during the gradient descent to the batch elements,
            by default None

        Returns:
        --------
        tf.Tensor
            Value of NegativeLogLikelihood loss for the batch
        """
        with tf.GradientTape() as tape:
            utilities = self.compute_batch_utility(
                fixed_items_features=fixed_items_features,
                contexts_features=contexts_features,
                contexts_items_features=contexts_items_features,
                contexts_items_availabilities=contexts_items_availabilities,
                choices=choices,
            )

            probabilities = tf_ops.softmax_with_availabilities(
                contexts_items_logits=utilities,
                contexts_items_availabilities=contexts_items_availabilities,
                normalize_exit=self.normalize_non_buy,
                axis=-1,
            )
            # Negative Log-Likelihood
            neg_loglikelihood = self.loss(
                y_pred=probabilities,
                y_true=tf.one_hot(choices, depth=probabilities.shape[1]),
                sample_weight=sample_weight,
            )

        grads = tape.gradient(neg_loglikelihood, self.weights)
        self.optimizer.apply_gradients(zip(grads, self.weights))
        return neg_loglikelihood

    def fit(
        self,
        choice_dataset,
        sample_weight=None,
        val_dataset=None,
        verbose=0,
        epochs=None,
        batch_size=None,
    ):
        """Method to train the model with a ChoiceDataset.

        Parameters
        ----------
        choice_dataset : ChoiceDataset
            Input data in the form of a ChoiceDataset
        sample_weight : np.ndarray, optional
            Sample weights to apply, by default None
        val_dataset : ChoiceDataset, optional
            Test ChoiceDataset to evaluate performances on test at each epoch, by default None
        verbose : int, optional
            print level, for debugging, by default 0
        epochs : int, optional
            Number of epochs, default is None, meaning we use self.epochs
        batch_size : int, optional
            Batch size, default is None, meaning we use self.batch_size

        Returns:
        --------
        dict:
            Different metrics values over epochs.
        """
        if hasattr(self, "instantiated"):
            if not self.instantiated:
                raise ValueError("Model not instantiated. Please call .instantiate() first.")
        if epochs is None:
            epochs = self.epochs
        if batch_size is None:
            batch_size = self.batch_size

        losses_history = {"train_loss": []}
        t_range = tqdm.trange(epochs, position=0)

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
                        contexts_batch,
                        contexts_items_batch,
                        availabilities_batch,
                        choices_batch,
                    ),
                    weight_batch,
                ) in enumerate(inner_range):
                    self.callbacks.on_train_batch_begin(batch_nb)

                    neg_loglikelihood = self.train_step(
                        items_batch,
                        contexts_batch,
                        contexts_items_batch,
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
                    contexts_batch,
                    contexts_items_batch,
                    availabilities_batch,
                    choices_batch,
                ) in enumerate(inner_range):
                    self.callbacks.on_train_batch_begin(batch_nb)
                    neg_loglikelihood = self.train_step(
                        items_batch,
                        contexts_batch,
                        contexts_items_batch,
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
                    contexts_batch,
                    contexts_items_batch,
                    availabilities_batch,
                    choices_batch,
                ) in enumerate(val_dataset.iter_batch(shuffle=False, batch_size=batch_size)):
                    self.callbacks.on_batch_begin(batch_nb)
                    self.callbacks.on_test_batch_begin(batch_nb)
                    test_losses.append(
                        self.batch_predict(
                            items_batch,
                            contexts_batch,
                            contexts_items_batch,
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
        fixed_items_features,
        contexts_features,
        contexts_items_features,
        contexts_items_availabilities,
        choices,
        sample_weight=None,
    ):
        """Function that represents one prediction (Probas + Loss) for one batch of a ChoiceDataset.

        Parameters
        ----------
        fixed_items_features : tuple of np.ndarray
            Fixed-Item-Features: formatting from ChoiceDataset: a matrix representing the products
            constant/fixed features.
            Shape must be (n_items, n_items_features)
        contexts_features : tuple of np.ndarray (contexts_features)
            a batch of contexts features
            Shape must be (n_contexts, n_contexts_features)
        contexts_items_features : tuple of np.ndarray (contexts_items_features)
            a batch of contexts items features
            Shape must be (n_contexts, n_contexts_items_features)
        contexts_items_availabilities : np.ndarray
            A batch of contexts items availabilities
            Shape must be (n_contexts, n_items)
        choices_batch : np.ndarray
            Choices
            Shape must be (n_contexts, )
        sample_weight : np.ndarray, optional
            List samples weights to apply during the gradient descent to the batch elements,
            by default None

        Returns:
        --------
        tf.Tensor (1, )
            Value of NegativeLogLikelihood loss for the batch
        tf.Tensor (batch_size, n_items)
            Probabilities for each product to be chosen for each context
        """
        # Compute utilities from features
        utilities = self.compute_batch_utility(
            fixed_items_features,
            contexts_features,
            contexts_items_features,
            contexts_items_availabilities,
            choices,
        )
        # Compute probabilities from utilities & availabilties
        probabilities = tf_ops.softmax_with_availabilities(
            contexts_items_logits=utilities,
            contexts_items_availabilities=contexts_items_availabilities,
            normalize_exit=self.normalize_non_buy,
            axis=-1,
        )

        # Compute loss from probabilities & actual choices
        # batch_loss = self.loss(probabilities, c_batch, sample_weight=sample_weight)
        batch_loss = self.loss(
            y_pred=probabilities,
            y_true=tf.one_hot(choices, depth=probabilities.shape[1]),
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

    def predict_probas(self, choice_dataset, batch_size=-1):
        """Predicts the choice probabilities for each context and each product of a ChoiceDataset.

        Parameters
        ----------
        choice_dataset : ChoiceDataset
            Dataset on which to apply to prediction
        batch_size : int, optional
            Batch size to use for the prediction, by default -1

        Returns:
        --------
        np.ndarray (n_contexts, n_items)
            Choice probabilties for each context and each product
        """
        stacked_probabilities = []
        for (
            fixed_items_features,
            contexts_features,
            contexts_items_features,
            contexts_items_availabilities,
            choices,
        ) in choice_dataset.iter_batch(batch_size=batch_size):
            _, probabilities = self.batch_predict(
                fixed_items_features=fixed_items_features,
                contexts_features=contexts_features,
                contexts_items_features=contexts_items_features,
                contexts_items_availabilities=contexts_items_availabilities,
                choices=choices,
            )
            stacked_probabilities.append(probabilities)

        return tf.concat(stacked_probabilities, axis=0)

    def evaluate(self, choice_dataset, sample_weight=None, batch_size=-1):
        """Evaluates the model for each context and each product of a ChoiceDataset.

        Predicts the probabilities according to the model and computes the Negative-Log-Likelihood
        loss from the actual choices.

        Parameters
        ----------
        choice_dataset : ChoiceDataset
            Dataset on which to apply to prediction

        Returns:
        --------
        np.ndarray (n_contexts, n_items)
            Choice probabilties for each context and each product
        """
        batch_losses = []
        for (
            fixed_items_features,
            contexts_features,
            contexts_items_features,
            contexts_items_availabilities,
            choices,
        ) in choice_dataset.iter_batch(batch_size=batch_size):
            loss, _ = self.batch_predict(
                fixed_items_features=fixed_items_features,
                contexts_features=contexts_features,
                contexts_items_features=contexts_items_features,
                contexts_items_availabilities=contexts_items_availabilities,
                choices=choices,
                sample_weight=sample_weight,
            )
            batch_losses.append(loss)
        if batch_size != -1:
            last_batch_size = contexts_items_availabilities.shape[0]
            coefficients = tf.concat(
                [tf.ones(len(batch_losses) - 1) * batch_size, [last_batch_size]], axis=0
            )
            batch_losses = tf.multiply(batch_losses, coefficients)
            batch_loss = tf.reduce_sum(batch_losses) / len(choice_dataset)
        else:
            batch_loss = tf.reduce_mean(batch_losses)
        return batch_loss

    def _lbfgs_train_step(self, dataset, sample_weight=None):
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
                loss_value = self.evaluate(dataset, sample_weight=sample_weight, batch_size=-1)

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

    def _fit_with_lbfgs(self, dataset, epochs=None, sample_weight=None, tolerance=1e-8):
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

        if epochs is None:
            epochs = self.epochs
        func = self._lbfgs_train_step(dataset, sample_weight=sample_weight)

        # convert initial model parameters to a 1D tf.Tensor
        init_params = tf.dynamic_stitch(func.idx, self.weights)

        # train the model with L-BFGS solver
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=func,
            initial_position=init_params,
            max_iterations=epochs,
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

    def compute_batch_utility(
        self,
        fixed_items_features,
        contexts_features,
        contexts_items_features,
        contexts_items_availabilities,
        choices,
    ):
        """Computes the random utility for each product of each context.

        Parameters
        ----------
        fixed_items_features : tuple of np.ndarray
            Fixed-Item-Features: formatting from ChoiceDataset: a matrix representing the products
            constant/fixed features.
            Shape must be (n_items, n_items_features)
        contexts_features : tuple of np.ndarray (contexts_features)
            a batch of contexts features
            Shape must be (n_contexts, n_contexts_features)
        contexts_items_features : tuple of np.ndarray (contexts_items_features)
            a batch of contexts items features
            Shape must be (n_contexts, n_contexts_items_features)
        contexts_items_availabilities : np.ndarray
            A batch of contexts items availabilities
            Shape must be (n_contexts, n_items)
        choices_batch : np.ndarray
            Choices
            Shape must be (n_contexts, )

        Returns:
        --------
        tf.Tensor
            (n_contexts, n_items) matrix of random utilities
        """
        # In order to avoid unused arguments warnings
        _ = fixed_items_features, contexts_features, contexts_items_availabilities, choices
        return np.squeeze(
            np.random.uniform(shape=(contexts_items_features.shape), minval=0, maxval=1)
        )

    def fit(**kwargs):
        """Make sure that nothing happens during .fit."""
        _ = kwargs
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
        _ = kwargs
        choices = choice_dataset.choices
        for i in range(choice_dataset.get_num_items()):
            self.weights.append(tf.reduce_sum(tf.cast(choices == i, tf.float32)))
        self.weights = tf.stack(self.weights) / len(choices)

    def compute_batch_utility(
        self,
        fixed_items_features,
        contexts_features,
        contexts_items_features,
        contexts_items_availabilities,
        choices,
    ):
        """Returns utility that is fixed. U = log(P).

        Parameters
        ----------
        fixed_items_features : tuple of np.ndarray
            Fixed-Item-Features: formatting from ChoiceDataset: a matrix representing the products
            constant/fixed features.
            Shape must be (n_items, n_items_features)
        contexts_features : tuple of np.ndarray (contexts_features)
            a batch of contexts features
            Shape must be (n_contexts, n_contexts_features)
        contexts_items_features : tuple of np.ndarray (contexts_items_features)
            a batch of contexts items features
            Shape must be (n_contexts, n_contexts_items_features)
        contexts_items_availabilities : np.ndarray
            A batch of contexts items availabilities
            Shape must be (n_contexts, n_items)
        choices_batch : np.ndarray
            Choices
            Shape must be (n_contexts, )

        Returns:
        --------
        np.ndarray (n_contexts, n_items)
            Utilities

        Raises:
        -------
        ValueError
            If the model has not been fitted cannot evaluate the utility
        """
        # In order to avoid unused arguments warnings
        _ = fixed_items_features, contexts_features, contexts_items_availabilities
        _ = contexts_items_features
        if self.weights is None:
            raise ValueError("Model not fitted")
        return np.stack([np.log(self.weights.numpy())] * len(choices), axis=0)


class BaseMixtureModel(object):
    def __init__(
        self,
        latent_classes,
        model_class,
        model_parameters,
        fit_method,
        epochs,
    ):
        self.latent_classes = latent_classes
        self.model_parameters = model_parameters
        self.model_class = model_class
        self.fit_method = fit_method

        self.epochs = epochs

    def instantiate(self):
        self.latent_logit = tf.Variable(tf.ones(self.latent_classes)) / self.latent_classes
        self.models = [
            self.model_class(**self.model_parameters) for _ in range(self.latent_classes)
        ]

    def _em_fit(self, dataset):
        for model in self.models:
            # model.instantiate()
            model.fit(dataset)
        for i in tqdm.trange(self.epochs):
            predicted_probas = [model.predict_probas(dataset) for model in self.models]
            predicted_probas = [
                latent
                * tf.gather_nd(
                    params=proba,
                    indices=tf.stack([tf.range(0, len(dataset), 1), dataset.choices], axis=1),
                )
                for latent, proba in zip(self.latent_logit, predicted_probas)
            ]

            weights = predicted_probas / tf.reduce_sum(predicted_probas, axis=0, keepdims=True)
            for q in range(self.latent_classes):
                print(weights[q].shape)
                self.models[q].fit(dataset, sample_weight=weights[q])

            self.latent_probas = tf.reduce_mean(weights, axis=0)
