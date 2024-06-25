"""Base class for choice models."""
import json
import logging
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
        add_exit_choice=False,
        optimizer="lbfgs",
        tolerance=1e-8,
        callbacks=None,
        lr=0.001,
        epochs=1000,
        batch_size=32,
        regularization=None,
        regularization_strength=0.0,
    ):
        """Instantiate the ChoiceModel.

        Parameters
        ----------
        label_smoothing : float, optional
            Whether (then is ]O, 1[ value) or not (then can be None or 0) to use label smoothing,
        during training, by default 0.0
            by default None. Label smoothing is applied to LogLikelihood loss.
        add_exit_choice : bool, optional
            Whether or not to add a normalization (then U=1) with the exit option in probabilites
            normalization,by default True
        callbacks : list of tf.kera callbacks, optional
            List of callbacks to add to model.fit, by default None and only add History
        optimizer : str, optional
            Name of the tf.keras.optimizers to be used, by default "lbfgs"
        tolerance : float, optional
            Tolerance for the L-BFGS optimizer if applied, by default 1e-8
        lr: float, optional
            Learning rate for the optimizer if applied, by default 0.001
        epochs: int, optional
            (Max) Number of epochs to train the model, by default 1000
        batch_size: int, optional
            Batch size in the case of stochastic gradient descent optimizer.
            Not used in the case of L-BFGS optimizer, by default 32
        regularization: str
            Type of regularization to apply: "l1", "l2" or "l1l2", by default None
        regularization_strength: float or list
            weight of regularization in loss computation. If "l1l2" is chosen as regularization,
            can be given as list or tuple: [l1_strength, l2_strength], by default 0.
        """
        self.is_fitted = False
        self.add_exit_choice = add_exit_choice
        self.label_smoothing = label_smoothing
        self.stop_training = False

        # Loss function wrapping tf.keras.losses.CategoricalCrossEntropy
        # with smoothing and normalization options
        self.loss = tf_ops.CustomCategoricalCrossEntropy(
            from_logits=False, label_smoothing=self.label_smoothing
        )
        self.exact_nll = tf_ops.CustomCategoricalCrossEntropy(
            from_logits=False,
            label_smoothing=0.0,
            sparse=False,
            axis=-1,
            epsilon=1e-35,
            name="exact_categorical_crossentropy",
            reduction="sum_over_batch_size",
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
        self.tolerance = tolerance

        if regularization is not None:
            if regularization.lower() == "l1":
                self.regularizer = tf.keras.regularizers.L1(l1=regularization_strength)
            elif regularization.lower() == "l2":
                self.regularizer = tf.keras.regularizers.L2(l2=regularization_strength)
            elif regularization.lower() == "l1l2":
                if isinstance(regularization_strength, (list, tuple)):
                    self.regularizer = tf.keras.regularizers.L1L2(
                        l1=regularization_strength[0], l2=regularization_strength[1]
                    )
                else:
                    self.regularizer = tf.keras.regularizers.L1L2(
                        l1=regularization_strength, l2=regularization_strength
                    )
            else:
                raise ValueError(
                    "Regularization type not recognized, choose among l1, l2 and l1l2."
                )
            self.regularization = regularization
            self.regularization_strength = regularization_strength
        else:
            self.regularization_strength = 0.0
            self.regularization = None

    @abstractmethod
    def compute_batch_utility(
        self,
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices,
    ):
        """Define how the model computes the utility of a product.

        MUST be implemented in children classe !
        For simpler use-cases this is the only method to be user-defined.

        Parameters
        ----------
        shared_features_by_choice : tuple of np.ndarray (choices_features)
            a batch of shared features
            Shape must be (n_choices, n_shared_features)
        items_features_by_choice : tuple of np.ndarray (choices_items_features)
            a batch of items features
            Shape must be (n_choices, n_items_features)
        available_items_by_choice : np.ndarray
            A batch of items availabilities
            Shape must be (n_choices, n_items)
        choices_batch : np.ndarray
            Choices
            Shape must be (n_choices, )

        Returns
        -------
        np.ndarray
            Utility of each product for each choice.
            Shape must be (n_choices, n_items)
        """
        # To be implemented in children classes
        # Can be NumPy or TensorFlow based
        return

    @tf.function
    def train_step(
        self,
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices,
        sample_weight=None,
    ):
        """Represent one training step (= one gradient descent step) of the model.

        Parameters
        ----------
        shared_features_by_choice : tuple of np.ndarray (choices_features)
            a batch of shared features
            Shape must be (n_choices, n_shared_features)
        items_features_by_choice : tuple of np.ndarray (choices_items_features)
            a batch of items features
            Shape must be (n_choices, n_items_features)
        available_items_by_choice : np.ndarray
            A batch of items availabilities
            Shape must be (n_choices, n_items)
        choices_batch : np.ndarray
            Choices
            Shape must be (n_choices, )
        sample_weight : np.ndarray, optional
            List samples weights to apply during the gradient descent to the batch elements,
            by default None

        Returns
        -------
        tf.Tensor
            Value of NegativeLogLikelihood loss for the batch
        """
        with tf.GradientTape() as tape:
            utilities = self.compute_batch_utility(
                shared_features_by_choice=shared_features_by_choice,
                items_features_by_choice=items_features_by_choice,
                available_items_by_choice=available_items_by_choice,
                choices=choices,
            )

            probabilities = tf_ops.softmax_with_availabilities(
                items_logit_by_choice=utilities,
                available_items_by_choice=available_items_by_choice,
                normalize_exit=self.add_exit_choice,
                axis=-1,
            )
            # Negative Log-Likelihood
            neg_loglikelihood = self.loss(
                y_pred=probabilities,
                y_true=tf.one_hot(choices, depth=probabilities.shape[1]),
                sample_weight=sample_weight,
            )
            if self.regularization is not None:
                regularization = tf.reduce_sum(
                    [self.regularizer(w) for w in self.trainable_weights]
                )
                neg_loglikelihood += regularization

        grads = tape.gradient(neg_loglikelihood, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return neg_loglikelihood

    def fit(
        self,
        choice_dataset,
        sample_weight=None,
        val_dataset=None,
        verbose=0,
    ):
        """Train the model with a ChoiceDataset.

        Parameters
        ----------
        choice_dataset : ChoiceDataset
            Input data in the form of a ChoiceDataset
        sample_weight : np.ndarray, optional
            Sample weight to apply, by default None
        val_dataset : ChoiceDataset, optional
            Test ChoiceDataset to evaluate performances on test at each epoch, by default None
        verbose : int, optional
            print level, for debugging, by default 0
        epochs : int, optional
            Number of epochs, default is None, meaning we use self.epochs
        batch_size : int, optional
            Batch size, default is None, meaning we use self.batch_size

        Returns
        -------
        dict:
            Different metrics values over epochs.
        """
        if hasattr(self, "instantiated"):
            if not self.instantiated:
                raise ValueError("Model not instantiated. Please call .instantiate() first.")
        epochs = self.epochs
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
                        shared_features_batch,
                        items_features_batch,
                        available_items_batch,
                        choices_batch,
                    ),
                    weight_batch,
                ) in enumerate(inner_range):
                    self.callbacks.on_train_batch_begin(batch_nb)

                    neg_loglikelihood = self.train_step(
                        shared_features_batch,
                        items_features_batch,
                        available_items_batch,
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
                    shared_features_batch,
                    items_features_batch,
                    available_items_batch,
                    choices_batch,
                ) in enumerate(inner_range):
                    self.callbacks.on_train_batch_begin(batch_nb)
                    neg_loglikelihood = self.train_step(
                        shared_features_batch,
                        items_features_batch,
                        available_items_batch,
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
                last_batch_size = available_items_batch.shape[0]
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
                    shared_features_batch,
                    items_features_batch,
                    available_items_batch,
                    choices_batch,
                ) in enumerate(val_dataset.iter_batch(shuffle=False, batch_size=batch_size)):
                    self.callbacks.on_batch_begin(batch_nb)
                    self.callbacks.on_test_batch_begin(batch_nb)
                    test_losses.append(
                        self.batch_predict(
                            shared_features_batch,
                            items_features_batch,
                            available_items_batch,
                            choices_batch,
                        )[0]["optimized_loss"]
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
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices,
        sample_weight=None,
    ):
        """Represent one prediction (Probas + Loss) for one batch of a ChoiceDataset.

        Parameters
        ----------
        shared_features_by_choice : tuple of np.ndarray (choices_features)
            a batch of shared features
            Shape must be (n_choices, n_shared_features)
        items_features_by_choice : tuple of np.ndarray (choices_items_features)
            a batch of items features
            Shape must be (n_choices, n_items_features)
        available_items_by_choice : np.ndarray
            A batch of items availabilities
            Shape must be (n_choices, n_items)
        choices_batch : np.ndarray
            Choices
            Shape must be (n_choices, )
        sample_weight : np.ndarray, optional
            List samples weights to apply during the gradient descent to the batch elements,
            by default None

        Returns
        -------
        tf.Tensor (1, )
            Value of NegativeLogLikelihood loss for the batch
        tf.Tensor (batch_size, n_items)
            Probabilities for each product to be chosen for each choice
        """
        # Compute utilities from features
        utilities = self.compute_batch_utility(
            shared_features_by_choice,
            items_features_by_choice,
            available_items_by_choice,
            choices,
        )
        # Compute probabilities from utilities & availabilties
        probabilities = tf_ops.softmax_with_availabilities(
            items_logit_by_choice=utilities,
            available_items_by_choice=available_items_by_choice,
            normalize_exit=self.add_exit_choice,
            axis=-1,
        )

        # Compute loss from probabilities & actual choices
        # batch_loss = self.loss(probabilities, c_batch, sample_weight=sample_weight)
        batch_loss = {
            "optimized_loss": self.loss(
                y_pred=probabilities,
                y_true=tf.one_hot(choices, depth=probabilities.shape[1]),
                sample_weight=sample_weight,
            ),
            # "NegativeLogLikelihood": tf.keras.losses.CategoricalCrossentropy()(
            #     y_pred=probabilities,
            #     y_true=tf.one_hot(choices, depth=probabilities.shape[1]),
            #     sample_weight=sample_weight,
            # ),
            "Exact-NegativeLogLikelihood": self.exact_nll(
                y_pred=probabilities,
                y_true=tf.one_hot(choices, depth=probabilities.shape[1]),
                sample_weight=sample_weight,
            ),
        }
        return batch_loss, probabilities

    def save_model(self, path):
        """Save the different models on disk.

        Parameters
        ----------
        path : str
            path to the folder where to save the model
        """
        if not os.exists(path):
            Path(path).mkdir(parents=True)

        for i, weight in enumerate(self.trainable_weights):
            tf.keras.savedmodel.save(Path(path) / f"weight_{i}")

        # To improve for non-string attributes
        params = self.__dict__
        json.dump(Path(path) / "params.json", params)

        # Save optimizer state

    @classmethod
    def load_model(cls, path):
        """Load a ChoiceModel previously saved with save_model().

        Parameters
        ----------
        path : str
            path to the folder where the saved model files are

        Returns
        -------
        ChoiceModel
            Loaded ChoiceModel
        """
        obj = cls()
        obj.trainable_weights = []
        i = 0
        weight_path = f"weight_{i}"
        while weight_path in os.listdir(path):
            obj.trainable_weights.append(tf.keras.load_model.load(Path(path) / weight_path))
            i += 1
            weight_path = f"weight_{i}"

        # To improve for non string attributes
        params = json.load(Path(path) / "params.json")
        for k, v in params.items():
            setattr(obj, k, v)

        # Load optimizer step
        return cls

    def predict_probas(self, choice_dataset, batch_size=-1):
        """Predicts the choice probabilities for each choice and each product of a ChoiceDataset.

        Parameters
        ----------
        choice_dataset : ChoiceDataset
            Dataset on which to apply to prediction
        batch_size : int, optional
            Batch size to use for the prediction, by default -1

        Returns
        -------
        np.ndarray (n_choices, n_items)
            Choice probabilties for each choice and each product
        """
        stacked_probabilities = []
        for (
            shared_features_by_choice,
            items_features_by_choice,
            available_items_by_choice,
            choices,
        ) in choice_dataset.iter_batch(batch_size=batch_size):
            _, probabilities = self.batch_predict(
                shared_features_by_choice=shared_features_by_choice,
                items_features_by_choice=items_features_by_choice,
                available_items_by_choice=available_items_by_choice,
                choices=choices,
            )
            stacked_probabilities.append(probabilities)

        return tf.concat(stacked_probabilities, axis=0)

    def evaluate(self, choice_dataset, sample_weight=None, batch_size=-1, mode="eval"):
        """Evaluate the model for each choice and each product of a ChoiceDataset.

        Predicts the probabilities according to the model and computes the Negative-Log-Likelihood
        loss from the actual choices.

        Parameters
        ----------
        choice_dataset : ChoiceDataset
            Dataset on which to apply to prediction

        Returns
        -------
        np.ndarray (n_choices, n_items)
            Choice probabilties for each choice and each product
        """
        batch_losses = []
        for (
            shared_features_by_choice,
            items_features_by_choice,
            available_items_by_choice,
            choices,
        ) in choice_dataset.iter_batch(batch_size=batch_size):
            loss, _ = self.batch_predict(
                shared_features_by_choice=shared_features_by_choice,
                items_features_by_choice=items_features_by_choice,
                available_items_by_choice=available_items_by_choice,
                choices=choices,
                sample_weight=sample_weight,
            )
            if mode == "eval":
                batch_losses.append(loss["Exact-NegativeLogLikelihood"])
            elif mode == "optim":
                batch_losses.append(loss["optimized_loss"])
        if batch_size != -1:
            last_batch_size = available_items_by_choice.shape[0]
            coefficients = tf.concat(
                [tf.ones(len(batch_losses) - 1) * batch_size, [last_batch_size]], axis=0
            )
            batch_losses = tf.multiply(batch_losses, coefficients)
            batch_loss = tf.reduce_sum(batch_losses) / len(choice_dataset)
        else:
            batch_loss = tf.reduce_mean(batch_losses)
        return batch_loss

    def _lbfgs_train_step(self, dataset, sample_weight=None):
        """Create a function required by tfp.optimizer.lbfgs_minimize.

        Parameters
        ----------
        dataset: ChoiceDataset
            Dataset on which to estimate the paramters.
        sample_weight: np.ndarray, optional
            Sample weights to apply, by default None

        Returns
        -------
        function
            with the signature:
                loss_value, gradients = f(model_parameters).
        """
        # obtain the shapes of all trainable parameters in the model
        shapes = tf.shape_n(self.trainable_weights)
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
            """Update the model's parameters with a 1D tf.Tensor.

            Pararmeters
            -----------
            params_1d: tf.Tensor
                a 1D tf.Tensor representing the model's trainable parameters.
            """
            params = tf.dynamic_partition(params_1d, part, n_tensors)
            for i, (shape, param) in enumerate(zip(shapes, params)):
                self.trainable_weights[i].assign(tf.reshape(param, shape))

        # now create a function that will be returned by this factory
        @tf.function
        def f(params_1d):
            """Can be used by tfp.optimizer.lbfgs_minimize.

            This function is created by function_factory.

            Parameters
            ----------
            params_1d: tf.Tensor
                a 1D tf.Tensor.

            Returns
            -------
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
                loss_value = self.evaluate(
                    dataset, sample_weight=sample_weight, batch_size=-1, mode="eval"
                )
                if self.regularization is not None:
                    regularization = tf.reduce_sum(
                        [self.regularizer(w) for w in self.trainable_weights]
                    )
                    loss_value += regularization

            # calculate gradients and convert to 1D tf.Tensor
            grads = tape.gradient(loss_value, self.trainable_weights)
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

    def _fit_with_lbfgs(self, dataset, sample_weight=None, verbose=0):
        """Fit function for L-BFGS optimizer.

        Replaces the .fit method when the optimizer is set to L-BFGS.

        Parameters
        ----------
        dataset : ChoiceDataset
            Dataset to be used for coefficients estimations
        epochs : int
            Maximum number of epochs allowed to reach minimum
        sample_weight : np.ndarray, optional
            Sample weights to apply, by default None
        verbose : int, optional
            print level, for debugging, by default 0

        Returns
        -------
        dict
            Fit history
        """
        # Only import tensorflow_probability if LBFGS optimizer is used, avoid unnecessary
        # dependency
        import tensorflow_probability as tfp

        epochs = self.epochs
        func = self._lbfgs_train_step(dataset, sample_weight=sample_weight)

        # convert initial model parameters to a 1D tf.Tensor
        init_params = tf.dynamic_stitch(func.idx, self.trainable_weights)
        # train the model with L-BFGS solver
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=func,
            initial_position=init_params,
            max_iterations=epochs,
            tolerance=self.tolerance,
            f_absolute_tolerance=-1,
            f_relative_tolerance=-1,
        )

        # after training, the final optimized parameters are still in results.position
        # so we have to manually put them back to the model
        func.assign_new_model_parameters(results.position)
        if results[1].numpy():
            logging.error("L-BFGS Optimization failed.")
        if verbose > 0:
            logging.warning("L-BFGS Opimization finished:")
            logging.warning("---------------------------------------------------------------")
            logging.warning(f"Number of iterations: {results[2].numpy()}")
            logging.warning(
                f"Algorithm converged before reaching max iterations: {results[0].numpy()}",
            )
        return {"train_loss": func.history}
