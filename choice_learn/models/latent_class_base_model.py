"""Base class for latent class choice models."""

import time

import numpy as np
import tensorflow as tf
import tqdm

import choice_learn.tf_ops as tf_ops


class BaseLatentClassModel:
    """Base Class to work with Mixtures of models."""

    def __init__(
        self,
        n_latent_classes,
        model_class,
        model_parameters,
        fit_method,
        epochs,
        batch_size=128,
        optimizer=None,
        add_exit_choice=False,
        lbfgs_tolerance=1e-6,
        lr=0.001,
    ):
        """Instantiate of the model mixture.

        Parameters
        ----------
        n_latent_classes : int
            Number of latent classes
        model_class : BaseModel
            class of models to get a mixture of
        model_parameters : dict
            hyper-parameters of the models
        fit_method : str
            Method to estimate the parameters: "EM", "MLE".
            "EM" for Expectation-Maximization, "MLE" for Maximum Likelihood Estimation
        epochs : int
            Number of epochs to train the model.
        optimizer: str, optional
            Name of the tf.keras.optimizers to be used if one is used, by default None
        add_exit_choice : bool, optional
            Whether or not to add an exit choice, by default False
        lbfgs_tolerance: float, optional
            Tolerance for the L-BFGS optimizer if applied, by default 1e-6
        lr: float, optional
            Learning rate for the optimizer if applied, by default 0.001
        """
        self.n_latent_classes = n_latent_classes
        if isinstance(model_parameters, list):
            if not len(model_parameters) == n_latent_classes:
                raise ValueError(
                    """If you specify a list of hyper-parameters, it means that you want to use\
                    different hyper-parameters for each latent class. In this case, the length\
                        of the list must be equal to the number of latent classes."""
                )
            self.model_parameters = model_parameters
        else:
            self.model_parameters = [model_parameters] * n_latent_classes
        self.model_class = model_class
        self.fit_method = fit_method

        self.epochs = epochs
        self.add_exit_choice = add_exit_choice
        self.lbfgs_tolerance = lbfgs_tolerance
        self.optimizer = optimizer
        self.lr = lr
        self.batch_size = batch_size

        self.loss = tf_ops.CustomCategoricalCrossEntropy(from_logits=False, label_smoothing=0.0)
        self.exact_nll = tf_ops.CustomCategoricalCrossEntropy(
            from_logits=False,
            label_smoothing=0.0,
            sparse=False,
            axis=-1,
            epsilon=1e-10,
            name="exact_categorical_crossentropy",
            reduction="sum_over_batch_size",
        )
        self.instantiated = False

    @property
    def trainable_weights(self):
        """Return trainable weights.

        Returns
        -------
        list
           list of trainable weights.
        """
        weights = [self.latent_logits]
        for model in self.models:
            weights += model.trainable_weights
        return weights

    def instantiate(self, **kwargs):
        """Instantiate the model."""
        init_logit = tf.Variable(
            tf.random_normal_initializer(0.0, 0.08, seed=42)(shape=(self.n_latent_classes - 1,)),
            name="Latent-Logits",
        )
        self.latent_logits = init_logit
        self.models = [self.model_class(**mp) for mp in self.model_parameters]
        for model in self.models:
            model.instantiate(**kwargs)

    # @tf.function
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
        choices: np.ndarray
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

        latent_probabilities = self.get_latent_classes_weights()
        # Compute probabilities from utilities & availabilties
        probabilities = []
        for i, class_utilities in enumerate(utilities):
            class_probabilities = tf_ops.softmax_with_availabilities(
                items_logit_by_choice=class_utilities,
                available_items_by_choice=available_items_by_choice,
                normalize_exit=self.add_exit_choice,
                axis=-1,
            )
            probabilities.append(class_probabilities * latent_probabilities[i])
        # Summing over the latent classes
        probabilities = tf.reduce_sum(probabilities, axis=0)

        # Compute loss from probabilities & actual choices
        # batch_loss = self.loss(probabilities, c_batch, sample_weight=sample_weight)
        batch_loss = {
            "optimized_loss": self.loss(
                y_pred=probabilities,
                y_true=tf.one_hot(choices, depth=probabilities.shape[1]),
                sample_weight=sample_weight,
            ),
            "NegativeLogLikelihood": self.exact_nll(
                y_pred=probabilities,
                y_true=tf.one_hot(choices, depth=probabilities.shape[1]),
                sample_weight=sample_weight,
            ),
        }
        return batch_loss, probabilities

    def compute_batch_utility(
        self,
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices,
    ):
        """Latent class computation of utility.

        It computes the utility for each of the latent models and stores them in a list.

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
        choices : np.ndarray
            Choices
            Shape must be (n_choices, )

        Returns
        -------
        list of np.ndarray
            List of:
                Utility of each product for each choice.
                Shape must be (n_choices, n_items)
            for each of the latent models.
        """
        utilities = []
        # Iterates over latent models
        for model in self.models:
            model_utilities = model.compute_batch_utility(
                shared_features_by_choice=shared_features_by_choice,
                items_features_by_choice=items_features_by_choice,
                available_items_by_choice=available_items_by_choice,
                choices=choices,
            )
            utilities.append(model_utilities)
        return utilities

    def fit(self, choice_dataset, sample_weight=None, verbose=0):
        """Fit the model on a ChoiceDataset.

        Parameters
        ----------
        choice_dataset : ChoiceDataset
            Dataset to be used for coefficients estimations
        sample_weight : np.ndarray, optional
            sample weights to apply, by default None
        verbose : int, optional
            print level, for debugging, by default 0

        Returns
        -------
        dict
            Fit history
        """
        if self.fit_method.lower() == "em":
            self.minf = np.log(1e-3)
            print("Expectation-Maximization estimation algorithm not well implemented yet.")
            return self._em_fit(
                choice_dataset=choice_dataset, sample_weight=sample_weight, verbose=verbose
            )

        if self.fit_method.lower() == "mle":
            if isinstance(self.optimizer, str):
                if self.optimizer.lower() == "lbfgs" or self.optimizer.lower() == "l-bfgs":
                    return self._fit_with_lbfgs(
                        choice_dataset=choice_dataset, sample_weight=sample_weight, verbose=verbose
                    )

                if self.optimizer.lower() == "adam":
                    self.optimizer = tf.keras.optimizers.Adam(self.lr)
                elif self.optimizer.lower() == "sgd":
                    self.optimizer = tf.keras.optimizers.SGD(self.lr)
                elif self.optimizer.lower() == "adamax":
                    self.optimizer = tf.keras.optimizers.Adamax(self.lr)
                else:
                    print(f"Optimizer {self.optimizer} not implemnted, switching for default Adam")
                    self.optimizer = tf.keras.optimizers.Adam(self.lr)

            return self._fit_with_gd(
                choice_dataset=choice_dataset, sample_weight=sample_weight, verbose=verbose
            )

        raise ValueError(f"Fit method not implemented: {self.fit_method}")

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
            shared_features,
            items_features,
            available_items,
            choices,
        ) in choice_dataset.iter_batch(batch_size=batch_size):
            loss, _ = self.batch_predict(
                shared_features_by_choice=shared_features,
                items_features_by_choice=items_features,
                available_items_by_choice=available_items,
                choices=choices,
                sample_weight=sample_weight,
            )
            if mode == "eval":
                batch_losses.append(loss["NegativeLogLikelihood"])
            elif mode == "optim":
                batch_losses.append(loss["optimized_loss"])
        if batch_size != -1:
            last_batch_size = available_items.shape[0]
            coefficients = tf.concat(
                [tf.ones(len(batch_losses) - 1) * batch_size, [last_batch_size]], axis=0
            )
            batch_losses = tf.multiply(batch_losses, coefficients)
            batch_loss = tf.reduce_sum(batch_losses) / len(choice_dataset)
        else:
            batch_loss = tf.reduce_mean(batch_losses)
        return batch_loss

    def _lbfgs_train_step(self, choice_dataset, sample_weight=None):
        """Create a function required by tfp.optimizer.lbfgs_minimize.

        Parameters
        ----------
        choice_dataset: ChoiceDataset
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
        trainable_weights = []
        w_to_model = []
        w_to_model_indexes = []
        for i, model in enumerate(self.models):
            for j, w in enumerate(model.trainable_weights):
                trainable_weights.append(w)
                w_to_model.append(i)
                w_to_model_indexes.append(j)
        trainable_weights.append(self.latent_logits)
        w_to_model.append(-1)
        w_to_model_indexes.append(-1)
        shapes = tf.shape_n(trainable_weights)
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
                if w_to_model[i] != -1:
                    self.models[w_to_model[i]].trainable_weights[w_to_model_indexes[i]].assign(
                        tf.reshape(param, shape)
                    )
                else:
                    self.latent_logits.assign(tf.reshape(param, shape))

        # now create a function that will be returned by this factory
        @tf.function
        def f(params_1d):
            """To be used by tfp.optimizer.lbfgs_minimize.

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
                    choice_dataset, sample_weight=sample_weight, batch_size=-1, mode="optim"
                )
            # calculate gradients and convert to 1D tf.Tensor
            grads = tape.gradient(loss_value, trainable_weights)
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

    def _fit_with_lbfgs(self, choice_dataset, sample_weight=None, verbose=0):
        """Fit function for L-BFGS optimizer.

        Replaces the .fit method when the optimizer is set to L-BFGS.

        Parameters
        ----------
        choice_dataset : ChoiceDataset
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
        func = self._lbfgs_train_step(choice_dataset, sample_weight=sample_weight)

        # convert initial model parameters to a 1D tf.Tensor
        init = []
        for model in self.models:
            for w in model.trainable_weights:
                init.append(w)
        init.append(self.latent_logits)
        init_params = tf.dynamic_stitch(func.idx, init)

        # train the model with L-BFGS solver
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=func,
            initial_position=init_params,
            max_iterations=epochs,
            tolerance=-1,
            f_absolute_tolerance=self.lbfgs_tolerance,
            f_relative_tolerance=-1,
            x_tolerance=-1,
        )

        # after training, the final optimized parameters are still in results.position
        # so we have to manually put them back to the model
        func.assign_new_model_parameters(results.position)
        if verbose > 0:
            print("L-BFGS Opimization finished:")
            print("---------------------------------------------------------------")
            print("Number of iterations:", results[2].numpy())
            print("Algorithm converged before reaching max iterations:", results[0].numpy())
        return func.history, results

    # @tf.function
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

            latent_probabilities = self.get_latent_classes_weights()
            # Compute probabilities from utilities & availabilties
            probabilities = []
            for i, class_utilities in enumerate(utilities):
                class_probabilities = tf_ops.softmax_with_availabilities(
                    items_logit_by_choice=class_utilities,
                    available_items_by_choice=available_items_by_choice,
                    normalize_exit=self.add_exit_choice,
                    axis=-1,
                )
                probabilities.append(class_probabilities * latent_probabilities[i])
            # Summing over the latent classes
            probabilities = tf.reduce_sum(probabilities, axis=0)
            # Negative Log-Likelihood
            neg_loglikelihood = self.loss(
                y_pred=probabilities,
                y_true=tf.one_hot(choices, depth=probabilities.shape[1]),
                sample_weight=sample_weight,
            )
            # if self.regularization is not None:
            #     regularization = tf.reduce_sum(
            #         [self.regularizer(w) for w in self.trainable_weights]
            #     )
            #     neg_loglikelihood += regularization

        grads = tape.gradient(neg_loglikelihood, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return neg_loglikelihood

    def _fit_with_gd(
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

        # self.callbacks.on_train_begin()

        # Iterate of epochs
        for epoch_nb in t_range:
            # self.callbacks.on_epoch_begin(epoch_nb)
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
                    # self.callbacks.on_train_batch_begin(batch_nb)

                    neg_loglikelihood = self.train_step(
                        shared_features_batch,
                        items_features_batch,
                        available_items_batch,
                        choices_batch,
                        sample_weight=weight_batch,
                    )

                    train_logs["train_loss"].append(neg_loglikelihood)

                    # temps_logs = {k: tf.reduce_mean(v) for k, v in train_logs.items()}
                    # self.callbacks.on_train_batch_end(batch_nb, logs=temps_logs)

                    # Optimization Steps
                    epoch_losses.append(neg_loglikelihood)

                    if verbose > 0:
                        inner_range.set_description(
                            f"Epoch Negative-LogLikeliHood: {np.sum(epoch_losses):.4f}"
                        )

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
                    # self.callbacks.on_train_batch_begin(batch_nb)
                    neg_loglikelihood = self.train_step(
                        shared_features_batch,
                        items_features_batch,
                        available_items_batch,
                        choices_batch,
                    )
                    train_logs["train_loss"].append(neg_loglikelihood)
                    # temps_logs = {k: tf.reduce_mean(v) for k, v in train_logs.items()}
                    # self.callbacks.on_train_batch_end(batch_nb, logs=temps_logs)

                    # Optimization Steps
                    epoch_losses.append(neg_loglikelihood)

                    if verbose > 0:
                        inner_range.set_description(
                            f"Epoch Negative-LogLikeliHood: {np.sum(epoch_losses):.4f}"
                        )

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
            print_loss = losses_history["train_loss"][-1].numpy()
            desc = f"Epoch {epoch_nb} Train Loss {print_loss:.4f}"
            if verbose > 1:
                print(
                    f"Loop {epoch_nb} Time:",
                    f"{time.time() - t_start:.4f}",
                    f"Loss: {print_loss:.4f}",
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
                    # self.callbacks.on_batch_begin(batch_nb)
                    # self.callbacks.on_test_batch_begin(batch_nb)
                    test_losses.append(
                        self.batch_predict(
                            shared_features_batch,
                            items_features_batch,
                            available_items_batch,
                            choices_batch,
                        )[0]["optimized_loss"]
                    )
                    val_logs["val_loss"].append(test_losses[-1])
                    # temps_logs = {k: tf.reduce_mean(v) for k, v in val_logs.items()}
                    # self.callbacks.on_test_batch_end(batch_nb, logs=temps_logs)

                test_loss = tf.reduce_mean(test_losses)
                if verbose > 1:
                    print("Test Negative-LogLikelihood:", test_loss.numpy())
                    desc += f", Test Loss {np.round(test_loss.numpy(), 4)}"
                losses_history["test_loss"] = losses_history.get("test_loss", []) + [
                    test_loss.numpy()
                ]
                train_logs = {**train_logs, **val_logs}

            # temps_logs = {k: tf.reduce_mean(v) for k, v in train_logs.items()}
            # self.callbacks.on_epoch_end(epoch_nb, logs=temps_logs)
            # if self.stop_training:
            #     print("Early Stopping taking effect")
            #     break
            t_range.set_description(desc)
            t_range.refresh()

        # temps_logs = {k: tf.reduce_mean(v) for k, v in train_logs.items()}
        # self.callbacks.on_train_end(logs=temps_logs)
        return losses_history

    def _nothing(self, inputs):
        """_summary_.

        Parameters
        ----------
        inputs : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        latent_probas = tf.clip_by_value(
            self.latent_logits - tf.reduce_max(self.latent_logits), self.minf, 0
        )
        latent_probas = tf.math.exp(latent_probas)
        # latent_probas = tf.math.abs(self.logit_latent_probas)  # alternative implementation
        latent_probas = latent_probas / tf.reduce_sum(latent_probas)
        proba_list = []
        avail = inputs[4]
        for q in range(self.n_latent_classes):
            combined = self.models[q].compute_batch_utility(*inputs)
            combined = tf.clip_by_value(
                combined - tf.reduce_max(combined, axis=1, keepdims=True), self.minf, 0
            )
            combined = tf.keras.layers.Activation(activation=tf.nn.softmax)(combined)
            # combined = tf.keras.layers.Softmax()(combined)
            combined = combined * avail
            combined = latent_probas[q] * tf.math.divide(
                combined, tf.reduce_sum(combined, axis=1, keepdims=True)
            )
            combined = tf.expand_dims(combined, -1)
            proba_list.append(combined)
            # print(combined.get_shape()) # it is useful to print the shape of tensors for debugging

        proba_final = tf.keras.layers.Concatenate(axis=2)(proba_list)
        return tf.math.reduce_sum(proba_final, axis=2, keepdims=False)

    def _expectation(self, choice_dataset):
        predicted_probas = [model.predict_probas(choice_dataset) for model in self.models]
        latent_probabilities = self.get_latent_classes_weights()
        if np.sum(np.isnan(predicted_probas)) > 0:
            print("A NaN values has been found. You should try again to fit with")
            print("smaller tolerance value (for l-bfgs) and epsilon value (in loss computation)")

        latent_model_probas = [
            latent * proba for latent, proba in zip(latent_probabilities, predicted_probas)
        ]
        latent_model_probas = tf.reduce_sum(latent_model_probas, axis=0)
        predicted_probas = [
            latent
            * tf.gather_nd(
                params=proba,
                indices=tf.stack(
                    [tf.range(0, len(choice_dataset), 1), choice_dataset.choices], axis=1
                ),
            )
            for latent, proba in zip(latent_probabilities, predicted_probas)
        ]
        predicted_probas = np.stack(predicted_probas, axis=1)
        loss = self.loss(
            y_pred=latent_model_probas,
            y_true=tf.one_hot(choice_dataset.choices, depth=latent_model_probas.shape[1]),
        )

        return tf.clip_by_value(
            predicted_probas / np.sum(predicted_probas, axis=1, keepdims=True), 1e-10, 1
        ), loss

    def _maximization(self, choice_dataset, verbose=0):
        """Maximize step.

        Parameters
        ----------
        choice_dataset : ChoiceDataset
            dataset to be fitted
        verbose : int, optional
            print level, for debugging, by default 0

        Returns
        -------
        np.ndarray
            latent probabilities resulting of maximization step
        """
        self.models = [self.model_class(**mp) for mp in self.model_parameters]
        # M-step: MNL estimation
        for q in range(self.n_latent_classes):
            self.models[q].fit(choice_dataset, sample_weight=self.weights[:, q], verbose=verbose)

        # M-step: latent probability estimation
        latent_probas = np.sum(self.weights, axis=0)
        return tf.math.log((latent_probas / latent_probas[0])[1:])

    def _em_fit(self, choice_dataset, sample_weight=None, verbose=0):
        """Fit with Expectation-Maximization Algorithm.

        Parameters
        ----------
        choice_dataset: ChoiceDataset
            Dataset to be used for coefficients estimations
        sample_weight : np.ndarray, optional
            sample weights to apply, by default None
        verbose : int, optional
            print level, for debugging, by default 0

        Returns
        -------
        list
            List of logits for each latent class
        list
            List of losses at each epoch
        """
        hist_logits = []
        hist_loss = []
        _ = sample_weight

        # Initialization
        init_sample_weight = np.random.rand(self.n_latent_classes, len(choice_dataset))
        init_sample_weight = init_sample_weight / np.sum(init_sample_weight, axis=0, keepdims=True)
        for i, model in enumerate(self.models):
            # model.instantiate()
            model.fit(choice_dataset, sample_weight=init_sample_weight[i], verbose=verbose)
        for i in tqdm.trange(self.epochs):
            self.weights, loss = self._expectation(choice_dataset)
            self.latent_logits = self._maximization(choice_dataset, verbose=verbose)
            hist_logits.append(self.latent_logits)
            hist_loss.append(loss)
            if np.sum(np.isnan(self.latent_logits)) > 0:
                print("Nan in logits")
                break
        return hist_logits, hist_loss

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
            shared_features,
            items_features,
            available_items,
            choices,
        ) in choice_dataset.iter_batch(batch_size=batch_size):
            _, probabilities = self.batch_predict(
                shared_features_by_choice=shared_features,
                items_features_by_choice=items_features,
                available_items_by_choice=available_items,
                choices=choices,
            )
            stacked_probabilities.append(probabilities)

        return tf.concat(stacked_probabilities, axis=0)

    def get_latent_classes_weights(self):
        """Return the latent classes weights / probabilities from logits.

        Returns
        -------
        np.ndarray (n_latent_classes, )
            Latent classes weights/probabilities
        """
        return tf.nn.softmax(tf.concat([[tf.constant(0.0)], self.latent_logits], axis=0))
