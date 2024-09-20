"""Implementation of ResLogit for easy use."""

import tensorflow as tf


class Logit(object):
    """The Logit model class."""

    def __init__(self, input, choice, n_vars, n_choices, beta=None, asc=None):
        """Initialize the Logit class.

        Parameters
        ----------
        input : tf.Variable
        choice : tf.Variable
        n_vars : int
            Number of input variables.
        n_choices : int
            Number of choice alternatives.
        """
        self.input = input
        self.choice = choice

        # Define initial value for asc parameter and parameters associated to explanatory variables
        asc_init = tf.zeros((n_choices,), dtype=tf.float64)
        if asc is None:
            asc = tf.Variable(asc_init, trainable=True)
        self.asc = asc

        beta_init = tf.zeros((n_vars, n_choices), dtype=tf.float64)
        if beta is None:
            beta = tf.Variable(beta_init, trainable=True)
        self.beta = beta

        self.params = [self.beta, self.asc]

        # Compute the utility function and the probability  of each alternative
        pre_softmax = tf.matmul(input, self.beta) + self.asc

        self.output = tf.nn.softmax(pre_softmax, axis=1)

        self.output_pred = tf.math.argmax(self.output, axis=1)

    def negative_log_likelihood(self, x, y):
        """Cost function.

        Parameters
        ----------
        y : tf.Variable
            The output
        x : tf.Variable
            The probability of alternatives

        Returns
        -------
        tf.Variable???
            Sum of the negative log likelihood
        """
        self.cost = -tf.reduce_sum(tf.math.log(x))[tf.range(tf.shape(y)[0])]

        return self.cost

    def prob_choice(self):
        """Output probabilities."""
        return self.output_logit

    def prediction(self):
        """Output prediction."""
        return self.output_pred

    def errors(self, y):
        """Cost function.

        Parameters
        ----------
        y : tf.Variable
            The correct label

        Returns
        -------
        int
            Number of errors in the minibatch for computing the accuracy of model
        """
        if y.ndim != self.output_pred.ndim:
            raise TypeError(
                "y should have the same shape as self.output_pred",
                ("y", y.type, "y_pred", self.output_pred_logit.type),
            )
        if y.dtype in [tf.int16, tf.int32, tf.int64]:
            not_equal = tf.math.not_equal(self.output_pred, y)
            return not_equal.sum().float() / not_equal.num_elements()  # tf.shape.num_elements
        else:
            raise NotImplementedError()


class ResNetLayer(tf.keras.layers.Layer):
    """The ResNet layer class."""

    def __init__(self, n_in, n_out):
        """Initialize the ResNetLayer class.

        Parameters
        ----------
        n_in : int
            Dimensionality of input
        n_out : int
            Dimensionality of output
        """
        super(ResNetLayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out

        # Define initial value of residual layer weights
        w_init = tf.eye(self.n_out, dtype=tf.float64)

        # Learnable parameters of a model
        self.w = tf.Variable(w_init, trainable=True)
        self.params = [self.w]

    # def forward(self, x):
    def call(self, x):
        """Return the output of each residual layer.

        Parameters
        ----------
        x : tf.Variable
            Input of each residual layer
        """
        # self.lin_output = tf.matmul(x, self.w)
        self.lin_output = tf.matmul(tf.cast(x, tf.float64), self.w)

        # return x - tf.math.softplus(self.lin_output)  # Not the same softplus function as in PyTorch???
        return x - tf.math.softplus(
            tf.cast(self.lin_output, tf.float32)
        )  # Not the same softplus function as in PyTorch???


class ResNet(tf.keras.layers.Layer):
    """The ResNet class."""

    def __init__(self, n_in, n_out, n_layers=16):
        """Initialize the ResNetLayer architecture.

        Parameters
        ----------
        n_in : int
            Dimensionality of input.
        n_out : int
            Dimensionality of output.
        n_layers : int
            Number of residual layers.
        """
        super(ResNet, self).__init__()

        self.n_layers = n_layers

        # Define n_layers residual layer
        self.layers = [
            ResNetLayer(n_in, n_out) for _ in range(n_layers)
        ]  # Not sure if it's working

    # def forward(self, x):
    def call(self, x):
        """Return the final output of ResNet architecture.

        Parameters
        ----------
        x : tf.Variable
            Input of the first residual layer
        """
        out = x
        for i in range(self.n_layers):
            out = self.layers[i](out)
        return out


class ResLogit(Logit):
    """The ResLogit class."""

    def __init__(self, input, choice, n_vars, n_choices, n_layers=16):
        """Initialize the ResLogit class.

        Parameters
        ----------
        input : tf.Variable
        choice : tf.Variable
            Actual label
        n_vars : int
            Number of input variables.
        n_choices : int
            Number of choice alternatives.
        n_layers : int
            Number of residual layers.
        """
        Logit.__init__(self, input, choice, n_vars, n_choices)

        self.n_vars = n_vars
        self.n_choices = n_choices
        self.n_layers = n_layers

        # define the ResNet architecture.
        self.resnet_layer = ResNet(self.n_choices, self.n_choices, n_layers=16)
        for i in range(self.n_layers):
            self.params.extend(self.resnet_layer.layers[i].params)

    def fit(self, input):
        """Fit ResLogit according to the input.

        Parameters
        ----------
        input : tf.Variable
        """
        self.input = input
        if self.n_layers < 1:
            raise ValueError("Number of residual layers must be at least 1.")

        resnet_input = tf.matmul(self.input, self.beta)

        # output_resnet = self.resnet_layer.forward(resnet_input)
        output_resnet = self.resnet_layer.call(resnet_input)

        pre_softmax = output_resnet + self.asc

        self.output = tf.nn.softmax(pre_softmax, axis=1)

        self.output_pred = tf.math.argmax(self.output, axis=1)

    def predict(self, input):
        self.input = input
        if self.n_layers < 1:
            raise ValueError("Number of residual layers must be at least 1.")

        resnet_input = tf.matmul(self.input, self.beta)

        # output_resnet = self.resnet_layer.forward(resnet_input)
        output_resnet = self.resnet_layer.call(resnet_input)

        # Final output of residual layers
        # pre_softmax = output_resnet + self.asc
        pre_softmax = output_resnet + tf.cast(self.asc, tf.float32)

        # Output of the softmax layer
        self.output = tf.nn.softmax(pre_softmax, axis=1)

        # Final prediction
        self.output_pred = tf.math.argmax(self.output, axis=1)

        return self.output_pred


#############################################################################


class TrainingResLogit(object):
    """The TrainingResLogit class."""

    def main_model_reslogit(self, x, y, n_vars, n_choices, n_layers):
        self.model = ResLogit(
            input=x, choice=y, n_vars=n_vars, n_choices=n_choices, n_layers=n_layers
        )

        self.cost = tf.keras.losses.SparseCategoricalCrossentropy(reduction="sum")
        # self.opt = tf.keras.optimizers.RMSprop(
        #     self.model.params, learning_rate=1e-3, rho=0.9, epsilon=1e-10, weight_decay=0
        # )
        self.opt = tf.keras.optimizers.RMSprop(
            learning_rate=1e-3, rho=0.9, epsilon=1e-10, weight_decay=0
        )

    def main_model_logit(self, x, y, n_vars, n_choices):
        self.model = Logit(input=x, choice=y, n_vars=n_vars, n_choices=n_choices)

        self.cost = tf.keras.losses.SparseCategoricalCrossentropy(reduction="sum")
        self.opt = tf.keras.optimizers.RMSprop(
            self.model.params, learning_rate=1e-3, rho=0.9, epsilon=1e-10, weight_decay=0
        )

    def train_model(self, inputs, choice):
        self.model.fit(inputs)
        loss = self.cost(self.model.output, choice)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss

    def validate_model(self, valid_x, valid_y):
        with tf.stop_gradient():
            self.model.fit(valid_x)
            return self.cost(self.model.output, valid_y)

    def error(self, choice):
        return self.model.errors(choice)

    def predict_validate(self, inputs):
        return self.model.predict(inputs)

    def prob(self, inputs):
        self.model.fit(inputs)

        return self.model.output_likelihood

    def hessian(self, inputs, category="low"):
        self.model.fit(inputs)
        prob_fun = tf.Tensor(self.model.output_likelihood, requires_grad=True)
        with tf.GradientTape() as tape:
            if category == "high":
                outputs = prob_fun.sum(axis=0)
            else:
                outputs = 1 - prob_fun.sum(axis=0)

        return tape.gradient(outputs, inputs)
