"""TasteNet model unofficial implementation."""

import tensorflow as tf

from choice_learn.models.base_model import ChoiceModel

def get_feed_forward_net(input_width, output_width, layers_width, activation):
    """Base function to get a feed-forward neural network."""
    net_input = tf.keras.layers.Input(shape=(input_width,))
    net_output = net_input
    for n_units in layers_width:
        net_output = tf.keras.layers.Dense(n_units, activation=activation)(net_output)
    net_output = tf.keras.layers.Dense(output_width, activation=None)(net_output)
    return tf.keras.Model(inputs=net_input, outputs=net_output)


class TasteNet():

    def __init__(self,
                 taste_net_layers,
                 taste_net_activation,
                 items_features_by_choice_parametrization,
                 **kwargs):
        """Initialization of the model."""
        super().__init__(**kwargs)
        self.taste_net_layers = taste_net_layers
        self.taste_net_activation = taste_net_activation
        self.items_features_by_choice_parametrization = items_features_by_choice_parametrization

    def get_activation_function(self, name):
        if name == 'relu':
            return tf.nn.relu
        elif name == "-relu":
            return lambda x: -tf.nn.relu(-x)
        elif name == "exp":
            return tf.exp
        elif name == "-exp":
            return lambda x: -tf.exp(-x)
        elif name == 'tanh':
            return tf.nn.tanh
        elif name == 'sigmoid':
            return tf.nn.sigmoid
        else:
            raise ValueError(f'Activation function {name} not supported.')
        
        self.instantiated  = False

    def instantiate(self, n_shared_features):
        """Instantiates the model.

        Parameters
        ----------
        n_shared_features : _type_
            _description_
        """
        # TODO: Add possibility for MNL-type weights
        items_features_to_weight_index = {}
        for i, item_param in enumerate(self.items_features_by_choice_parametrization):
            for j, param in enumerate(item_param):
                if isinstance(param, str):
                    items_features_to_weight_index[(i, j)] = len(items_features_to_weight_index)
        self.items_features_to_weight_index = items_features_to_weight_index

        self.taste_params_module = get_feed_forward_net(n_shared_features,
                                                        len(items_features_to_weight_index),
                                                        self.taste_net_layers,
                                                        self.taste_net_activation)
        self.instantiated = True
    @property
    def trainable_weights(self):
        if self.instantiated:
            return self.taste_params_module.trainable_variables
        else:
            return []
        
    def compute_batch_utility(self,
                              shared_features_by_choice,
                              items_features_by_choice,
                              available_items_by_choice,
                              choices):
        """Computes the utility of the choices for the given batch."""

        _ = choices
        taste_weights = self.taste_params_module(shared_features_by_choice)
        item_utility_by_choice = tf.zeros_like(available_items_by_choice, dtype=tf.float32)

        item_utility_by_choice = []
        for i, item_param in enumerate(self.items_features_by_choice):
            utility = tf.zeros_like(choices, dtype=tf.float32)
            for j, param in enumerate(item_param):
                if isinstance(param, str):
                    item_feature = self.get_activation_function(param)(items_features_by_choice[:, i, j])
                    item_feature = taste_weights[:, self.items_features_to_weight_index[(i, j)]] * item_feature
                elif isinstance(param, float):
                    item_feature = param * items_features_by_choice[:, i, j]
                utility += item_feature
            item_utility_by_choice.append(utility)
        return tf.stack(item_utility_by_choice, axis=1)
