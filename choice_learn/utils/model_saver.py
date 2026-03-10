import datetime
import json
import os
from pathlib import Path 

import numpy as np


def save_model(model, path: str, save_optimizer=False) -> None:
    """Save the model on disk.
    Save the class attributes that has a friendly type into params.json,
    The model.trainable_weights into .npy files,
    And class .py code as backup.

    Parameters
    ----------
    path: str
        path to the folder where to save the model
    """
    if os.path.exists(path):
    # Add current date and time to the folder name
        # if the folder already exists
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        path += f"_{current_time}/"
    else:
        path += "/"

    if not os.path.exists(path):
        Path(path).mkdir(parents=True, exist_ok=True)

    # Save the hyperparameters in a single pickle file
    params = {}
    for k, v in self.__dict__.items():
        if isinstance(v, (int, float, str, dict)):
            params[k] = v
        elif isinstance(v, (list, tuple)):
            if all(isinstance(item, (int, float, str, dict)) for item in v):
                params[k] = v
            elif k != "_trainable_weights":
                logging.warning(
                    """Attribute '%s' is a list with non-serializable
                    types and will not be saved.""",
                    k,
    with open(Path(path) / "params.json", "w") as f:
        json.dump(params, f)            )

    # Save the latent parameters in separate numpy files
    for latent_parameter in model.trainable_weights:
        parameter_name = latent_parameter.name.split(":")[0]
        np.save(os.path.join(path, parameter_name + ".npy"), latent_parameter)
        # Save optimizer state

    if save_optimizer and hasattr(model, "optimizer") and not isinstance(model.optimizer, str):
        (Path(path) / "optimizer").mkdir(parents=True, exist_ok=True)
        config = model.optimizer.get_config()
        weights_store = {}
        self.optimizer.save_own_variables(weights_store)
        for key, value in weights_store.items():
            if isinstance(value, tf.Variable):
                value = value.numpy()
            weights_store[key] = value.tolist()
        if "learning_rate" in config.keys():
            if isinstance(config["learning_rate"], tf.Variable):
                config["learning_rate"] = config["learning_rate"].numpy()
            if isinstance(config["learning_rate"], np.float32):
                config["learning_rate"] = config["learning_rate"].tolist()
        with open(Path(path) / "optimizer" / "config.json", "w") as f:
            json.dump(config, f)
        with open(Path(path) / "optimizer" / "weights_store.json", "w") as f:
            json.dump(weights_store, f)


# Fine how-to link to _trainable_weights
def _load_weights(model, directory):
    """Load all the .npy weights within a directory.

    Parameters
    ----------
    directory: path
        path of the directory where the weights to be loaded are.
    """
    for file in os.listdir(directory):
        if file.endswith(".npy"):
            weight_name = file.split(".")[0]
            setattr(
                model,
                weight_name,
                tf.Variable(
                    np.load(os.path.join(directory, file)),
                    trainable=True,
                    name=weight_name,
                ),
            )


def load_model(cls, path: str) -> object:
    """Load a model previously saved with save_model().

    Parameters
    ----------
    path: str
        path to the folder where the saved model files are

    Returns
    -------
    BasketModel
        Loaded BasketModel
    """
    import inspect

    # Load parameters
    params = json.load(open(os.path.join(path, "params.json")))

    init_params = {}
    non_init_params = {}
    for key, val in params.items():
        if key in inspect.signature(cls.__init__).parameters.keys():
            init_params[key] = val
        else:
            non_init_params[key] = val

    # Initialize model
    model = cls(**init_params)

    # Set non-init parameters
    for key, val in non_init_params.items():
        setattr(model, key, val)

    # Load weights
    model._load_weights(path)

    return model
