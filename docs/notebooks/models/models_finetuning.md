# Model finetuning: generic hyper-parameters

A few hyperparameters are shared by most models and can be used to improve model's estimation process and generalization.

## Optimizer

Two main types of optimizers are available for model estimation with Choice-Learn. You should choose among these two types depending on the model you wish to estimate and the size of your dataset.

### *Quasi-Newton methods:*
For smaller models and datasets you should use the L-BFGS algorithm by specifying `optimizer="lbfgs"` in the model instantiation. The algorithm is faster but has a high memory consumption since it needs to evaluate the whole dataset negative log-likelihood for several sets of parameters.
You can find more information on the algorithm on [Wikipedia](https://en.wikipedia.org/wiki/Limited-memory_BFGS) and on the implementation on the official [TensorFlow documentation](https://www.tensorflow.org/probability/api_docs/python/tfp/optimizer/lbfgs_minimize).

#### Tolerance

Choice-Learn models have a hyperparameter called `tolerance` that controls the precision of the L-BFGS algorithm. You can change it to either converge faster or get better precision.
The default value is $10^{-8}$ and the algorithm will consider to have reached an optimal when the objective function value has a different smaller than `tolerance` between two steps.

#### Epochs
When using L-BFGS you can also set up the `epochs` hyperparameter to specify the maximum number of iteration accepted. If an optimal respecting tolerance is not found within this number of iterations it is stopped nonetheless.

### *Gradient descent methods:*

If you have memory error using L-BFGS you can use gradient-descent base algorithms that are particularly popular for neural networks estimation since they have many parameters.
[Different versions](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers) exist among which [SGD](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD), [Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam) and [Adamax](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adamax) are particularly popular. You can select them by specifying `optimizer="Adam"` in the model instantiation.

#### Learning Rate

Gradient descent algorithms can be parametrized with the `lr` argument. The learning rate basically control the step size of each weight update. Default value is $10^{-3}$.

#### Batch Size

The batch size hyperparameters sets up how many data (= choices) should be used in each update of the gradient descent algorithm. A compromise has to be found between higher values taking up a more memory that are more stable but can become slower in terms of convergence time and smaller values with fast iterations but that also introduce more noise.
The default value is `batch_size=32`.

#### Epochs

You can also control the number of epochs of gradient descent algorithms with the `epochs` argument. The default value being 1000 you are strongly advised to change it.

### Callbacks

You can also use [tf.keras.callbacks](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback) to have learning rate variation or early stopping strategies. You should create it in pure TensorFlow and give it to the model as a list of callbacks: `callbacks=[...]`.

## Regularization

A weight regularization can be added during model estimation. The regularization type can be chosen among $L_1$ and $L_2$ and an importance coefficient factor can be specified.
For example:
```model = SimpleMNL(regularization_type="l2", regularization_strength=0.001, **kwargs)```.

For a short reminder this will set the following estimation objective:
$$
\arg \max \sum_i log \left[ \mathbb{P}(c_i|\mathcal{A}_i) \right] + 0.001 \cdot ||W||_2
$$

With $W$ the model's parameters and $\mathbb{P}(c_i|\mathcal{A}_i)$ the model's probability to choose the chosen item $c_i$.

## Exit Choice
You can specify if you want to integrate an exit choice or not during the model's estimation with the argument `add_exit_choice`. The default value is `False`, if set to `True`, the probabilities will integrate an exist choice of utility zero.

## Label Smoothing

You can integrate a label smoothing that sometimes help the optimization process. It is more recommended with descent gradient algorithms.

Basically, the probabilities are float in [0, 1]. When > 0, label values are smoothed, meaning the confidence on label values is relaxed. For example, if 0.1, use 0.1 / num_items for non-chosen labels and 0.9 + 0.1 / num_classes for chosen labels.

More explanations [here](https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy).
## Final Word

If you want or need more details checkout the [references page](./../../references/models/references_base_model.md) of models.
