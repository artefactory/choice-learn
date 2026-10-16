Here are some in-depth examples to help you with mastering Choice-Learn.
In particular you will find notebooks to handle:

<br>

**DATA**
- An introduction to data handling with Choice-Learn
- More in-depth examples to instantiate a ChoiceDataset
- Explanations and examples on how to use Features by IDs if you have RAM usage issues

<br>

**MODELS**

- *Linear Models* :
    - [MNL](./notebooks/models/simple_mnl.md): An introduction to choice modeling with the Multi Nomial Logit model
    - [cLogit](./notebooks/models/conditional_logit.md): Tutorials on how to parametrize and fit a Conditional Logit model
    - [nLogit](./notebooks/models/nested_logit.md): How to parametrize and fit a Nested Logit model
    - [Latent Class](./notebooks/models/latent_class_model.md): A basic examples on how to estimate several Latent Class models

- *Non-Linear Models* :
    - [RUMnet](./notebooks/models/rumnet.md):  Representing Random Utility Choice Models with Neural Networks (Aouad and Désir, 2023).
    - [TasteNet-MNL](./notebooks/models/tastenet.md): A neural-embedded discrete choice model: Learning taste representation with strengthened interpretability (Han, Pereira, Ben-Akiva and Zegras, 2022)

- *Diverse* :
    - [Logistic Regression](./notebooks/models/logistic_regression.md): A reproduction of the logistic regression tutorial by scikit-learn
    - [On model finetuning](./notebooks/models/models_finetuning.md): Hyperparameters and learning tools

<br>

**AUXILIARY TOOLS**

We currently handle two types of post-processing that leverage choice models:

- [Assortment Optimization](./notebooks/auxiliary_tools/assortment_example.md): How to best select a subset of alternatives to sell to a customer.
- [Pricing](./notebooks/auxiliary_tools/assortment_example.md): How to best select alternative prices - can be combined with assortment optimization.

If you feel like adding adding a dataset, a model, a tool or a usecase algorithm would bring value to the package, feel free to reach out !
