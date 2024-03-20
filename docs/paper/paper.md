---
title: 'Choice-Learn: A Python package for generic choice modelling with large datasets.'
# Idea to introduce: ML&Classical, toolbox
# A Python Toolbox for generic and custom choice modelling ?
tags:
  - Python
  - choice
  - decision
authors:
  - name: Vincent Auriau
    corresponding: true # (This is how to denote the corresponding author)
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Author Without ORCID
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Author with no affiliation
    affiliation: 3
  - given-names: Ludwig
    dropping-particle: van
    surname: Beethoven
    affiliation: 3
affiliations:
 - name: CentraleSupélec, université Paris-Saclay, France
   index: 1
 - name: Artefact Research Center, France
   index: 2
 - name: Independent Researcher, Country
   index: 3
date: 13 August 2017
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Discrete choice models aim at explaining or predicting a choice from a set of alternatives. Well known use-cases include analyzing people choice of mean of transport or products purchases in stores. One key attribute of choice models is their ability to handle sets of variable sizes, with some alternatives being possibly unavailable. Choice models can be used to estimate interpretable values such as a consumer's price elasticity. Once estimated, they can also be used in second processing step such as assortment optimization or pricing. Recent outbreaks in the Machine-Learning community called for the use of more complex models and larger datasets in the estimation of choice models.

`Choice-Learn` aims at providing useful tools for academic researchs as well as practioners.(# Add information here)  In particular, the package focuses on three main points to extend choice modelling tools:
- Providing "classical" and "MachineLearning-based" literature choice models within the same codebase
- Possibility to work with very large datasets with RAM optimization and batching processes
- Creating an easy-to-use interface to build custom choice models

# Statement of need

### Small introduction on choice modelling

With the fast-paced improvement of companies data architectures, larger reliable datasets emerge. Choice modelling is a natural tool for a retailer to understand its customer base and to improve or optimize its commercial offer. The large datasets now available open the door for the use of more complex machine learning models that can otherwise be difficult to estimate with little data. While several efficient Python packages have been made available to estimate choice models [@Bierlaire:2023; @Brathwaite:2018] they are usually not built to work with large-scale datasets.

With these large datasets comes the possibility to use more complex models. Recent publications outlines this possibility with neural networks approaches [@Han:2022, @Aouad:2023] or tree-based boosting models [@Salvadé:2024]. The existing libraries [@Bierlaire:2023; @Brathwaite:2018, @Du:2023] are usually not built to allow for models customization making necessary the use of different packages that can make comparisons difficult.

Choice-Learn is organized around three pillars (?) that can be used independantly or altogether.
- Dataset handling: a NumPy [@Harris:2020] based method to create batches of data is proposed. By limiting data replication, it optimizes the memory usage.
- Choice models: Choice-Learn proposes ready-to-use models with a Python interface such as the Conditional-MNL[@Train:1987] or RUMnet[@Aouad:2023]. Based on Tensorflow[@Abadi:2015], the implementation ensures efficient learning with the different available optimization algorithm, and offers GPU compatibility. Choice-Learn also aims at helping for building new and custom choice models with a common inheritance scheme that minimizes the user's work. Compared to usual implementations non linear formulations of utility are possible.
- Tools: Choice Models can be used for usecases such as assortment optimization. The models signature let easily use the model's output. Implementations are also proposed.


# Batching and RAM usage

Choice models estimate a utility function from which a probability to choose each alternative is derived.

# Choice model customization

Inheriting the ChoiceModel class lets the user define its own choice model. One only needs to precise how to compute the utility of a batch of data using TensorFlow operations. Here is an example.
### Check example > What would be a great example ?

```python
from tensorflow.keras.layers import Dense
from choice_learn.models import ChoiceModel

class ExampleCustomizedModel(ChoiceModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # First non-linear layer
        self.dense_1 = Dense(units=10, activation="elu")
        # Second linear layer
        self.dense_2 = Dense(units=1, activation="linear")
        # We do not forget to specify self.weights with all coefficients that need to be estimated.
        # Easy with TensorFlow.Layer
        self.weights = self.dense_1.trainable_variables + self.dense_2.trainable_variables

    def compute_batch_utility(self,
                        fixed_items_features,
                        contexts_features,
                        contexts_items_features,
                        contexts_items_availabilities,
                        choices):
        # We apply the neural network to all sessions_items_features for all the items
        # We then concatenate the utilities of each item of shape (n_sessions, 1) into a single one of shape (n_sessions, n_items)
        u = tf.concat([self.dense_2(self.dense_1(contexts_items_features[0][:, i])) for i in range(contexts_items_features[0].shape[1])], axis=1)
        return u
```

# Acknowledgements

# References
