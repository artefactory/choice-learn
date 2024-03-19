---
title: 'Choice-Learn: A Python package for generic choice modelling with large datasets.'
# Idea to introduce: ML&Classical, toolbox
tags:
  - Python
  - choice
  - decision
authors:
  - name: Vincent Auriau
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Author Without ORCID
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Author with no affiliation
    corresponding: true # (This is how to denote the corresponding author)
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

`Choice-Learn` aims at providing useful tools for academic researchs as well as practioners.(# Add information here)  In particular, the package focuses on three main points to extend choice models to more recent use-cases:
- Providing "classical" and "ML-based" literature choice models within the same codebase
- Possibility to work with very large datasets with RAM optimization and batching processes
- Creating an easy-to-use interface to build custom choice models

# Statement of need

With the fast-paced improvement of companies data architectures, larger reliable datasets emerge. Choice modelling is a natural tool for a retailer to understand its customer base and to improve or optimize its commercial offer. The large datasets now available open the door for the use of more complex machine learning models that can otherwise be difficult to estimate with little data. While several Python packages have been made available to estimate choice models [@Bierlaire:2023; @Brathwaite:2018] they are usually not built to work with large-scale datasets. The codebase is also usually not built to allow for users models customization [@Du:2023] making necessary the use of different packages that can make comparisons difficult.

> Reprendre les 3 piliers + (TensorFlow, prod ?) + ()

# Batching and RAM usage


# Choice model customization

# Acknowledgements`

# References
