# Geometry of Representations Analysis of Human Intracranial Neural Spiking Data from Movie Viewing and Recall

Second iteration of development.

Created data processing pipeline to:
- extract spike timing info from mat files
- filter and sort neurons
- bin neurons during concept onsets in movie and recall
- create datasets between two concept onsets, dropping shared onsets and accounting for temporal autocorrelations
- train linear classifiers to decode between concepts and concept groups
- perform shattering dimensionality and Cross-Conditional Generalization Performance on the data
