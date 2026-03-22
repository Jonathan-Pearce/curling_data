"""Graph ML Siamese GNN pipeline for curling shot analysis.

This package implements a CPU-compatible Siamese Graph Neural Network
for multi-task prediction of curling shot outcomes:

- End score regression (MSE)
- Shot accuracy regression (MSE)
- Shot type classification (Cross-entropy)

Architecture follows Kendall et al. (2018) homoscedastic uncertainty
weighting for multi-task loss balancing.
"""
