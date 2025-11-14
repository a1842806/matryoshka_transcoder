"""
Sparse Probing Evaluation Module

This module implements sparse probing evaluation for SAE Bench.
For each SAE feature, it trains linear probes to predict concepts,
evaluating how well individual features correspond to interpretable concepts.
"""

from .config import SparseProbingConfig
from .evaluation import evaluate_feature_probing, evaluate_multiple_layers

__all__ = [
    'SparseProbingConfig',
    'evaluate_feature_probing',
    'evaluate_multiple_layers',
]

