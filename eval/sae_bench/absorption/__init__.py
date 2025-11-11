"""
Absorption Score Evaluation for Sparse Autoencoders

This module implements the feature absorption evaluation methodology from:
"A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders"
https://arxiv.org/pdf/2409.14507v3

Feature absorption occurs when an SAE/transcoder latent appears to track a concept 
but fails to activate on arbitrary tokens where token-aligned latents activate instead.
"""

from .absorption_metric import AbsorptionMetric, compute_absorption_score
from .first_letter_task import FirstLetterTask, train_linear_probe
from .ablation import (
    FeatureAblator,
    integrated_gradients_ablation,
    compute_ablation_effect,
    batch_ablation_analysis
)
from .evaluation import run_absorption_evaluation

__all__ = [
    'AbsorptionMetric',
    'compute_absorption_score',
    'FirstLetterTask',
    'train_linear_probe',
    'FeatureAblator',
    'integrated_gradients_ablation',
    'compute_ablation_effect',
    'batch_ablation_analysis',
    'run_absorption_evaluation',
]

