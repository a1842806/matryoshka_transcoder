"""
Absorption Score Evaluation for Sparse Autoencoders

This module implements the feature absorption evaluation methodology from SAE Bench:
https://github.com/adamkarvonen/SAEBench

Feature absorption measures how much of a probe's signal is "absorbed" by other features
when the main feature fails to activate.
"""

from .feature_absorption_calculator import (
    FeatureAbsorptionCalculator,
    FeatureScore,
    WordAbsorptionResult,
    AbsorptionResults,
)
from .evaluation import (
    run_absorption_evaluation,
    evaluate_multiple_layers,
)
from .config import AbsorptionConfig
from .matryoshka_wrapper import (
    MatryoshkaTranscoderWrapper,
    load_matryoshka_transcoder,
)

__all__ = [
    'FeatureAbsorptionCalculator',
    'FeatureScore',
    'WordAbsorptionResult',
    'AbsorptionResults',
    'run_absorption_evaluation',
    'evaluate_multiple_layers',
    'AbsorptionConfig',
    'MatryoshkaTranscoderWrapper',
    'load_matryoshka_transcoder',
]
