"""Utilities for running SAEBench evaluations on project transcoders."""

from .wrappers import (
    MatryoshkaAutoInterpAdapter,
    build_matryoshka_adapter,
    build_npz_transcoder_adapter,
)

from .config import AutoInterpSAEConfig, NPZTranscoderSpec
from .activation_pipeline import (
    AutoInterpActivationCache,
    AutoInterpDatasetSummary,
    prepare_auto_interp_cache,
)

__all__ = [
    "AutoInterpSAEConfig",
    "NPZTranscoderSpec",
    "MatryoshkaAutoInterpAdapter",
    "build_matryoshka_adapter",
    "build_npz_transcoder_adapter",
    "AutoInterpActivationCache",
    "AutoInterpDatasetSummary",
    "prepare_auto_interp_cache",
]

