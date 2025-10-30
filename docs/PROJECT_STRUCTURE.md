# Matryoshka SAE Project Overview

This guide stays high level so the training and evaluation documents can focus on execution details.

## Purpose

- **Matryoshka sparse autoencoders** learn hierarchical feature dictionaries that can be truncated without retraining.
- **Transcoders** map activations between transformer layers/sites while preserving interpretability through nested groups and top-k sparsity.
- The repository supports research on interpretable feature discovery for models such as GPT-2 and Gemma-2.

## Repository Layout

| Path | Contents | Notes |
|------|----------|-------|
| `src/models/` | SAE + transcoder implementations | Includes Matryoshka encoders, activation stores, adapter utilities |
| `src/training/` | Training loops and orchestration | Central entry point for optimizer logic and run lifecycle |
| `src/scripts/` | CLI entry points | Thin wrappers that call into `src/training/`
| `src/utils/` | Config, logging, data helpers | Houses defaults, checkpoint helpers, and sampling utilities |
| `src/adapters/` | Integration glue for external benchmarks | Keeps third-party APIs isolated from core code |
| `results/` | Standardized training run outputs | Naming enforced by the training utilities (see below) |
| `checkpoints/` | Legacy checkpoints | Gradually converging with `results/`
| `docs/` | This documentation set | Only three focused guides remain after consolidation |

### Standard Results Layout

All new training runs should save artifacts under:

```
results/{model_type}/{layer_number}/{step_count}/
    ├── config.json    # Stores run configuration + completion date
    ├── checkpoints/   # Optional: weights, shards, or tokenizer state
    └── metrics.json   # Optional: summary statistics
```

Keeping this structure consistent allows downstream scripts to enumerate experiments without bespoke parsing logic.

## Key Components

- `MatryoshkaTranscoder` exposes nested groups with cumulative decoders for progressive reconstructions.
- `TranscoderActivationsStore` collects paired activations (source/target) for cross-layer training.
- Config helpers define model-specific defaults (e.g., GPT-2 vs Gemma-2 hidden sizes, group sizes, top-k).
- Activation sampling utilities capture the highest-activation examples per latent for interpretability analysis.

Consult `docs/TRAINING_GUIDE.md` for practical usage and `docs/EVALUATION_GUIDE.md` for analysis workflows.

## Status Snapshot

- **Infrastructure**: modular `src/` layout, multi-model hooks, logging, and checkpoint helpers are stable.
- **Training Coverage**: GPT-2 Small and Gemma-2-2B Matryoshka transcoders have completed runs with healthy dead-feature rates and absorption scores.
- **Interpretability Tools**: activation sample collection and SAEBench-compatible evaluation are being rebuilt after the evaluation cleanup.
- **Near-Term Focus**: deeper layer coverage, cross-layer transcoders, and automated feature reports.

## Documentation Map

- `docs/TRAINING_GUIDE.md` – environment setup, configuration patterns, nested groups, activation sampling, GPU notes.
- `docs/EVALUATION_GUIDE.md` – interpretability metrics, AutoInterp alternatives, evaluation workflow.
- `README.md` – quick project introduction and onboarding pointers.

Keep long-form experiment logs in external notebooks or reports; the docs directory is intentionally compact.

