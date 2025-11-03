# Matryoshka SAE: Project Overview & Repository Structure

## Purpose

- Matryoshka sparse autoencoders learn hierarchical feature dictionaries that can be truncated without retraining.
- Transcoders map activations between transformer layers/sites while preserving interpretability via nested groups and top-k sparsity.
- This repository supports research workflows for interpretable feature discovery on models such as GPT-2 and Gemma-2.

## Repository Layout

| Path | Contents | Notes |
|------|----------|-------|
| `src/models/` | SAE + transcoder implementations | Matryoshka encoders/decoders, activation stores |
| `src/training/` | Training loops and orchestration | Optimizer logic and run lifecycle |
| `src/scripts/` | CLI entry points | Thin wrappers that call into `src/training/` |
| `src/utils/` | Config, logging, data helpers | Defaults, checkpoint helpers, sampling utilities |
| `results/` | Standardized run outputs | See layout below |
| `checkpoints/` | Legacy checkpoints | Converging with `results/` over time |
| `docs/` | This documentation set | Consolidated into five focused guides |

### Standard Results Layout

All training runs are saved using the standardized layout:

```
results/{model_name}/layer{layer}/{steps}/
    ├── config.json         # Run configuration (final includes completed_at)
    ├── checkpoints/        # Weights (final.pt) and/or interim step_*.pt
    ├── metrics/step_*.json # Interim metric snapshots (optional)
    ├── metrics.json        # Final summary metrics
    └── activation_samples/ # Optional interpretability samples
```

Keeping this structure consistent allows downstream scripts to enumerate experiments without bespoke parsing logic.

## Key Components

- `MatryoshkaTranscoder` exposes nested groups with cumulative decoders for progressive reconstructions.
- `TranscoderActivationsStore` collects paired activations (source/target) for cross-layer training.
- Config helpers define model-specific defaults (e.g., GPT-2 vs Gemma-2 hidden sizes, prefix sizes, top-k).
- Activation sampling utilities capture highest-activation examples per latent for interpretability analysis.

## Documentation Map

- `docs/02_TRAINING_AND_CONFIGURATION.md` — environment setup, configuration patterns, nested groups refresher, GPU notes, Gemma-specific hooks.
- `docs/03_EVALUATION_AND_METRICS.md` — core metrics, SAEBench-compatible procedures, AutoInterp backends, workflow.
- `docs/04_MATRYOSHKA_TRANSCODER_AND_NESTED_GROUPS.md` — architecture, nested groups, quick start, configs, troubleshooting.
- `docs/05_INTERPRETABILITY_TOOLS_ACTIVATION_SAMPLES.md` — activation sample collection and analysis tools.

Tip: keep long-form experiment logs in notebooks or external reports; the `docs/` directory is intentionally concise and task-focused.


