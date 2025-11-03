# Interpretability Tools: Activation Sample Collection

Activation sample collection helps you understand what each latent represents by saving top-activating text snippets and context for later analysis.

## Quick Start

Enable in your training configuration:

```python
cfg["save_activation_samples"] = True
cfg["sample_collection_freq"] = 1000
cfg["max_samples_per_feature"] = 100
cfg["sample_context_size"] = 20
cfg["sample_activation_threshold"] = 0.1
```

Samples are stored alongside the run under `activation_samples/` and can be analyzed post hoc.

## Analyzing Collected Samples

```python
from utils.analyze_activation_samples import ActivationSampleAnalyzer

analyzer = ActivationSampleAnalyzer("path/to/samples/")
analyzer.print_feature_report(feature_idx=42, num_examples=10)
analyzer.generate_interpretability_report("report.md", top_k_features=50, examples_per_feature=10)
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `save_activation_samples` | `False` | Enable/disable sample collection |
| `sample_collection_freq` | `1000` | Collect every N steps |
| `max_samples_per_feature` | `100` | Cap per latent |
| `sample_context_size` | `20` | Context window (tokens) |
| `sample_activation_threshold` | `0.1` | Min activation stored |
| `top_features_to_save` | `100` | Features to persist |
| `samples_per_feature_to_save` | `10` | Samples per feature to write |

## Storage & Performance

- Storage: ~30 KB per feature (100 samples) typical; a few MB per checkpoint for a subset of features.
- Overhead: +5–10% training time and +15–25% memory during collection steps.
- Optimizations: heap-based top-k storage, periodic collection, thresholds, selective saving.

## File Structure

```
checkpoints/sae/{model}/{run}_activation_samples/
├── collection_summary.json
├── feature_00042_samples.json
└── interpretability_report.md
```

## CLI Utilities

```bash
python src/utils/analyze_activation_samples.py checkpoints/sae/gemma-2-2b/my_run_activation_samples/
```

## Best Practices

- Collect less frequently early in training; more frequently once features stabilize.
- Start with ~50–100 most active features for manual inspection.
- Use 20–30 tokens of context for readability; increase for complex features.

## Troubleshooting

- No samples collected: ensure the flag is enabled and thresholds are reasonable.
- OOM during collection: reduce `max_samples_per_feature` or collect less often.
- Unclear examples: increase context size; verify the feature actually activates.

For metric computation and evaluation workflows, see `docs/03_EVALUATION_AND_METRICS.md`.


