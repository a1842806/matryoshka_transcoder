# Activation Sample Collection for SAE Interpretability

This feature enables collection and analysis of high-activation samples for Sparse Autoencoder (SAE) features, following [Anthropic's approach](https://transformer-circuits.pub/2023/monosemantic-features) to interpretability research.

## Overview

The activation sample collection system automatically tracks and saves the input samples (text tokens and context) that cause the highest activations for each feature in your SAE. This enables you to:

- **Understand what each feature represents** by examining real examples
- **Validate feature monosemanticity** - checking if features respond to coherent concepts
- **Debug feature behavior** - identifying polysemantic or dead features
- **Generate interpretability reports** for research and analysis

## Quick Start

### 1. Enable Sample Collection in Training

```python
from utils.config import get_default_cfg, post_init_cfg

cfg = get_default_cfg()

# Enable activation sample collection
cfg["save_activation_samples"] = True
cfg["sample_collection_freq"] = 1000     # Collect every 1000 steps
cfg["max_samples_per_feature"] = 100     # Store top 100 samples per feature
cfg["sample_context_size"] = 20          # 20 tokens of context
cfg["sample_activation_threshold"] = 0.1 # Only activations > 0.1

cfg = post_init_cfg(cfg)
```

### 2. Train Your SAE

```python
from training.training import train_sae
from models.sae import BatchTopKSAE

sae = BatchTopKSAE(cfg)
train_sae(sae, activation_store, model, cfg)
```

Sample collection happens automatically during training. At the end, samples are saved to:
```
checkpoints/sae/{model_name}/{run_name}_activation_samples/
```

### 3. Analyze Collected Samples

```python
from utils.analyze_activation_samples import ActivationSampleAnalyzer

analyzer = ActivationSampleAnalyzer("path/to/samples/")

# Print report for a specific feature
analyzer.print_feature_report(feature_idx=42, num_examples=10)

# Generate comprehensive markdown report
analyzer.generate_interpretability_report(
    "report.md",
    top_k_features=50,
    examples_per_feature=10
)

# Compare features
analyzer.compare_features(feature_idx1=10, feature_idx2=20)

# Find similar features
analyzer.find_similar_features(feature_idx=42, top_k=5)
```

## Configuration Parameters

Add these to your config dictionary:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `save_activation_samples` | `False` | Enable/disable sample collection |
| `sample_collection_freq` | `1000` | Collect samples every N training steps |
| `max_samples_per_feature` | `100` | Maximum samples to store per feature |
| `sample_context_size` | `20` | Number of tokens of context around activation |
| `sample_activation_threshold` | `0.1` | Minimum activation value to consider storing |
| `top_features_to_save` | `100` | Number of top features to save samples for |
| `samples_per_feature_to_save` | `10` | Number of samples to save per feature to disk |

## Example Output

When you print a feature report, you'll see:

```
================================================================================
FEATURE 42 INTERPRETABILITY REPORT
================================================================================

Statistics:
  Max activation: 0.8734
  Mean activation: 0.4521
  Std activation: 0.1234
  Total activations seen: 1523
  Samples stored: 100

Top 5 Activating Examples:
--------------------------------------------------------------------------------

[Example 1] Activation: 0.8734
  Activating text: ' cat'
  Context: ...the fluffy cat sat on the warm mat...
  Position in sequence: 15

[Example 2] Activation: 0.8521
  Activating text: ' cat'
  Context: ...a curious cat explored the garden...
  Position in sequence: 8

...
```

## Storage and Performance Impact

### Storage Requirements

For a typical SAE with 36,864 features:
- **Per sample**: ~200-500 bytes (activation value + tokens + text)
- **Per feature**: 100 samples × 300 bytes = ~30 KB
- **Total for 100 features**: ~3 MB per checkpoint

### Computational Overhead

- **Training time increase**: ~5-10% (depends on `sample_collection_freq`)
- **Memory usage**: +15-25% during sample collection steps
- **Disk I/O**: Minimal (only at end of training)

### Recommended Settings

**For full interpretability research:**
```python
cfg["save_activation_samples"] = True
cfg["sample_collection_freq"] = 500
cfg["max_samples_per_feature"] = 200
cfg["top_features_to_save"] = 500
```

**For lightweight monitoring:**
```python
cfg["save_activation_samples"] = True
cfg["sample_collection_freq"] = 5000
cfg["max_samples_per_feature"] = 50
cfg["top_features_to_save"] = 50
```

**For production training (disable):**
```python
cfg["save_activation_samples"] = False
```

## File Structure

After training with sample collection enabled:

```
checkpoints/sae/gemma-2-2b/my_model_activation_samples/
├── collection_summary.json           # Overall statistics
├── feature_00042_samples.json        # Samples for feature 42
├── feature_00123_samples.json        # Samples for feature 123
├── ...
└── interpretability_report.md        # Auto-generated report
```

### Sample File Format

Each `feature_XXXXX_samples.json` contains:

```json
{
  "statistics": {
    "feature_idx": 42,
    "max_activation": 0.8734,
    "mean_activation": 0.4521,
    "num_samples": 100,
    "activation_count": 1523
  },
  "top_samples": [
    {
      "feature_idx": 42,
      "activation_value": 0.8734,
      "tokens": [262, 3797],
      "text": " cat",
      "position": 15,
      "context_tokens": [262, 39145, 3797, 3332, 319, 262, ...],
      "context_text": "the fluffy cat sat on the warm mat"
    },
    ...
  ]
}
```

## Analysis Tools

### Command-Line Analysis

```bash
# Analyze saved samples
python src/utils/analyze_activation_samples.py checkpoints/sae/gemma-2-2b/my_model_activation_samples/

# Run example with visualization
python src/scripts/example_activation_sample_collection.py demo
```

### Programmatic Analysis

```python
from utils.activation_samples import ActivationSampleCollector

# Load existing samples
collector = ActivationSampleCollector()
collector.load_samples("path/to/samples/")

# Get top samples for a feature
samples = collector.get_top_samples(feature_idx=42, k=10)

# Print examples to console
collector.print_feature_examples(feature_idx=42, k=5)

# Get statistics
stats = collector.get_feature_statistics(feature_idx=42)
print(f"Max activation: {stats['max_activation']}")
```

## Testing

Run comprehensive tests to verify functionality:

```bash
cd /home/datngo/User/matryoshka_sae
python tests/test_activation_samples.py
```

Tests cover:
- ✅ Basic sample collection functionality
- ✅ Heap-based storage with size limits
- ✅ Activation threshold filtering
- ✅ Context extraction
- ✅ Sample saving and loading
- ✅ Feature interpretability patterns
- ✅ Feature specialization
- ✅ Edge cases and error handling

## Integration with Existing Training Scripts

### For SAE Training

```python
# In your training script
from training.training import train_sae

cfg["save_activation_samples"] = True
cfg["sample_collection_freq"] = 1000

# Train as normal - samples collected automatically
train_sae(sae, activation_store, model, cfg)
```

### For Transcoder Training

```python
# In your transcoder training script
from training.training import train_transcoder

cfg["save_activation_samples"] = True
cfg["sample_collection_freq"] = 1000

# Train as normal
train_transcoder(transcoder, activation_store, model, cfg)
```

## Best Practices

### 1. Collection Frequency
- **Early training**: Collect less frequently (every 5000 steps) to save compute
- **Later training**: Collect more frequently (every 500 steps) when features stabilize
- **Final checkpoint**: Always collect samples for interpretability analysis

### 2. Feature Selection
- Focus on top 50-100 most active features for initial analysis
- Investigate dead or rarely-activating features separately
- Compare similar features to identify redundancy

### 3. Context Size
- Use **20-30 tokens** for general interpretability
- Use **50+ tokens** for complex features requiring more context
- Balance between interpretability and storage

### 4. Activation Threshold
- Start with **0.1** for broad coverage
- Increase to **0.3-0.5** to focus on strong activations only
- Lower to **0.05** if features rarely activate

## Troubleshooting

### Issue: No samples collected
- Check that `save_activation_samples = True`
- Verify `sample_activation_threshold` isn't too high
- Ensure `sample_collection_freq` is reached during training

### Issue: Out of memory during collection
- Reduce `max_samples_per_feature`
- Increase `sample_collection_freq` (collect less often)
- Enable `eval_compute_on_cpu` if available

### Issue: Samples not interpretable
- Check tokenizer decoding
- Increase `sample_context_size` for more context
- Verify feature is actually activating (check statistics)

## API Reference

### ActivationSampleCollector

Main class for collecting activation samples.

```python
collector = ActivationSampleCollector(
    max_samples_per_feature=100,
    context_size=20,
    storage_threshold=0.1
)
```

**Methods:**
- `collect_batch_samples(activations, tokens, tokenizer)` - Collect samples from a batch
- `get_top_samples(feature_idx, k)` - Get top-k samples for a feature
- `get_feature_statistics(feature_idx)` - Get statistics for a feature
- `save_samples(save_dir)` - Save all samples to disk
- `load_samples(save_dir)` - Load samples from disk
- `print_feature_examples(feature_idx, k)` - Print examples to console

### ActivationSampleAnalyzer

Analyzer for visualizing and analyzing collected samples.

```python
analyzer = ActivationSampleAnalyzer("path/to/samples/")
```

**Methods:**
- `print_feature_report(feature_idx, num_examples)` - Print detailed feature report
- `generate_interpretability_report(output_file)` - Generate markdown report
- `analyze_feature_clustering()` - Analyze feature activation patterns
- `compare_features(feature_idx1, feature_idx2)` - Compare two features
- `find_similar_features(feature_idx, top_k)` - Find similar features
- `plot_activation_distribution(feature_idx)` - Plot activation histogram

## Citation

If you use this feature in your research, please cite both this codebase and Anthropic's original work:

```bibtex
@article{anthropic2023monosemantic,
  title={Towards Monosemanticity: Decomposing Language Models With Dictionary Learning},
  author={Anthropic},
  journal={Transformer Circuits Thread},
  year={2023},
  url={https://transformer-circuits.pub/2023/monosemantic-features}
}
```

## Related Documentation

- [Training Guide](./TRANSCODER_QUICKSTART.md) - How to train SAEs and Transcoders
- [Feature Evaluation Guide](./FEATURE_EVALUATION_GUIDE.md) - Other interpretability metrics
- [Project Structure](./PROJECT_STRUCTURE.md) - Overall codebase organization

## Examples

See `src/scripts/example_activation_sample_collection.py` for complete working examples.

