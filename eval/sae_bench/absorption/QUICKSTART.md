# Absorption Score Evaluation - Quick Start Guide

## What is Feature Absorption?

Feature absorption is a problematic phenomenon in Sparse Autoencoders (SAEs) and transcoders where:
- A feature **appears** to track a concept (e.g., "starts with S")
- But **fails to activate** on seemingly random tokens that should trigger it
- Instead, **token-specific features** activate and "absorb" the concept direction
- This makes features unreliable classifiers for safety-critical applications

## Quick Start

### 1. Evaluate a Single Layer

```bash
cd eval/sae_bench/absorption
python run_evaluation.py --layer 3 --model gemma-2-2b
```

This will:
- Load your trained Matryoshka transcoder for layer 3
- Create a first-letter classification dataset
- Train linear probes as ground truth
- Identify transcoder features for each letter
- Measure absorption rates and classification performance
- Save results to `results/absorption/gemma-2-2b/layer3/`

### 2. Run the Example Script

```bash
cd eval/sae_bench/absorption
python example.py
```

This runs a simplified evaluation with detailed output.

### 3. Evaluate Multiple Layers

```bash
python run_evaluation.py --layers 0 1 2 3 4 5
```

Or all layers (0-17):
```bash
python run_evaluation.py --all-layers
```

## Understanding the Output

### Key Metrics

**Classification Performance:**
- **F1 Score**: Balance of precision and recall (higher is better, max 1.0)
  - Linear probes typically achieve ~0.95
  - Good transcoder features: > 0.8
  - Problematic features: < 0.6

- **Precision**: Of activations, how many are correct?
  - High precision but low recall → absorption likely occurring

- **Recall**: Of correct tokens, how many activate the feature?
  - Low recall → feature missing many tokens it should catch

**Absorption Metrics:**
- **Absorption Rate**: Fraction of correct tokens where main feature doesn't fire
  - Low (< 0.2): Good, minimal absorption
  - Medium (0.2-0.5): Some absorption, investigate
  - High (> 0.5): Serious absorption problem

- **Absorbing Features**: Number of other features that fire instead
  - Should be 0 for perfectly monosemantic features
  - High numbers indicate feature splitting/absorption

### Example Output

```
Letter 's':
  Feature 1234 (cosine=0.85)
    Precision: 0.92
    Recall: 0.45      ← Low recall despite high precision!
    F1: 0.60
    Detecting absorption...
    Absorption rate: 0.55    ← High absorption!
    Absorbed tokens: 550/1000
    Absorbing features: 12   ← Many features absorbing
```

This shows a problematic feature with high absorption.

## Using Different Prefix Levels

Matryoshka transcoders have nested feature groups. Evaluate a specific level:

```bash
python run_evaluation.py --layer 3 --prefix-level 0  # Smallest (2304 features)
python run_evaluation.py --layer 3 --prefix-level 4  # Largest (18432 features)
python run_evaluation.py --layer 3                    # Full model (all features)
```

## Configuration Options

### Basic Options
- `--layer N`: Evaluate layer N
- `--layers N1 N2 ...`: Evaluate multiple layers
- `--all-layers`: Evaluate layers 0-17
- `--model NAME`: Model name (default: gemma-2-2b)
- `--device DEVICE`: cuda:0, cuda:1, or cpu

### Advanced Options
- `--num-samples N`: Number of tokens to sample (default: 10000)
- `--batch-size N`: Batch size (default: 32)
- `--cosine-threshold T`: Threshold for absorption detection (default: 0.3)
- `--ablation-threshold T`: Threshold for causality (default: 0.01)
- `--no-causality-check`: Skip ablation (faster but less rigorous)
- `--results-dir PATH`: Output directory

## Interpreting Results for Your Transcoders

### Good Performance
```
Mean F1: 0.82 ± 0.05
Mean absorption rate: 0.15 ± 0.08
```
- Features are learning interpretable, reliable concepts
- Minimal absorption issues

### Moderate Performance
```
Mean F1: 0.65 ± 0.12
Mean absorption rate: 0.35 ± 0.15
```
- Features partially track concepts but have reliability issues
- Significant absorption occurring
- May need architecture changes or hyperparameter tuning

### Poor Performance
```
Mean F1: 0.45 ± 0.18
Mean absorption rate: 0.60 ± 0.20
```
- Features are not reliable classifiers
- Severe absorption problems
- Consider: different sparsity, more features, or architectural changes

## Programmatic Usage

```python
from absorption import (
    AbsorptionConfig,
    run_absorption_evaluation,
    load_matryoshka_transcoder
)

# Create config
config = AbsorptionConfig(
    model_name="gemma-2-2b",
    layers_to_eval=[3],
    num_samples=10000,
)

# Load transcoder
transcoder = load_matryoshka_transcoder(
    model_name="gemma-2-2b",
    layer=3,
)

# Run evaluation
results = run_absorption_evaluation(config, transcoder)

# Access results
print(f"F1: {results['summary']['mean_f1']:.3f}")
print(f"Absorption: {results['summary']['mean_absorption_rate']:.3f}")

# Per-letter details
for letter in ['a', 's', 'z']:
    letter_results = results['results_by_letter'][letter]
    print(f"{letter}: F1={letter_results['classification']['f1']:.3f}")
```

## Comparing with Paper Results

The original paper evaluated GemmaScope SAEs on Gemma-2-2B and found:
- Linear probes: F1 ≈ 0.95
- SAE features: F1 ≈ 0.60-0.75 (varies by layer and sparsity)
- Absorption rates: 0.3-0.6 (varies significantly by letter and layer)

Compare your Matryoshka transcoder results with these baselines!

## Troubleshooting

**Error: Checkpoint not found**
- Make sure you've trained a transcoder for the specified layer
- Run: `python src/scripts/train_all_layers_10k.py`

**Out of memory**
- Reduce `--num-samples` or `--batch-size`
- Use `--no-integrated-gradients` (uses less memory but slower)

**Evaluation too slow**
- Enable integrated gradients (default, faster)
- Use `--no-causality-check` to skip ablation (much faster but less rigorous)
- Reduce `--num-samples`

**Unexpected results**
- Check that your transcoder trained successfully
- Verify the checkpoint path is correct
- Try a different layer (early layers often easier to interpret)

## Next Steps

1. **Run evaluation** on your trained transcoders
2. **Compare results** across layers and prefix levels
3. **Identify issues** if absorption rates are high
4. **Iterate** on training hyperparameters if needed
5. **Document findings** for your research

## Support

For issues or questions:
- Check the full README.md for detailed documentation
- Review the example.py script for usage patterns
- Examine the paper: https://arxiv.org/pdf/2409.14507v3

