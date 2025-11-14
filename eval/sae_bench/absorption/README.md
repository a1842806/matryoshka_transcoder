# Absorption Score Evaluation for Matryoshka Transcoders

This module implements the feature absorption evaluation methodology from:
**"A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders"**  
Paper: https://arxiv.org/pdf/2409.14507v3  
Explorer: https://feature-absorption.streamlit.app

## Overview

Feature absorption occurs when a Sparse Autoencoder (SAE) or transcoder latent appears to track a human-interpretable concept but fails to activate on seemingly arbitrary tokens. Instead, approximately token-aligned latents activate and contribute a portion of the probe direction, "absorbing" the feature.

This is problematic because it suggests SAE/transcoder features may be inherently unreliable classifiers, which is particularly important for safety applications where we need confidence that features fully track behaviors like bias or deception.

## Key Concepts

1. **First-Letter Task**: A ground-truth classification task where we know exactly which tokens should activate for each feature (e.g., tokens starting with 'S').

2. **Linear Probe**: A simple classifier trained on model activations to detect the concept. Used as ground truth.

3. **Feature Selection**: Finding transcoder features that should track the concept (via cosine similarity with probe or k-sparse probing).

4. **Absorption Detection**: Identifying when the main feature fails to fire and token-aligned features activate instead.

5. **Causal Verification**: Using ablation experiments to verify that absorbing features have causal impact on model behavior.

## Installation

The module is self-contained within the Matryoshka SAE repository. Required dependencies:
- PyTorch
- TransformerLens
- scikit-learn
- numpy
- tqdm

## Usage

### Basic Usage

Evaluate a single layer:
```bash
python run_evaluation.py --layer 3 --model gemma-2-2b
```

Evaluate multiple layers:
```bash
python run_evaluation.py --layers 0 1 2 3 --model gemma-2-2b
```

Evaluate all layers (0-17, before attention moves information):
```bash
python run_evaluation.py --all-layers --model gemma-2-2b
```

### Using Matryoshka Prefix Levels

Evaluate a specific prefix level (0-4):
```bash
python run_evaluation.py --layer 3 --prefix-level 2
```

### Advanced Options

```bash
python run_evaluation.py \
    --layer 3 \
    --model gemma-2-2b \
    --num-samples 20000 \
    --batch-size 64 \
    --cosine-threshold 0.4 \
    --ablation-threshold 0.02 \
    --device cuda:0 \
    --results-dir results/absorption
```

### Programmatic Usage

```python
from absorption.config import AbsorptionConfig
from absorption.evaluation import run_absorption_evaluation
from absorption.matryoshka_wrapper import load_matryoshka_transcoder

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
    prefix_level=None,  # Use full size
)

# Run evaluation
results = run_absorption_evaluation(config, transcoder)
print(f"Mean F1: {results['summary']['mean_f1']}")
print(f"Absorption rate: {results['summary']['mean_absorption_rate']}")
```

## Output

The evaluation produces:
- Per-letter classification metrics (precision, recall, F1)
- Absorption rate (fraction of tokens where main feature fails to fire)
- List of absorbing features with cosine similarities and ablation effects
- Examples of absorbed tokens

Results are saved to JSON in the specified results directory:
```
results/absorption/
  gemma-2-2b/
    layer3/
      absorption_results.json
    absorption_all_layers.json
```

## Metrics

### Classification Metrics
- **Precision**: Of tokens where feature fires, how many actually have the concept
- **Recall**: Of tokens with the concept, how many activate the feature
- **F1**: Harmonic mean of precision and recall

### Absorption Metrics
- **Absorption Rate**: Fraction of positive tokens where main feature doesn't fire
- **Absorbing Features**: Features that fire instead, with causal impact
- **Cosine Similarity**: Alignment between absorbing feature and probe direction
- **Ablation Effect**: Causal impact of absorbing feature on correct prediction

## Understanding Results

Good features:
- High F1 score (> 0.8)
- Low absorption rate (< 0.2)
- Few or no absorbing features

Problematic features (showing absorption):
- Lower recall (< 0.6) despite high precision
- High absorption rate (> 0.5)
- Many absorbing features with significant ablation effects

## Comparison with Linear Probes

The paper shows that linear probes significantly outperform SAE features on the first-letter task. Our implementation allows you to compare:

```python
from absorption.evaluation import compare_probe_vs_transcoder

comparison = compare_probe_vs_transcoder(config, transcoder_wrapper)
print(f"Probe F1: {comparison['probe_performance']['f1']}")
```

## Limitations

1. **Layer Range**: The evaluation only works reliably on layers 0-17 for Gemma-2-2B, before attention moves information from subject tokens to prediction positions.

2. **Causality Verification**: Ablation experiments are slower. Use `--no-integrated-gradients` for exact ablation or enable integrated gradients for faster approximation.

3. **Task Specificity**: This evaluation uses the first-letter task. Other tasks may show different patterns.

## Citation

If you use this evaluation in your research, please cite both the absorption paper and SAE Bench:

```bibtex
@article{chanin2024absorption,
  title={A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders},
  author={Chanin, David and Wilken-Smith, James and Dulka, Tom{\'a}{\v{s}} and Bhatnagar, Hardik and Bloom, Joseph},
  journal={arXiv preprint arXiv:2409.14507},
  year={2024}
}

@article{karvonen2025saebench,
  title={SAEBench: A Comprehensive Benchmark for Evaluating Sparse Autoencoders},
  author={Karvonen, Adam and others},
  journal={arXiv preprint arXiv:2503.09532},
  year={2025}
}
```

## Contact

For questions or issues, please open an issue on the repository.

