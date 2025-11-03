# Matryoshka Transcoder

Matryoshka Transcoders are cross-layer feature transformers that learn hierarchical feature representations by mapping activations between different layers of transformer models. They use a multi-level dictionary structure where earlier groups learn high-level concepts and later groups learn low-level details, enabling interpretable cross-layer feature transformations.

## Key Features

### FVU Metrics
All transcoders track **FVU (Fraction of Variance Unexplained)** - a scale-independent measure of cross-layer reconstruction quality:
- FVU = 0: Perfect cross-layer mapping
- FVU < 0.1: Good cross-layer reconstruction
- FVU shown in progress bar and W&B logs

### Gemma-2 Support
Full support for Gemma-2 models with automatic configuration:
```python
cfg["model_name"] = "gemma-2-2b"  # Auto-detects act_size=2304
```

### Matryoshka Transcoders
Train cross-layer feature transformers with hierarchical feature groups:
```bash
# Quick start - GPT-2 Small transcoder
python src/scripts/train_gpt2_simple.py

# Full training with W&B
python src/scripts/train_gpt2_transcoder.py

# Gemma-2 examples
python src/scripts/train_gemma_example.py
```

## Installation

```bash
pip install transformer_lens wandb datasets torch
```

## Quick Start

### Training Matryoshka Transcoders

```bash
# Gemma-2-2B Layer 8 (≈16k steps)
python -m src.scripts.train_layer8_18k_topk96

# Gemma-2-2B Layer 17 with warmup + cosine decay + samples
python -m src.scripts.train_gemma_layer17_with_warmup_decay_samples
```

### Training with Gemma-2-2B

```bash
# Compare model configurations
python train_gemma_example.py compare

# Train Gemma-2-2B SAE
python train_gemma_example.py sae

# Train Gemma-2-2B Transcoder
python train_gemma_example.py transcoder
```

## Usage Examples

### Transcoder Training

```python
from src.models.sae import MatryoshkaTranscoder
from src.models.transcoder_activation_store import TranscoderActivationsStore, create_transcoder_config
from src.training.training import train_transcoder
from src.utils.config import get_default_cfg

cfg = get_default_cfg()
cfg["model_name"] = "gemma-2-2b"

# Configure source/target mapping (true MLP transformation)
cfg = create_transcoder_config(
    cfg,
    source_layer=8,
    target_layer=8,
    source_site="mlp_in",
    target_site="mlp_out",
)

activation_store = TranscoderActivationsStore(model, cfg)
transcoder = MatryoshkaTranscoder(cfg)
train_transcoder(transcoder, activation_store, model, cfg)
```

### Using Trained Models

```python
# Load from results directory (final.pt + config.json)
import json, torch
from src.models.sae import MatryoshkaTranscoder

run_dir = "results/gemma-2-2b/layer8/16384"  # example
with open(f"{run_dir}/config.json", "r") as f:
    cfg = json.load(f)

transcoder = MatryoshkaTranscoder(cfg)
state = torch.load(f"{run_dir}/checkpoints/final.pt", map_location=cfg.get("device", "cpu"))
transcoder.load_state_dict(state)
transcoder.eval()

# Encode and decode
sparse_features = transcoder.encode(source_acts)
target_reconstruction = transcoder.decode(sparse_features)
```

## Model Support

| Model | Activation Size | Layers | Typical Dict Size (16x) |
|-------|----------------|---------|------------------------|
| gpt2 | 768 | 12 | 12,288 |
| gpt2-medium | 1024 | 24 | 16,384 |
| gpt2-large | 1280 | 36 | 20,480 |
| gpt2-xl | 1600 | 48 | 25,600 |
| **gemma-2-2b** | **2304** | **26** | **36,864** |
| gemma-2-9b | 3584 | 42 | 57,344 |
| gemma-2-27b | 4608 | 46 | 73,728 |

## Configuration

### Essential Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `dict_size` | Total features | 16x activation size |
| `prefix_sizes` | Cumulative prefixes | [1x, 2x, 4x, 9x] |
| `top_k` | Active features | dict_size / 200 |
| `lr` | Learning rate | 3e-4 |
| `batch_size` | Batch size | 1024-4096 |
| `aux_penalty` | Dead feature weight | 1/32 |

### Prefix Size Guidelines

```python
# Small model (GPT-2, 768-dim)
"prefix_sizes": [768, 1536, 3072, 6912]

# Medium model (Gemma-2-2B, 2304-dim)
"prefix_sizes": [2304, 4608, 9216, 20736]

# Large expansion (example)
"prefix_sizes": [512, 1024, 2048, 4096, 8192, 16128]
```

## Common Use Cases

### Within-Layer Transcoders

**MLP Transformation** (learns the true MLP mapping):
```python
source_site="mlp_in", target_site="mlp_out"
```

**Attention Transformation** (learns what attention does):
```python
source_site="resid_pre", target_site="attn_out"
```

**Full Layer** (learns complete layer transformation):
```python
source_site="resid_pre", target_site="resid_post"
```

### Cross-Layer Transcoders

**Between layers** (learns how features evolve):
```python
source_layer=8, target_layer=9
source_site="resid_post", target_site="resid_pre"
```

## Monitoring Training

### Real-time Metrics

Training displays metrics in real-time:
```
Training: 45%|████▌     | 450/1000 [02:30<02:45, Loss: 0.023, L0: 64, L2: 0.018, FVU: 0.18, Dead: 12]
```

**Metrics:**
- **Loss**: Total training loss (lower is better)
- **L0**: Number of active features (~64 expected)
- **L2**: Reconstruction error (lower is better)
- **FVU**: Fraction of Variance Unexplained (0=perfect, <0.1=good)
- **Dead**: Number of inactive features (lower is better)

### W&B Dashboard

Your training will be logged to Weights & Biases:
- **Project**: `gpt2-matryoshka-transcoder`
- **Dashboard**: https://wandb.ai/your-entity/gpt2-matryoshka-transcoder

## Evaluation

### Reconstruction and Comparison

```bash
# Compare your transcoder vs Gemma Scope at layer 17
python -m src.scripts.compare_transcoders \
  --matryoshka_checkpoint results/gemma-2-2b/layer17/15000/checkpoints \
  --gemma_scope_dir models/gemma-scope-2b-pt-transcoders \
  --dataset HuggingFaceFW/fineweb-edu --tokens 2000000
```

**Key internal metrics (typical):**
- **Absorption Score**: Feature redundancy (lower is better, < 0.05 good)
- **Splitting Score**: Co-activation frequency (lower is better, < 0.05 good)
- **Monosemanticity**: Feature specificity (higher is better, > 0.7 good)
- **Dead Features**: Inactive features (lower is better, < 5% good)

## Troubleshooting

### CUDA out of memory:
```python
cfg["model_batch_size"] = 8  # Reduce batch size
cfg["batch_size"] = 1024
```

### Training too slow:
```python
cfg["num_tokens"] = int(5e6)  # Reduce tokens for quick test
```

### Too many dead features:
```python
cfg["aux_penalty"] = 1/16  # Increase auxiliary loss weight
cfg["top_k"] = 96  # Increase active features
```

### Poor reconstruction (high FVU):
```python
cfg["dict_size"] = 24576  # Increase dictionary size
cfg["top_k"] = 96  # Increase active features
# Or train longer:
cfg["num_tokens"] = int(5e7)
```

## Documentation

### Core Guides
- **docs/01_PROJECT_OVERVIEW_AND_STRUCTURE.md**: Project overview and repository layout
- **docs/02_TRAINING_AND_CONFIGURATION.md**: Training, configuration, Gemma notes, results layout
- **docs/03_EVALUATION_AND_METRICS.md**: Metrics, SAEBench methodology overview
- **docs/04_MATRYOSHKA_TRANSCODER_AND_NESTED_GROUPS.md**: Architecture and nested prefixes
- **docs/05_INTERPRETABILITY_TOOLS_ACTIVATION_SAMPLES.md**: Activation sample collection and analysis

## File Structure

```
matryoshka_sae/
├── src/
│   ├── models/
│   │   ├── sae.py
│   │   └── transcoder_activation_store.py
│   ├── training/
│   │   └── training.py
│   ├── scripts/
│   │   ├── train_layer8_18k_topk96.py
│   │   ├── train_gemma_layer17_with_warmup_decay_samples.py
│   │   └── compare_transcoders.py
│   └── utils/
│       ├── config.py
│       ├── activation_samples.py
│       ├── analyze_activation_samples.py
│       ├── clean_results.py
│       └── logs.py
├── results/             # results/{model_name}/layer{L}/{steps}/
├── docs/                # consolidated docs
└── README.md
```

## Acknowledgments

The training code is heavily inspired and basically a stripped-down version of [SAELens](https://github.com/jbloomAus/SAELens). Thanks to the SAELens team for their foundational work in this area!

## Citation

If you use this implementation, please cite:

```bibtex
@article{matryoshka_sae_2025,
  title={Learning Multi-Level Features with Matryoshka Sparse Autoencoders},
  author={},
  journal={arXiv preprint arXiv:2503.17547},
  year={2025}
}
```
