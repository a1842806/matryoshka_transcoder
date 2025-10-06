# Matryoshka Transcoder - Quick Start Guide

## Installation

```bash
pip install transformer_lens torch wandb datasets
```

## 5-Minute Quick Start

### 1. Train a Single Transcoder

```bash
python train_matryoshka_transcoder.py single
```

This trains a transcoder that learns how GPT-2's MLP transforms features (resid_mid → mlp_out).

### 2. Basic Python Usage

```python
import torch
from transformer_lens import HookedTransformer
from sae import MatryoshkaTranscoder
from transcoder_activation_store import TranscoderActivationsStore, create_transcoder_config
from training import train_transcoder
from config import get_default_cfg

# 1. Setup config
cfg = get_default_cfg()
cfg["model_name"] = "gpt2-small"
cfg["num_tokens"] = 1e7  # 10M tokens
cfg["dict_size"] = 12288
cfg["group_sizes"] = [768, 1536, 3072, 6912]  # Hierarchical groups
cfg["top_k"] = 64

# 2. Configure source/target mapping
cfg = create_transcoder_config(
    cfg,
    source_layer=8,
    target_layer=8,
    source_site="resid_mid",  # Source: after attention
    target_site="mlp_out"     # Target: after MLP
)

# 3. Load model and create data store
model = HookedTransformer.from_pretrained_no_processing("gpt2-small").to("cuda")
activation_store = TranscoderActivationsStore(model, cfg)

# 4. Create and train transcoder
transcoder = MatryoshkaTranscoder(cfg)
train_transcoder(transcoder, activation_store, model, cfg)
```

## Common Configurations

### Within-Layer Transcoders

**MLP Transformation** (learns what MLP does):
```python
source_site="resid_mid", target_site="mlp_out"
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

## Group Size Guidelines

Choose groups that grow exponentially:

```python
# Small model (GPT-2, 768-dim)
"group_sizes": [768, 1536, 3072, 6912]

# Medium model (Gemma-2-2B, 2304-dim)  
"group_sizes": [2304, 4608, 9216, 20736]

# Large expansion (32x total)
"group_sizes": [512, 1024, 2048, 4096, 8192, 16128]
```

**Rule of thumb**: Each group ≈ 1-2x the previous group

## Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `dict_size` | Total features | 16x activation size |
| `group_sizes` | Feature groups | [1x, 2x, 4x, 9x] |
| `top_k` | Active features | dict_size / 200 |
| `lr` | Learning rate | 3e-4 |
| `batch_size` | Batch size | 1024-4096 |
| `aux_penalty` | Dead feature weight | 1/32 |

## Training Modes

```bash
# Single transcoder
python train_matryoshka_transcoder.py single

# Multiple transcoders (parallel)
python train_matryoshka_transcoder.py multiple

# Cross-layer transcoder
python train_matryoshka_transcoder.py cross_layer
```

## Monitoring Training

Weights & Biases dashboard shows:
- **L2 Loss**: Reconstruction quality (lower is better)
- **L0 Norm**: Active features per batch (should ≈ top_k)
- **Dead Features**: Inactive features (should be low)
- **Min/Max L2**: Best/worst group performance

## Using Trained Transcoders

```python
# Load from checkpoint
from utils import load_sae_from_wandb
from sae import MatryoshkaTranscoder

transcoder, cfg = load_sae_from_wandb(
    "your-project/artifact:version",
    MatryoshkaTranscoder
)

# Encode source activations
source_acts = ...  # Get from model
sparse_features = transcoder.encode(source_acts)

# Decode to target
target_reconstruction = transcoder.decode(sparse_features)

# Or full forward pass
with torch.no_grad():
    output = transcoder(source_acts, target_acts)
    reconstruction = output["sae_out"]
    sparsity = output["l0_norm"]
```

## Interpreting Results

**Good transcoder:**
- L2 loss < 0.1 (relative to activation variance)
- L0 norm ≈ top_k (efficient sparsity)
- Few dead features (< 10% of dict_size)
- Min/Max L2 gap is moderate (groups contribute differently)

**Signs of issues:**
- Very high L2: Increase dict_size or top_k
- L0 >> top_k: TopK not working, check batch size
- Many dead features: Increase aux_penalty or top_k

## File Structure

```
matryoshka_sae/
├── sae.py                           # MatryoshkaTranscoder class
├── transcoder_activation_store.py   # Data pipeline
├── training.py                      # Training loops
├── train_matryoshka_transcoder.py   # Example scripts
├── config.py                        # Configuration
├── MATRYOSHKA_TRANSCODER.md         # Full documentation
└── checkpoints/transcoder/          # Saved models
```

## Next Steps

1. **Read full docs**: See `MATRYOSHKA_TRANSCODER.md`
2. **Experiment**: Try different layer mappings
3. **Analyze features**: Use interpretability tools
4. **Scale up**: Train on larger models (Gemma, Llama)

## Troubleshooting

**Import errors?**
```bash
pip install transformer_lens wandb datasets torch
```

**CUDA out of memory?**
- Reduce `batch_size`
- Reduce `model_batch_size`
- Use `model_dtype=torch.bfloat16`

**Training too slow?**
- Increase `batch_size` (more efficient)
- Reduce `num_batches_in_buffer`
- Use smaller `dict_size`

**Poor results?**
- Train longer (`num_tokens`)
- Increase `dict_size`
- Adjust `group_sizes` ratios

## Support

For issues or questions, see the full documentation in `MATRYOSHKA_TRANSCODER.md`.

