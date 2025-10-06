# Matryoshka Transcoder Implementation

## Overview

This document describes the implementation of **Matryoshka Transcoders** - a novel architecture that combines the hierarchical feature learning of Matryoshka SAEs with cross-layer feature transformation (transcoders).

## What is a Matryoshka Transcoder?

A **Matryoshka Transcoder** learns to map activations from a source layer/site to a target layer/site using hierarchical groups of features at multiple abstraction levels.

### Key Concepts

1. **Traditional SAE**: `Layer L → Sparse Features → Reconstruct Layer L`
2. **Traditional Transcoder**: `Layer L (source) → Sparse Features → Reconstruct Layer L+1 (target)`
3. **Matryoshka Transcoder**: `Layer L (source) → Hierarchical Sparse Features → Reconstruct Layer L+1 (target)`

### Architecture

```
Source Activations (e.g., resid_mid)
    ↓
Encoder (W_enc)
    ↓
Sparse Features (TopK selection)
    ↓
├─ Group 1: Coarse features (e.g., 768 features)
│   └─ Partial Reconstruction R₁
├─ Group 2: Medium features (e.g., 1536 features)  
│   └─ Partial Reconstruction R₂ = R₁ + Group₂
├─ Group 3: Fine features (e.g., 3072 features)
│   └─ Partial Reconstruction R₃ = R₂ + Group₃
└─ Group 4: Finest features (e.g., 6912 features)
    └─ Final Reconstruction R₄ = R₃ + Group₄
        ↓
Target Activations (e.g., mlp_out)
```

## Implementation Files

### 1. `sae.py` - MatryoshkaTranscoder Class

The main implementation is in `sae.py`:

```python
from sae import MatryoshkaTranscoder

cfg = {
    "source_act_size": 768,
    "target_act_size": 768,
    "dict_size": 12288,
    "group_sizes": [768, 1536, 3072, 6912],  # Must sum to dict_size
    "top_k": 64,
    "device": "cuda",
    "dtype": torch.float32,
    # ... other config
}

transcoder = MatryoshkaTranscoder(cfg)
```

**Key methods:**
- `forward(x_source, x_target)`: Training forward pass
- `encode(x_source)`: Encode source to sparse features
- `decode(features)`: Decode features to target reconstruction

### 2. `transcoder_activation_store.py` - Data Pipeline

Handles paired activation collection:

```python
from transcoder_activation_store import TranscoderActivationsStore

store = TranscoderActivationsStore(model, cfg)
source_batch, target_batch = store.next_batch()
```

**Features:**
- Streams dataset (OpenWebText, C4, etc.)
- Collects activations from two hook points
- Buffers and shuffles for efficient training

### 3. `training.py` - Training Functions

Two training modes:

```python
from training import train_transcoder, train_transcoder_group_seperate_wandb

# Single transcoder
train_transcoder(transcoder, activation_store, model, cfg)

# Multiple transcoders with separate W&B runs
train_transcoder_group_seperate_wandb(transcoders, activation_stores, model, cfgs)
```

### 4. `train_matryoshka_transcoder.py` - Example Script

Ready-to-use training scripts:

```bash
# Train single transcoder (resid_mid -> mlp_out)
python train_matryoshka_transcoder.py single

# Train multiple transcoders in parallel
python train_matryoshka_transcoder.py multiple

# Train cross-layer transcoder (layer 8 -> layer 9)
python train_matryoshka_transcoder.py cross_layer
```

## Configuration

### Essential Parameters

```python
cfg = {
    # Model
    "model_name": "gpt2-small",
    "model_dtype": torch.bfloat16,
    
    # Source/Target Layers
    "source_layer": 8,
    "target_layer": 8,  # Can be different for cross-layer
    "source_site": "resid_mid",  # resid_pre, resid_mid, resid_post, attn_out, mlp_out
    "target_site": "mlp_out",
    
    # Architecture
    "dict_size": 12288,
    "group_sizes": [768, 1536, 3072, 6912],  # Matryoshka groups
    "top_k": 64,  # Total active features across all groups
    
    # Training
    "batch_size": 1024,
    "lr": 3e-4,
    "num_tokens": 1e8,
    "aux_penalty": 1/32,
    
    # Dead feature handling
    "n_batches_to_dead": 20,
    "top_k_aux": 512,
}
```

### Group Size Design

Choose group sizes that grow exponentially:

```python
# Small model (GPT-2 Small, act_size=768)
"group_sizes": [768, 1536, 3072, 6912]  # 1x, 2x, 4x, 9x

# Medium model (Gemma-2-2B, act_size=2304)
"group_sizes": [2304, 4608, 9216, 20736]  # 1x, 2x, 4x, 9x

# Large expansion
"group_sizes": [512, 1024, 2048, 4096, 8192, 16128]  # For act_size=768
```

## Common Use Cases

### 1. MLP Transformation

Learn how the MLP transforms features:

```python
source_site = "resid_mid"  # After attention, before MLP
target_site = "mlp_out"    # After MLP
```

### 2. Attention Transformation

Learn how attention transforms features:

```python
source_site = "resid_pre"  # Before attention
target_site = "attn_out"   # After attention
```

### 3. Full Layer Transformation

Learn the complete layer transformation:

```python
source_site = "resid_pre"   # Layer input
target_site = "resid_post"  # Layer output
```

### 4. Cross-Layer Mapping

Learn how features evolve between layers:

```python
source_layer = 8
target_layer = 9
source_site = "resid_post"  # Output of layer 8
target_site = "resid_pre"   # Input of layer 9
```

## Training Example

```python
import torch
from transformer_lens import HookedTransformer
from sae import MatryoshkaTranscoder
from transcoder_activation_store import TranscoderActivationsStore, create_transcoder_config
from training import train_transcoder
from config import get_default_cfg

# Setup configuration
cfg = get_default_cfg()
cfg["model_name"] = "gpt2-small"
cfg["num_tokens"] = 1e7
cfg["dict_size"] = 12288
cfg["group_sizes"] = [768, 1536, 3072, 6912]
cfg["top_k"] = 64

# Configure source/target mapping
cfg = create_transcoder_config(
    cfg,
    source_layer=8,
    target_layer=8,
    source_site="resid_mid",
    target_site="mlp_out"
)

# Load model
model = HookedTransformer.from_pretrained_no_processing("gpt2-small").to("cuda")

# Create data pipeline
activation_store = TranscoderActivationsStore(model, cfg)

# Create transcoder
transcoder = MatryoshkaTranscoder(cfg)

# Train
train_transcoder(transcoder, activation_store, model, cfg)
```

## Evaluation Metrics

The training logs several metrics:

1. **L2 Loss**: Mean reconstruction error across all groups
2. **Min/Max L2 Loss**: Best/worst group reconstruction
3. **L0 Norm**: Average number of active features
4. **L1 Loss**: L1 sparsity penalty (usually 0 for TopK)
5. **Auxiliary Loss**: Dead feature reactivation loss
6. **Dead Features**: Number of inactive features
7. **Threshold**: Learned activation threshold

## Advantages

1. **Hierarchical Feature Learning**: Coarse-to-fine feature organization
2. **Reduced Feature Interference**: Groups specialize in different abstraction levels
3. **Flexible Granularity**: Can use fewer groups for faster inference
4. **Better Interpretability**: Separated high-level vs low-level transformations
5. **Cross-Layer Understanding**: Learn how features transform through the network

## Comparison to Standard Transcoder

| Aspect | Standard Transcoder | Matryoshka Transcoder |
|--------|-------------------|---------------------|
| **Feature Organization** | Flat dictionary | Hierarchical groups |
| **Reconstruction** | Single step | Progressive refinement |
| **Interpretability** | All features equal | Clear abstraction levels |
| **Flexibility** | Fixed capacity | Adjustable by group |
| **Training** | Single loss | Multi-level loss |

## Checkpoints

Trained models are saved to:
```
checkpoints/transcoder/{model_name}/{config_name}_{step}/
├── sae.pt          # Model weights
└── config.json     # Configuration
```

Load a trained transcoder:
```python
from utils import load_sae_from_wandb
from sae import MatryoshkaTranscoder

transcoder, cfg = load_sae_from_wandb(
    "your-project/artifact-name:version",
    MatryoshkaTranscoder
)
```

## Related Work

This implementation builds on:

1. **Matryoshka SAEs** ([Paper](https://arxiv.org/pdf/2503.17547)): Hierarchical feature learning
2. **BatchTopK SAEs**: Efficient global sparsity selection
3. **Transcoders/CLT**: Cross-layer feature transformation
4. **SAELens**: Foundation for training infrastructure

## Tips for Best Results

1. **Group Size Ratio**: Use exponentially growing groups (1x, 2x, 4x, 9x)
2. **TopK Selection**: Start with K ≈ dict_size/200 for balance
3. **Learning Rate**: 3e-4 works well for most models
4. **Auxiliary Loss**: Weight of 1/32 helps prevent dead features
5. **Batch Size**: Larger batches (1024-4096) for stable TopK
6. **Data**: Use diverse text (OpenWebText, C4, FineWeb)

## Troubleshooting

**Too many dead features?**
- Increase `top_k`
- Increase `aux_penalty`
- Decrease `n_batches_to_dead`

**Poor reconstruction?**
- Increase `dict_size` or group sizes
- Increase `top_k`
- Train longer

**Training unstable?**
- Reduce learning rate
- Increase batch size
- Check activation magnitudes

## Future Extensions

Potential improvements:

1. **Adaptive group selection**: Learn which groups to use
2. **Non-uniform TopK**: Different K per group
3. **Multi-target transcoders**: Map to multiple targets
4. **Causal transcoders**: Enforce temporal dependencies
5. **Cross-model transcoders**: Map between different architectures

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

