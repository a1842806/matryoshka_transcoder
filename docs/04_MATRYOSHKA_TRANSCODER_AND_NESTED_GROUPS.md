# Matryoshka Transcoder & Nested Groups

Technical overview of the Matryoshka Transcoder, its nested-group architecture, and practical usage.

## What is a Matryoshka Transcoder?

Learns a mapping from a source layer/site to a target layer/site using hierarchical (nested) groups of sparse features that enable progressive reconstruction and adjustable compute.

```
Source Activations ──▶ Encoder (W_enc) ──▶ TopK Sparse Features
                                      ├─▶ Group 1: Coarse → R₁
                                      ├─▶ Group 2: Medium → R₂ = R₁ + Δ₂
                                      ├─▶ Group 3: Fine   → R₃ = R₂ + Δ₃
                                      └─▶ Group 4: Finest → R₄ = R₃ + Δ₄ = Target Reconstruction
```

## Nested Groups: Core Mechanism

Groups are cumulative slices. With `group_sizes = [1152, 2304, 4608, 10368]` the effective totals are `[1152, 3456, 8064, 18432]`.

```python
# Initialization (conceptual)
total_dict_size = sum(cfg["group_sizes"])  # e.g., 18432
group_indices = [0] + list(torch.cumsum(torch.tensor(cfg["group_sizes"]), dim=0))
# [0, 1152, 3456, 8064, 18432]

# Reconstruction uses [0:end_idx] for each group i
for i in range(active_groups):
    start_idx = 0
    end_idx = group_indices[i + 1]
    # Use features [0:end_idx]
```

### Nested vs Sequential

- Nested (correct): Group i uses `[0:end_i]` and strictly contains group i-1.
- Sequential (incorrect for Matryoshka): disjoint ranges per group.

## Why Nested Groups?

- Adaptive compute: choose smaller groups for speed, larger for quality.
- Hierarchical features: coarse→fine organization across groups.
- One model, multiple capacities: avoid training separate models.

## Quick Start

```bash
# Train a single transcoder (example)
python -m src.scripts.train_matryoshka_transcoder --model-name gpt2-small --source-layer 8 --target-layer 8
```

Common within-layer mappings:

```python
# MLP transformation
source_site="resid_mid"; target_site="mlp_out"

# Attention transformation
source_site="resid_pre"; target_site="attn_out"

# Full layer transformation
source_site="resid_pre"; target_site="resid_post"
```

## Configuration Highlights

```python
cfg = {
    "model_name": "gpt2-small",
    "source_layer": 8,
    "target_layer": 8,
    "source_site": "resid_mid",
    "target_site": "mlp_out",
    "dict_size": 12288,                 # ≈ 16× activation size
    "group_sizes": [768, 1536, 3072, 6912],  # 1×, 2×, 4×, 9×
    "top_k": 64,
    "batch_size": 1024,
    "lr": 3e-4,
}
```

Guidelines:
- Use exponentially growing `group_sizes`.
- Start with `top_k ≈ dict_size / 200`.
- Keep shared logic in `src/training/` and helpers in `src/utils/`.

## Using Trained Transcoders

```python
from sae import MatryoshkaTranscoder

# Encode source to sparse features
sparse = transcoder.encode(source_acts)

# Decode to target reconstruction
recon = transcoder.decode(sparse)

# Full forward pass (training loop returns a dict of metrics)
out = transcoder(source_acts, target_acts)
recon, sparsity = out["sae_out"], out["l0_norm"]
```

## Verification & Troubleshooting

- Verify nesting with `src/utils/verify_nested_groups.py`.
- If many dead features: increase `top_k` or `aux_penalty`.
- If high absorption (cosine): increase `dict_size` or `top_k`.
- If unstable training: reduce `lr` or increase `batch_size`.

For Gemma-specific hook correctness (true MLP input), see `docs/02_TRAINING_AND_CONFIGURATION.md`.


