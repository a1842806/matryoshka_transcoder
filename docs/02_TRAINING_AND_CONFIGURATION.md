# Training & Configuration

This guide consolidates the practical pieces needed to train Matryoshka SAEs and transcoders across GPT-2 and Gemma-2 models.

## Environment & Prerequisites

- Python 3.10+ with CUDA-enabled PyTorch
- Recommended packages:

```bash
pip install torch transformer_lens datasets wandb rich
```

- (Optional) SAEBench extras for evaluation parity:

```bash
pip install saebench[all]
```

Set `WANDB_API_KEY` in your shell if you want experiment tracking.

## Configuration Essentials

Start from the helper in `src/utils/config.py`:

```python
from utils.config import get_default_cfg, post_init_cfg

cfg = get_default_cfg(model_type="gemma-2-2b")
cfg.update({
    "layer": 8,
    "source_site": "resid_mid",
    "target_site": "mlp_out",
    "dict_size": 12288,
    "top_k": 64,
    "group_sizes": [768, 1536, 3072, 6912],
})

cfg = post_init_cfg(cfg)
```

| Setting | Typical Values | Notes |
|---------|----------------|-------|
| `dict_size` | 16× activation width | Total number of features |
| `group_sizes` | Geometric progression | Must sum to `dict_size`; see nested groups below |
| `top_k` | `dict_size / 200` (rounded) | Controls sparsity per token |
| `lr` | `3e-4` | AdamW default |
| `batch_size` | `1024–4096` | Scale based on GPU RAM |
| `save_every` | `500–2000` steps | Checkpoint cadence |

### Nested Groups (Refresher)

Matryoshka transcoders treat groups as cumulative slices. For `group_sizes = [1152, 2304, 4608, 10368]`, the effective feature counts are `[1152, 3456, 8064, 18432]`. Each successive group reuses features from index 0 up to its cumulative boundary, enabling truncation at inference time.

See `docs/04_MATRYOSHKA_TRANSCODER_AND_NESTED_GROUPS.md` for full details.

## Running Training

CLI entry points live in `src/scripts/` and ultimately call functions in `src/training/`.

```bash
# GPT-2 small transcoder (quick smoke test)
python -m src.scripts.train_gpt2_simple --layer 8 --steps 2000

# Full-featured Matryoshka transcoder
python -m src.scripts.train_matryoshka_transcoder \
    --model-name gemma-2-2b \
    --source-layer 8 --target-layer 8 \
    --config overrides/run_layer8.yaml

# Launch multiple transcoders with shared activation store
python -m src.scripts.train_matryoshka_transcoder --mode multi
```

Prefer configuration files or CLI overrides over modifying scripts directly.

## Model-Specific Notes (Gemma-2)

Use post-RMSNorm activations as the true MLP input. In Gemma-2, prefer `ln2.hook_normalized * ln2.w` (post-norm, scaled) → `mlp_out` rather than `resid_mid` → `mlp_out`.

```python
def create_gemma_mlp_transcoder_config(base_cfg, layer, use_ln2_normalized=True):
    if use_ln2_normalized:
        base_cfg["source_site"] = "ln2_normalized"
        base_cfg["apply_rmsnorm_scaling"] = True
        base_cfg["source_hook_point"] = f"blocks.{layer}.ln2.hook_normalized"
    else:
        base_cfg["source_site"] = "resid_mid"
        base_cfg["apply_rmsnorm_scaling"] = False
        base_cfg["source_hook_point"] = f"blocks.{layer}.hook_resid_mid"
    return base_cfg
```

This produces better reconstruction FVU and aligns features with the actual MLP input space.

## GPU Memory & Multi-GPU (Quick Reference)

| Model | Approx GPU RAM | CPU RAM | Notes |
|-------|-----------------|---------|-------|
| GPT-2 Small | 2 GB | 4 GB | Large batches possible |
| GPT-2 Medium | 4 GB | 8 GB | Consider grad accumulation |
| Gemma-2-2B | 8 GB | 16 GB | Fits on a single 24 GB card |
| Gemma-2-9B | 16 GB | 32 GB | Multi-GPU or CPU offload |
| Gemma-2-27B | 32 GB | 64 GB | Sharded or CPU evaluation |

Tips when memory is tight:
- Lower `batch_size` or `context_length`.
- Temporarily disable activation sampling.
- Evaluate on CPU when needed using compiled kernels.

For safety-first evaluation, use the memory-safe or CPU variants in `src/scripts/`.

## Logging & Outputs

Use the standardized results writer and keep outputs under `results/{model_type}/{layer}/{steps}/`.

```python
from utils.results import create_run_dir
run_dir = create_run_dir(model_type="gemma-2-2b", layer=8, steps=7000)
```

`config.json` should contain the full configuration plus an ISO-formatted `completed_at` timestamp when a run finishes.


