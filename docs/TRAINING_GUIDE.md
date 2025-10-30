# Training Guide

This document consolidates the prior quickstart, implementation notes, and hardware guides into a focused reference for running Matryoshka SAE and transcoder training.

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
- Configure `WANDB_API_KEY` in your shell if you want experiment tracking.

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
| `dict_size` | 16× activation width | Sets total number of features |
| `group_sizes` | Geometric progression | Must sum to `dict_size`; see nested groups below |
| `top_k` | `dict_size / 200` (rounded) | Controls sparsity per token |
| `lr` | 3e-4 | AdamW default |
| `batch_size` | 1024 – 4096 | Scale based on GPU RAM |
| `save_every` | 500 – 2000 steps | Determine checkpoint cadence |

### Nested Groups Refresher

- Matryoshka transcoders treat groups as cumulative slices. If `group_sizes = [1152, 2304, 4608, 10368]`, the effective feature counts become `[1152, 3456, 8064, 18432]`.
- During the forward pass each group reuses the features from index 0 up to its cumulative boundary, enabling truncation at runtime.
- Keep each successive group ~1.5–2× larger than the previous one for smooth performance scaling.

## Running Training

CLI entry points live in `src/scripts/`. Recommended invocations:

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

- Scripts ultimately call functions in `src/training/training.py`.
- Use configuration files or CLI overrides rather than editing scripts directly.
- Respect the results directory contract (see below); helper functions are provided in `src/utils/logging.py`.

## Activation Sample Collection

Enable sample tracking to inspect high-activation sequences:

```python
cfg["save_activation_samples"] = True
cfg["sample_collection_freq"] = 1000
cfg["max_samples_per_feature"] = 100
cfg["sample_context_size"] = 20
cfg["sample_activation_threshold"] = 0.1
```

Samples are stored alongside the run directory under `activation_samples/` and can be parsed with `ActivationSampleAnalyzer`.

## Model-Specific Notes

- **Gemma hooks**: use the patched hook map in `src/adapters/gemma_hooks.py` to ensure correct activation naming (`resid_pre`, `resid_mid`, `mlp_out`).
- **Group presets**:
  - GPT-2 Small (768 dim): `[768, 1536, 3072, 6912]`
  - Gemma-2-2B (2304 dim): `[2304, 4608, 9216, 20736]`
- **FVU metrics**: Prefer `utils.metrics.compute_fvu` over raw L2 for scale-invariant reporting. Track `fvu_min`, `fvu_mean`, and `fvu_max` per group when logging.

### GPU Memory Reference

| Model | GPU RAM (approx) | CPU RAM | Notes |
|-------|------------------|---------|-------|
| GPT-2 Small | 2 GB | 4 GB | High batch sizes possible |
| GPT-2 Medium | 4 GB | 8 GB | Consider gradient accumulation |
| Gemma-2-2B | 8 GB | 16 GB | Fits on a single 24 GB card |
| Gemma-2-9B | 16 GB | 32 GB | Multi-GPU or CPU offload recommended |
| Gemma-2-27B | 32 GB | 64 GB | Requires sharded or CPU evaluation |

If memory is tight:
- Lower `batch_size` or `context_length` in the config.
- Disable activation sampling temporarily.
- Move evaluation to CPU (`device="cpu"`) using compiled kernels.

## Logging & Outputs

- All training helpers should call the standardized results writer:
  ```python
  from utils.results import create_run_dir
  run_dir = create_run_dir(model_type="gemma-2-2b", layer=8, steps=7000)
  ```
- Persist checkpoints and metrics under `results/{model_type}/{layer}/{step_count}/`.
- Ensure `config.json` contains the full configuration plus an ISO-formatted `completed_at` timestamp once the run finishes.

Keep scripts lightweight; place shared logic in `src/training/` and shared helpers in `src/utils/` so maintenance stays simple.

