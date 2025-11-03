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

cfg = get_default_cfg()
cfg.update({
    "model_name": "gemma-2-2b",
    "layer": 8,
    "dict_size": 12288,
    "top_k": 64,
    "prefix_sizes": [768, 1536, 3072, 6912],
})

cfg = post_init_cfg(cfg)
```

| Setting | Typical Values | Notes |
|---------|----------------|-------|
| `dict_size` | 16× activation width | Total number of features |
| `prefix_sizes` | Geometric/exp spacing | Must end at `dict_size`; see nested prefixes below |
| `top_k` | `dict_size / 200` (rounded) | Controls sparsity per token |
| `lr` | `3e-4` | AdamW default |
| `batch_size` | `1024–4096` | Scale based on GPU RAM |
| `save_every` | `500–2000` steps | Checkpoint cadence |

### Nested Prefixes (Refresher)

Matryoshka transcoders treat prefixes as cumulative slices. For `prefix_sizes = [1152, 2304, 4608, 10368]`, the effective totals are `[1152, 3456, 8064, 18432]`. Each successive prefix reuses features from index 0 up to its cumulative boundary, enabling truncation at inference time. Both individual and cumulative formats are supported; cumulative is monotonic.

See `docs/04_MATRYOSHKA_TRANSCODER_AND_NESTED_GROUPS.md` for full details.

## Running Training

CLI entry points live in `src/scripts/` and ultimately call functions in `src/training/`.

```bash
# Layer 8 example (Gemma-2-2B)
python -m src.scripts.train_layer8_18k_topk96

# Layer 17 example with warmup + decay
python -m src.scripts.train_gemma_layer17_with_warmup_decay_samples
```

Prefer configuration files or CLI overrides over modifying scripts directly.

## Model-Specific Notes (Gemma-2)

Prefer `mlp_in → mlp_out` for the true MLP transformation. Use the config helper and let the activation store reconstruct `mlp_in` as `ln2.hook_normalized * RMSNorm_scale` when the direct hook isn’t available.

```python
from models.transcoder_activation_store import create_transcoder_config

cfg = create_transcoder_config(
    cfg,
    source_layer=8,
    target_layer=8,
    source_site="mlp_in",
    target_site="mlp_out",
)
```

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

Use the standardized results manager and keep outputs under `results/{model_name}/layer{layer}/{steps}/`.

```python
from utils.clean_results import CleanResultsManager

manager = CleanResultsManager()
run_dir = manager.create_experiment_dir(
    model_name=cfg["model_name"], layer=cfg["layer"], steps=expected_steps
)
```

`config.json` will include an ISO-formatted `completed_at` timestamp on final save; interim metrics live under `metrics/step_*.json` and final summary in `metrics.json`.


