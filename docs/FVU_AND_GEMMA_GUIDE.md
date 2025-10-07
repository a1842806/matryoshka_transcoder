# FVU Metrics and Gemma-2 Support Guide

## Overview

This guide covers two major enhancements to the Matryoshka SAE/Transcoder implementation:
1. **FVU (Fraction of Variance Unexplained) Metrics** - Better reconstruction quality measurement
2. **Gemma-2 Model Support** - Proper handling of Gemma-2-2B and other Gemma models

---

## FVU Metrics

### What is FVU?

**FVU (Fraction of Variance Unexplained)** measures the proportion of variance in the data that the model fails to explain:

```
FVU = Var(residuals) / Var(original)
    = MSE(original, reconstruct) / Var(original)
```

**Interpretation:**
- **FVU = 0**: Perfect reconstruction
- **FVU = 1**: Reconstruction is no better than using the mean
- **FVU < 0.1**: Good reconstruction
- **FVU > 0.5**: Poor reconstruction

### Why FVU is Better than L2 Loss

| Metric | Description | Problem |
|--------|-------------|---------|
| **L2 Loss** | Mean squared error | Scale-dependent, hard to interpret across datasets |
| **FVU** | Normalized reconstruction error | Scale-independent, comparable across datasets |

**Example:**
- L2 = 0.5 could be excellent or terrible depending on activation scale
- FVU = 0.2 always means 20% of variance unexplained (clear interpretation)

### FVU in Training Outputs

All SAE and Transcoder classes now output FVU metrics:

#### Standard SAEs (BatchTopK, TopK, Vanilla, JumpReLU):
```python
output = {
    "loss": ...,
    "l2_loss": ...,
    "fvu": ...  # ← NEW: Overall reconstruction quality
}
```

#### Matryoshka SAEs and Transcoders:
```python
output = {
    "loss": ...,
    "l2_loss": ...,
    "fvu": ...,          # ← Final reconstruction FVU
    "fvu_min": ...,      # ← Best group FVU
    "fvu_max": ...,      # ← Worst group FVU
    "fvu_mean": ...      # ← Average group FVU
}
```

### FVU in Progress Bar

Training now displays FVU in real-time:

```
Training: 45%|████▌     | 450/1000 [02:30<02:45, Loss: 0.0234, L0: 64.2, L2: 0.0180, FVU: 0.18]
```

### FVU in Weights & Biases

FVU is automatically logged to W&B for every training step:

```python
wandb.log({
    "fvu": fvu,
    "fvu_min": fvu_min,  # For Matryoshka models
    "fvu_max": fvu_max,
    "fvu_mean": fvu_mean,
})
```

### Using FVU in Code

```python
from config import compute_fvu

# During training
x_original = ...  # Original activations
x_reconstruct = ...  # Reconstructed activations

fvu = compute_fvu(x_original, x_reconstruct)
print(f"FVU: {fvu:.4f}")  # e.g., "FVU: 0.1823"
```

---

## Gemma-2 Model Support

### Supported Models

The implementation now automatically handles:

- **GPT-2 family**: `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`
- **Gemma-2 family**: `gemma-2-2b`, `gemma-2-9b`, `gemma-2-27b`

### Automatic Configuration

When you specify a model name, the system automatically detects:

1. **Activation size** (`act_size`)
2. **Number of layers** (`n_layers`)
3. **Proper hook names** (via TransformerLens)

```python
from config import get_model_config, post_init_cfg

cfg = get_default_cfg()
cfg["model_name"] = "gemma-2-2b"  # Just specify the model

cfg = post_init_cfg(cfg)  # Auto-detects act_size=2304, proper hooks
```

### Model Specifications

| Model | Activation Size | Layers | Typical Dict Size (16x) |
|-------|----------------|---------|------------------------|
| gpt2 | 768 | 12 | 12,288 |
| gpt2-medium | 1024 | 24 | 16,384 |
| gpt2-large | 1280 | 36 | 20,480 |
| gpt2-xl | 1600 | 48 | 25,600 |
| **gemma-2-2b** | **2304** | **26** | **36,864** |
| gemma-2-9b | 3584 | 42 | 57,344 |
| gemma-2-27b | 4608 | 46 | 73,728 |

### Hook Name Handling

TransformerLens automatically handles hook naming differences:

```python
from config import get_hook_name

# Works for both GPT-2 and Gemma-2
hook = get_hook_name("resid_pre", layer=8, model_name="gemma-2-2b")
# Returns: "blocks.8.hook_resid_pre"
```

**Available hook sites:**
- `resid_pre`: Residual stream before the layer
- `resid_mid`: After attention, before MLP
- `mlp_out`: Output of MLP (before adding to residual)
- `attn_out`: Output of attention (before adding to residual)
- `resid_post`: Residual stream after the layer

### Training with Gemma-2-2B

#### Quick Example:

```python
from transformer_lens import HookedTransformer
from sae import GlobalBatchTopKMatryoshkaSAE
from activation_store import ActivationsStore
from training import train_sae
from config import get_default_cfg, post_init_cfg

# Configure for Gemma-2-2B
cfg = get_default_cfg()
cfg["model_name"] = "gemma-2-2b"
cfg["layer"] = 8
cfg["site"] = "resid_pre"
cfg["dict_size"] = 36864  # 16x expansion
cfg["group_sizes"] = [2304, 4608, 9216, 20736]  # 1x, 2x, 4x, 9x
cfg["top_k"] = 96
cfg["model_batch_size"] = 4  # Smaller for larger model
cfg["model_dtype"] = torch.bfloat16

# Auto-detect activation size and set hooks
cfg = post_init_cfg(cfg)

# Load model
model = HookedTransformer.from_pretrained_no_processing(
    "gemma-2-2b"
).to(cfg["model_dtype"]).to(cfg["device"])

# Train
activation_store = ActivationsStore(model, cfg)
sae = GlobalBatchTopKMatryoshkaSAE(cfg)
train_sae(sae, activation_store, model, cfg)
```

#### Using the Example Script:

```bash
# Compare configurations
python train_gemma_example.py compare

# Train Gemma-2-2B SAE
python train_gemma_example.py sae

# Train Gemma-2-2B Transcoder
python train_gemma_example.py transcoder
```

### Gemma-2 Training Tips

1. **Batch Size**: Gemma-2 models are larger, use smaller batch sizes:
   ```python
   cfg["model_batch_size"] = 4  # vs 32 for GPT-2
   ```

2. **Dictionary Size**: Scale with activation size:
   ```python
   # GPT-2: 768 * 16 = 12,288
   # Gemma-2-2B: 2304 * 16 = 36,864
   ```

3. **Group Sizes**: Maintain proportions:
   ```python
   # GPT-2: [768, 1536, 3072, 6912]
   # Gemma-2-2B: [2304, 4608, 9216, 20736]
   ```

4. **Top-K**: Scale with dictionary size:
   ```python
   # Rule of thumb: dict_size / 200
   # GPT-2: 12288 / 200 ≈ 64
   # Gemma-2-2B: 36864 / 200 ≈ 192
   ```

5. **Data Type**: Use `bfloat16` for efficiency:
   ```python
   cfg["model_dtype"] = torch.bfloat16
   cfg["dtype"] = torch.bfloat16
   ```

6. **Dataset**: Use high-quality datasets:
   ```python
   cfg["dataset_path"] = "HuggingFaceFW/fineweb-edu"  # Better than OpenWebText
   ```

---

## Migrating Existing Code

### If you have existing training scripts:

#### 1. No changes needed for basic usage!
```python
# Old code works as-is
cfg = get_default_cfg()
cfg["model_name"] = "gpt2-small"
# ... rest of code unchanged
```

#### 2. To use Gemma-2-2B, just change model name:
```python
cfg = get_default_cfg()
cfg["model_name"] = "gemma-2-2b"  # ← Only change
cfg["dict_size"] = 36864  # Scale up dictionary
cfg["group_sizes"] = [2304, 4608, 9216, 20736]  # Scale up groups
# Everything else handled automatically!
```

#### 3. FVU metrics appear automatically:
```python
# Training loop - FVU now in output
sae_output = sae(batch)
print(f"FVU: {sae_output['fvu']:.4f}")  # ← NEW metric
```

---

## API Reference

### New Functions in `config.py`

#### `get_model_config(model_name: str) -> dict`
Get model-specific configuration.

```python
config = get_model_config("gemma-2-2b")
# Returns: {"act_size": 2304, "n_layers": 26}
```

#### `get_hook_name(site: str, layer: int, model_name: str) -> str`
Get correct hook name for the model.

```python
hook = get_hook_name("resid_pre", 8, "gemma-2-2b")
# Returns: "blocks.8.hook_resid_pre"
```

#### `compute_fvu(x_original: Tensor, x_reconstruct: Tensor) -> Tensor`
Compute Fraction of Variance Unexplained.

```python
fvu = compute_fvu(original_acts, reconstructed_acts)
# Returns: scalar tensor (e.g., 0.182)
```

### Updated Outputs

All SAE classes now include FVU in their output dictionaries:

```python
output = sae(batch)
# Standard SAEs:
output["fvu"]  # Overall FVU

# Matryoshka SAEs/Transcoders:
output["fvu"]       # Final reconstruction FVU
output["fvu_min"]   # Best group FVU
output["fvu_max"]   # Worst group FVU
output["fvu_mean"]  # Average group FVU
```

---

## Examples

### Example 1: Training GPT-2 with FVU Monitoring

```python
cfg = get_default_cfg()
cfg["model_name"] = "gpt2-small"
cfg = post_init_cfg(cfg)

model = HookedTransformer.from_pretrained_no_processing("gpt2-small").to("cuda")
activation_store = ActivationsStore(model, cfg)
sae = GlobalBatchTopKMatryoshkaSAE(cfg)

# Train and monitor FVU
for i in range(num_batches):
    batch = activation_store.next_batch()
    output = sae(batch)
    
    print(f"Step {i}: FVU={output['fvu']:.4f}, "
          f"FVU_min={output['fvu_min']:.4f}, "
          f"FVU_max={output['fvu_max']:.4f}")
```

### Example 2: Gemma-2-2B Transcoder

```python
from transcoder_activation_store import create_transcoder_config

cfg = get_default_cfg()
cfg["model_name"] = "gemma-2-2b"
cfg["dict_size"] = 18432
cfg["group_sizes"] = [2304, 4608, 6912, 4608]
cfg["top_k"] = 64

# Configure for MLP transformation
cfg = create_transcoder_config(
    cfg,
    source_layer=8,
    target_layer=8,
    source_site="resid_mid",
    target_site="mlp_out"
)

model = HookedTransformer.from_pretrained_no_processing("gemma-2-2b").to("cuda")
activation_store = TranscoderActivationsStore(model, cfg)
transcoder = MatryoshkaTranscoder(cfg)

train_transcoder(transcoder, activation_store, model, cfg)
```

### Example 3: Comparing Models

```python
from config import get_model_config

for model_name in ["gpt2-small", "gemma-2-2b"]:
    config = get_model_config(model_name)
    print(f"{model_name}:")
    print(f"  act_size: {config['act_size']}")
    print(f"  dict_size (16x): {config['act_size'] * 16}")
```

---

## Troubleshooting

### "act_size mismatch"
**Problem**: Manual `act_size` conflicts with model.
**Solution**: Remove `act_size` from config, let `post_init_cfg()` detect it:
```python
cfg = get_default_cfg()
# Don't set cfg["act_size"] manually
cfg["model_name"] = "gemma-2-2b"
cfg = post_init_cfg(cfg)  # Auto-detects act_size=2304
```

### "CUDA out of memory" with Gemma-2
**Problem**: Gemma models are larger.
**Solution**: Reduce batch sizes:
```python
cfg["model_batch_size"] = 4  # Smaller
cfg["batch_size"] = 1024  # Smaller
```

### "FVU > 1.0"
**Problem**: Reconstruction worse than baseline.
**Solution**: 
- Increase `dict_size`
- Increase `top_k`
- Train longer
- Check for bugs in custom code

### "Hook not found" with Gemma
**Problem**: Using wrong hook name format.
**Solution**: Use `get_hook_name()` or `create_transcoder_config()`:
```python
# Don't manually construct hook names
# DO use helper functions
cfg = post_init_cfg(cfg)  # Sets correct hook_point
```

---

## Summary

### What Changed

1. **FVU Metrics Added**:
   - All SAE classes output `fvu`
   - Matryoshka models output `fvu_min`, `fvu_max`, `fvu_mean`
   - FVU shown in progress bar
   - FVU logged to W&B

2. **Gemma-2 Support Added**:
   - Auto-detection of activation sizes
   - Proper hook name handling
   - Model-specific defaults
   - Example training script

3. **Backward Compatible**:
   - Existing code works unchanged
   - Only new features added, nothing removed
   - Optional to use new functionality

### Quick Migration Checklist

- [ ] Update `config.py` (already done)
- [ ] Update `sae.py` (already done)
- [ ] Update `training.py` (already done)
- [ ] Update `transcoder_activation_store.py` (already done)
- [ ] Add `train_gemma_example.py` (already done)
- [ ] Test with GPT-2 (backward compatibility)
- [ ] Test with Gemma-2-2B (new functionality)
- [ ] Monitor FVU metrics in training

### Files Modified

- ✅ `config.py` - Added model configs, hook helpers, FVU computation
- ✅ `sae.py` - Added FVU to all loss dicts
- ✅ `training.py` - Added FVU to progress bars
- ✅ `transcoder_activation_store.py` - Updated to use config helpers
- ✅ `train_gemma_example.py` - New example script

### Ready to Use!

```bash
# Test with GPT-2 (existing)
python main.py

# Test with Gemma-2-2B (new)
python train_gemma_example.py sae

# See model comparison
python train_gemma_example.py compare
```

