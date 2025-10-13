# Gemma Models & FVU Metrics Guide

This comprehensive guide covers FVU metrics, Gemma-2 model support, hook corrections, and multi-GPU optimization.

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

All Transcoder classes output FVU metrics:

#### Matryoshka Transcoders:
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

### Using FVU in Code

```python
from utils.config import compute_fvu

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

### Training with Gemma-2-2B

#### Quick Example:

```python
from transformer_lens import HookedTransformer
from models.sae import MatryoshkaTranscoder
from models.transcoder_activation_store import TranscoderActivationsStore
from training import train_transcoder
from utils.config import get_default_cfg, post_init_cfg, create_gemma_mlp_transcoder_config

# Configure for Gemma-2-2B transcoder
cfg = get_default_cfg()
cfg["model_name"] = "gemma-2-2b"
cfg = create_gemma_mlp_transcoder_config(cfg, layer=8)
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

# Train transcoder
activation_store = TranscoderActivationsStore(model, cfg)
transcoder = MatryoshkaTranscoder(cfg)
train_transcoder(transcoder, activation_store, model, cfg)
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

---

## Gemma-2 Hook Correction

### The Problem

A reviewer identified a critical issue in our Gemma-2 transcoder implementation:

> **"`resid_mid` isn't wrong, but it's pre-norm; the MLP runs on the post-norm vector. Using `ln2.hook_normalized` (and multiplying by the RMSNorm weight) aligns your input with what the MLP actually consumes."**

### Gemma-2-2B Architecture Flow

```
Input → Attention → resid_mid → ln2 (RMSNorm) → ln2.w (scale) → MLP → mlp_out
                    ↑              ↑                ↑              ↑
                (pre-norm)    (post-norm)     (actual MLP)   (MLP output)
```

### The Issue with `resid_mid`

**What we were doing (INCORRECT):**
```python
# Source: resid_mid (pre-norm activations)
# Target: mlp_out (MLP output)
# Problem: Learning resid_mid → mlp_out (not the true MLP transformation)
```

**What we should do (CORRECT):**
```python
# Source: ln2.hook_normalized * ln2.w (actual MLP input)
# Target: mlp_out (MLP output)
# Result: Learning true MLP transformation
```

### The Solution

#### Correct Hook Usage

```python
from transformer_lens import HookedTransformer, HookedTransformerConfig

# Enable proper hooks
cfg = HookedTransformerConfig.from_pretrained(
    "google/gemma-2-2b",
    use_hook_mlp_in=True,  # Optional: exposes pre-norm hook
)
model = HookedTransformer.from_pretrained("google/gemma-2-2b", hf_cfg=cfg)

# CORRECT: Get actual MLP input
x_normalized = cache[f"blocks.{L}.ln2.hook_normalized"]  # Post-norm
ln2_weight = model.blocks[L].ln2.w                      # RMSNorm weight
x_mlp_input = x_normalized * ln2_weight                  # Actual MLP input

# Target: MLP output
y_mlp_output = cache[f"blocks.{L}.hook_mlp_out"]
```

#### Updated Configuration Helper

```python
def create_gemma_mlp_transcoder_config(base_cfg, layer, use_ln2_normalized=True):
    """
    Create proper transcoder configuration for Gemma-2 MLP transformation.
    
    Args:
        use_ln2_normalized: If True, use correct approach. If False, use legacy.
    """
    if use_ln2_normalized:
        # CORRECT: Use post-RMSNorm activations
        cfg["source_site"] = "ln2_normalized"
        cfg["apply_rmsnorm_scaling"] = True
        cfg["source_hook_point"] = f"blocks.{layer}.ln2.hook_normalized"
    else:
        # LEGACY: Use pre-norm activations
        cfg["source_site"] = "resid_mid"
        cfg["apply_rmsnorm_scaling"] = False
        cfg["source_hook_point"] = f"blocks.{layer}.hook_resid_mid"
```

### Impact Assessment

#### Scientific Validity
- ✅ **CORRECT**: Captures true MLP transformation
- ⚠️ **LEGACY**: Captures pre-norm → MLP transformation (different)

#### Reconstruction Quality
- ✅ **CORRECT**: Better FVU scores (closer to actual MLP behavior)
- ⚠️ **LEGACY**: Slightly worse FVU (learning wrong transformation)

#### Interpretability
- ✅ **CORRECT**: Features align with actual MLP input space
- ⚠️ **LEGACY**: Features in pre-norm space (less interpretable)

### Usage Examples

#### Correct Approach (Recommended)

```bash
# Use the corrected implementation
python train_gemma2_corrected.py
```

**What it does:**
- Uses `ln2.hook_normalized` as source
- Applies RMSNorm scaling: `x_normalized * ln2.w`
- Learns true MLP transformation
- Better reconstruction quality

#### Legacy Approach (For Comparison)

```bash
# Use the legacy implementation for comparison
python train_gemma2_legacy.py
```

**What it does:**
- Uses `resid_mid` as source (pre-norm)
- No RMSNorm scaling
- Learns pre-norm → MLP transformation
- Still works, but not the true MLP transformation

---

## Multi-GPU Setup

### Your Setup: 2x 24GB GPUs (48GB Total)

**Excellent configuration!** You can handle:
- ✅ Gemma-2-2B (needs ~8GB)
- ✅ Gemma-2-9B (needs ~16GB) 
- ✅ Multiple models in parallel
- ✅ Large batch sizes
- ✅ High-performance evaluation

### Quick Start

#### 1. Check Your GPU Status
```bash
python src/scripts/check_gpu.py
```

#### 2. Use Multi-GPU Evaluation
```bash
# For your Gemma-2-2B checkpoint
python src/scripts/run_evaluation_multi_gpu.py
```

#### 3. Monitor Both GPUs
```bash
# In another terminal
watch -n 1 nvidia-smi
```

### Memory Requirements by Model

| Model | GPU Memory Needed | CPU Memory Needed | Safe Batch Size |
|-------|------------------|-------------------|-----------------|
| GPT-2 Small | ~2GB | ~4GB | 50 batches |
| GPT-2 Medium | ~4GB | ~8GB | 30 batches |
| Gemma-2-2B | ~8GB | ~16GB | 10 batches |
| Gemma-2-9B | ~16GB | ~32GB | 5 batches |
| Gemma-2-27B | ~32GB | ~64GB | 3 batches |

### Multi-GPU Evaluation Benefits

#### `run_evaluation_multi_gpu.py` Features:
- ✅ **Optimized for 2x 24GB**: Uses 100-200 batches (vs 10-50 for safe)
- ✅ **Memory distribution**: Spreads load across both GPUs
- ✅ **High-performance**: Much faster than single-GPU
- ✅ **Comprehensive results**: More reliable with larger sample size
- ✅ **Automatic optimization**: Adjusts based on available memory

#### Expected Performance:
- **Gemma-2-2B**: ~100 batches in 5-10 minutes
- **GPT-2 Small**: ~200 batches in 2-5 minutes
- **Memory usage**: ~15-20GB per GPU (well within 24GB limit)

### Training with Multi-GPU

#### Option 1: Parallel Training (Recommended)
Train different models on different GPUs:

```bash
# Terminal 1: Train GPT-2 Small on GPU 0
CUDA_VISIBLE_DEVICES=0 python src/scripts/train_gpt2_simple.py

# Terminal 2: Train Gemma-2-2B on GPU 1  
CUDA_VISIBLE_DEVICES=1 python src/scripts/train_gemma_example.py
```

#### Option 2: Large Model Training
Use both GPUs for a single large model:

```bash
# Train Gemma-2-9B using both GPUs
python src/scripts/train_gemma_example.py
# (Code will automatically use available GPUs)
```

### Memory Management

#### Your 48GB Total Memory:
- **GPU 0**: 24GB VRAM
- **GPU 1**: 24GB VRAM
- **System RAM**: 32GB+ recommended

#### Memory Allocation Strategy:
```
Gemma-2-2B Training:
├── GPU 0: Model (~8GB) + Activations (~8GB) = 16GB
├── GPU 1: Free for other tasks
└── System: Dataset + Python = ~8GB

Gemma-2-2B Evaluation:
├── GPU 0: Model (~8GB) + Evaluation (~12GB) = 20GB
├── GPU 1: Free for other tasks
└── System: Results + Python = ~4GB
```

### Advanced Multi-GPU Usage

#### 1. Parallel Evaluation
Evaluate multiple checkpoints simultaneously:

```bash
# Terminal 1: Evaluate GPT-2 checkpoint on GPU 0
CUDA_VISIBLE_DEVICES=0 python src/scripts/run_evaluation_multi_gpu.py checkpoints/transcoder/gpt2-small/model_1/

# Terminal 2: Evaluate Gemma checkpoint on GPU 1
CUDA_VISIBLE_DEVICES=1 python src/scripts/run_evaluation_multi_gpu.py checkpoints/transcoder/gemma-2-2b/model_2/
```

#### 2. Model Comparison
Train and evaluate different models in parallel:

```bash
# Terminal 1: GPT-2 Small
CUDA_VISIBLE_DEVICES=0 python src/scripts/train_gpt2_simple.py

# Terminal 2: Gemma-2-2B
CUDA_VISIBLE_DEVICES=1 python src/scripts/train_gemma_example.py

# Terminal 3: Monitor both
watch -n 1 nvidia-smi
```

### Monitoring Multi-GPU Usage

#### Real-time Monitoring:
```bash
# Monitor both GPUs
watch -n 1 nvidia-smi

# Monitor specific GPU
nvidia-smi -i 0  # GPU 0
nvidia-smi -i 1  # GPU 1

# Monitor processes
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

#### Memory Optimization:
```bash
# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Clear all GPU memory
for i in {0..1}; do
    python -c "import torch; torch.cuda.set_device($i); torch.cuda.empty_cache()"
done
```

### Troubleshooting Multi-GPU Issues

#### Issue: "CUDA out of memory" on one GPU
**Solution:**
```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=1 python src/scripts/run_evaluation_multi_gpu.py
```

#### Issue: Both GPUs showing high usage
**Solution:**
```bash
# Use memory-safe evaluation
python src/scripts/run_evaluation_safe.py
```

#### Issue: Model too large for single GPU
**Solution:**
```bash
# Use CPU evaluation
python src/scripts/run_evaluation_cpu.py
```

### Performance Optimization

#### 1. GPU Selection Strategy:
```bash
# Use GPU 0 for primary tasks
export CUDA_VISIBLE_DEVICES=0

# Use GPU 1 for secondary tasks  
export CUDA_VISIBLE_DEVICES=1

# Use both GPUs
export CUDA_VISIBLE_DEVICES=0,1
```

#### 2. Memory Optimization:
```python
# In your scripts, add:
import torch
torch.cuda.empty_cache()  # Clear cache before large operations
```

#### 3. Batch Size Optimization:
```python
# For 2x 24GB GPUs, you can use:
cfg["model_batch_size"] = 8-16    # Large model batches
cfg["batch_size"] = 2048-4096     # Large evaluation batches
cfg["num_tokens"] = int(1e8)      # More training data
```

### Recommended Workflow

#### For Your Gemma-2-2B Checkpoint:

1. **Check GPU status:**
   ```bash
   python src/scripts/check_gpu.py
   ```

2. **Use multi-GPU evaluation:**
   ```bash
   python src/scripts/run_evaluation_multi_gpu.py
   ```

3. **Monitor during evaluation:**
   ```bash
   # In another terminal
   watch -n 1 nvidia-smi
   ```

4. **If issues occur, fallback:**
   ```bash
   # Memory-safe evaluation
   python src/scripts/run_evaluation_safe.py
   
   # Or CPU evaluation
   python src/scripts/run_evaluation_cpu.py
   ```

### Expected Results with Your Setup

#### Gemma-2-2B Evaluation:
- **Time**: 5-10 minutes (vs 30+ minutes on CPU)
- **Batches**: 100-200 (vs 5-10 on memory-safe)
- **Memory**: ~20GB per GPU (well within 24GB limit)
- **Reliability**: Very high (large sample size)

#### GPT-2 Small Evaluation:
- **Time**: 2-5 minutes
- **Batches**: 200+ (comprehensive)
- **Memory**: ~10GB per GPU
- **Reliability**: Excellent

### Quick Commands Summary

```bash
# Check your 2x 24GB setup
python src/scripts/check_gpu.py

# Multi-GPU evaluation (recommended)
python src/scripts/run_evaluation_multi_gpu.py

# Monitor both GPUs
watch -n 1 nvidia-smi

# Use specific GPU
CUDA_VISIBLE_DEVICES=0 python src/scripts/run_evaluation_multi_gpu.py

# Fallback options
python src/scripts/run_evaluation_safe.py
python src/scripts/run_evaluation_cpu.py
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

3. **Hook Correction Implemented**:
   - Correct RMSNorm scaling for Gemma-2
   - Better reconstruction quality
   - More interpretable features

4. **Multi-GPU Optimization**:
   - Optimized for 2x 24GB setup
   - Parallel training and evaluation
   - Memory management strategies

5. **Backward Compatible**:
   - Existing code works unchanged
   - Only new features added, nothing removed
   - Optional to use new functionality

### Quick Migration Checklist

- [x] Update `config.py` (already done)
- [x] Update `sae.py` (already done)
- [x] Update `training.py` (already done)
- [x] Update `transcoder_activation_store.py` (already done)
- [x] Add `train_gemma_example.py` (already done)
- [x] Test with GPT-2 (backward compatibility)
- [x] Test with Gemma-2-2B (new functionality)
- [x] Monitor FVU metrics in training

### Ready to Use!

```bash
# Test with GPT-2 (existing)
python src/scripts/main.py

# Test with Gemma-2-2B (new)
python src/scripts/train_gemma_example.py sae

# See model comparison
python src/scripts/train_gemma_example.py compare

# Multi-GPU evaluation
python src/scripts/run_evaluation_multi_gpu.py
```

With your 2x 24GB GPU setup, you have **excellent** hardware for:
- ✅ **Fast evaluation**: Use `run_evaluation_multi_gpu.py`
- ✅ **Large models**: Gemma-2-2B, Gemma-2-9B
- ✅ **Parallel processing**: Multiple models simultaneously
- ✅ **High reliability**: Large batch sizes for accurate results

**No need for CPU evaluation** - your GPUs can handle everything! 🚀
