# Gemma-2 Hook Correction: Addressing Reviewer Feedback

## 🎯 The Problem

A reviewer identified a critical issue in our Gemma-2 transcoder implementation:

> **"`resid_mid` isn't wrong, but it's pre-norm; the MLP runs on the post-norm vector. Using `ln2.hook_normalized` (and multiplying by the RMSNorm weight) aligns your input with what the MLP actually consumes."**

## 🔍 Technical Analysis

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

## ✅ The Solution

### 1. Correct Hook Usage

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

### 2. Implementation Changes

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

#### Updated Activation Store

```python
def get_paired_activations(self, batch_tokens):
    """Get activations with optional RMSNorm scaling."""
    source_acts = cache[self.source_hook_point]
    target_acts = cache[self.target_hook_point]
    
    # Apply RMSNorm scaling for Gemma-2 models if needed
    if self.config.get("apply_rmsnorm_scaling", False):
        from utils.config import apply_rmsnorm_scaling
        source_acts = apply_rmsnorm_scaling(
            source_acts, 
            self.model, 
            self.config["source_layer"]
        )
    
    return source_acts, target_acts
```

## 📊 Impact Assessment

### Scientific Validity
- **CORRECT**: Captures true MLP transformation
- **LEGACY**: Captures pre-norm → MLP transformation (different)

### Reconstruction Quality
- **CORRECT**: Better FVU scores (closer to actual MLP behavior)
- **LEGACY**: Slightly worse FVU (learning wrong transformation)

### Interpretability
- **CORRECT**: Features align with actual MLP input space
- **LEGACY**: Features in pre-norm space (less interpretable)

## 🚀 Usage Examples

### Correct Approach (Recommended)

```bash
# Use the corrected implementation
python train_gemma2_corrected.py
```

**What it does:**
- Uses `ln2.hook_normalized` as source
- Applies RMSNorm scaling: `x_normalized * ln2.w`
- Learns true MLP transformation
- Better reconstruction quality

### Legacy Approach (For Comparison)

```bash
# Use the legacy implementation for comparison
python train_gemma2_legacy.py
```

**What it does:**
- Uses `resid_mid` as source (pre-norm)
- No RMSNorm scaling
- Learns pre-norm → MLP transformation
- Still works, but not the true MLP transformation

## 🔧 Configuration Options

### For Correct Implementation

```python
# Use the corrected configuration helper
transcoder_cfg = create_gemma_mlp_transcoder_config(
    base_cfg, 
    layer=13, 
    use_ln2_normalized=True  # CORRECT approach
)
```

### For Legacy Implementation

```python
# Use the legacy configuration
transcoder_cfg = create_gemma_mlp_transcoder_config(
    base_cfg, 
    layer=13, 
    use_ln2_normalized=False  # LEGACY approach
)
```

## 📈 Expected Results

### Correct Implementation
- **FVU**: Should be lower (better reconstruction)
- **Features**: More interpretable (in MLP input space)
- **Scientific validity**: Captures true MLP transformation

### Legacy Implementation
- **FVU**: Slightly higher (worse reconstruction)
- **Features**: Less interpretable (in pre-norm space)
- **Scientific validity**: Captures pre-norm transformation

## 🎓 Key Takeaways

1. **Always use `ln2.hook_normalized` for Gemma-2 MLP transcoders**
2. **Apply RMSNorm scaling**: `x_normalized * ln2.w`
3. **This captures the TRUE MLP transformation**
4. **`resid_mid` is pre-norm, not what the MLP actually sees**

## 🔗 References

- [TransformerLens Documentation](https://transformerlensorg.github.io/TransformerLens/)
- [Gemma-2 Model Architecture](https://huggingface.co/google/gemma-2-2b)
- [RMSNorm vs LayerNorm](https://arxiv.org/abs/1910.07467)

## 📝 Migration Guide

### From Legacy to Correct

1. **Update configuration:**
   ```python
   # OLD
   cfg["source_site"] = "resid_mid"
   
   # NEW
   cfg["source_site"] = "ln2_normalized"
   cfg["apply_rmsnorm_scaling"] = True
   ```

2. **Update activation store:**
   ```python
   # The activation store now automatically applies scaling
   # when apply_rmsnorm_scaling=True
   ```

3. **Update model loading:**
   ```python
   # Enable hook_mlp_in for optional pre-norm access
   hf_cfg = HookedTransformerConfig.from_pretrained(
       "google/gemma-2-2b",
       use_hook_mlp_in=True,
   )
   ```

### Verification

To verify you're using the correct approach:

```python
# Check that RMSNorm scaling is enabled
assert cfg.get("apply_rmsnorm_scaling", False) == True

# Check that source hook is ln2_normalized
assert "ln2.hook_normalized" in cfg["source_hook_point"]

# Check that scaling is applied in activation store
# (This happens automatically when apply_rmsnorm_scaling=True)
```

## ✅ Summary

The reviewer was **absolutely correct**. Using `resid_mid` captures a pre-norm transformation, not the true MLP transformation. The correct approach uses `ln2.hook_normalized` with RMSNorm scaling to capture what the MLP actually sees as input.

This correction improves:
- **Scientific validity** (true MLP transformation)
- **Reconstruction quality** (better FVU scores)
- **Feature interpretability** (features in MLP input space)

Thank you to the reviewer for catching this important technical detail! 🙏
