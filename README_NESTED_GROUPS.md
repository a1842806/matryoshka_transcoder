# Matryoshka Transcoder - Nested Groups Implementation

## 🎯 Quick Start: Google Colab Training

### Method 1: Automated Script (Easiest)
```python
# In a new Colab cell, paste and run:
!git clone https://github.com/a1842806/matryoshka_transcoder.git
%cd matryoshka_transcoder
!git checkout interpretability-evaluation
!python colab_train.py --layer 17
```

### Method 2: Manual Setup
```python
# Clone and setup
!git clone https://github.com/a1842806/matryoshka_transcoder.git
%cd matryoshka_transcoder
!git checkout interpretability-evaluation
!pip install torch transformers transformer-lens wandb datasets einops jaxtyping -q

# Train
!python src/scripts/train_gemma_layer17_with_warmup_decay_samples.py
```

## 📖 What's New: Nested Groups Architecture

### Architecture Change

Your Matryoshka Transcoder now implements **nested groups** matching the original paper:

**Before (Disjoint Groups)**:
- Group 0: Features 0-1151 (separate)
- Group 1: Features 1152-3455 (separate)
- Group 2: Features 3456-8063 (separate)
- Group 3: Features 8064-18431 (separate)

**After (Nested Groups)** ✅:
- Group 0: Features 0-1151 (base)
- Group 1: Features 0-3455 (includes Group 0)
- Group 2: Features 0-8063 (includes Groups 0+1)
- Group 3: Features 0-18431 (all features)

### Key Benefits

1. **Hierarchical Learning**: Features naturally organize from coarse to fine
2. **Adaptive Complexity**: Each group is a complete model
3. **Paper Alignment**: Matches original Matryoshka design
4. **Better Interpretability**: Clear feature hierarchy

## 📁 Documentation Files

| File | Purpose |
|------|---------|
| `QUICK_START_COLAB.md` | Copy-paste commands for immediate use |
| `COLAB_TRAINING.md` | Comprehensive training guide |
| `NESTED_GROUPS_SUMMARY.md` | Complete architecture explanation |
| `colab_train.py` | Automated one-command training script |

## 🚀 Training Options

### Layer 17 (Recommended)
```bash
!python src/scripts/train_gemma_layer17_with_warmup_decay_samples.py
```
- ⏱️ Time: ~6 hours on T4 GPU
- 📊 Steps: 15,000
- 🎯 Features: 18,432 (nested groups)
- ✨ Best for: Research quality results

### Layer 8 (Quick Test)
```bash
!python src/scripts/train_gemma_layer8_with_warmup_decay_samples.py
```
- ⏱️ Time: ~30 minutes on T4 GPU
- 📊 Steps: 1,000
- 🎯 Features: 18,432 (nested groups)
- ✨ Best for: Testing setup

### Layer 12 (Alternative)
```bash
!python src/scripts/train_gemma_layer12_with_warmup_decay_samples.py
```
- ⏱️ Time: ~8 hours on T4 GPU
- 📊 Steps: 20,000
- 🎯 Features: 36,864 (nested groups)
- ✨ Best for: Mid-layer analysis

## 📊 Monitoring Training

### Weights & Biases Dashboard
Your training automatically logs to W&B. The dashboard link appears in the output:
```
W&B dashboard: https://wandb.ai/YOUR_USERNAME/PROJECT_NAME
```

### Key Metrics to Watch
- ✅ `loss` - Should decrease steadily
- ✅ `fvu` - Target: < 0.1 (good reconstruction)
- ✅ `dead_features_percentage` - Target: < 10%
- ✅ `learning_rate` - Should follow warmup + cosine decay
- ✅ `l0_norm` - Number of active features (~96 for Layer 17)

### Group-Specific Metrics
- `fvu_min` - Best group's FVU
- `fvu_max` - Worst group's FVU
- `fvu_mean` - Average FVU across groups

## 💾 Saving Results

### To Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')

# Save checkpoints
!cp -r checkpoints/transcoder/gemma-2-2b/ /content/drive/MyDrive/matryoshka_checkpoints/

# Save activation samples
!cp -r checkpoints/transcoder/gemma-2-2b/*_activation_samples/ /content/drive/MyDrive/matryoshka_samples/
```

### To Local Machine
```python
from google.colab import files

# Zip and download
!zip -r model_checkpoint.zip checkpoints/transcoder/gemma-2-2b/
files.download('model_checkpoint.zip')
```

## 🔍 After Training

### 1. Evaluate Interpretability
```bash
!python src/eval/evaluate_interpretability_standalone.py \
    --checkpoint checkpoints/transcoder/gemma-2-2b/YOUR_MODEL_FOLDER \
    --layer 17 \
    --batches 1000
```

### 2. Compare with Google's Transcoder
```bash
!python src/eval/compare_interpretability.py \
    --ours_checkpoint checkpoints/transcoder/gemma-2-2b/YOUR_MODEL \
    --google_dir /path/to/gemma_scope \
    --layer 17
```

### 3. Analyze Activation Samples
```bash
!python src/utils/analyze_activation_samples.py \
    checkpoints/transcoder/gemma-2-2b/YOUR_MODEL_activation_samples/
```

## 🛠️ Technical Details

### Code Changes
The main changes are in `src/models/sae.py`:

```python
# Decoding with nested groups
for i in range(self.active_groups):
    start_idx = 0  # Always start from 0 (nested)
    end_idx = self.group_indices[i+1]  # Cumulative end
    W_dec_slice = self.W_dec[start_idx:end_idx, :]
    acts_topk_slice = all_acts_topk[:, start_idx:end_idx]
    x_reconstruct = acts_topk_slice @ W_dec_slice + self.b_dec
    intermediate_reconstructs.append(x_reconstruct)
```

### Group Indices
For `group_sizes = [1152, 2304, 4608, 10368]`:
```python
self.group_indices = [0, 1152, 3456, 8064, 18432]
```

Each group uses features from `0` to `group_indices[i+1]`.

## 🆘 Troubleshooting

### Out of Memory
```python
# Edit training script before running:
cfg["batch_size"] = 512  # Reduce from 1024
cfg["model_batch_size"] = 2  # Reduce from 4
```

### No GPU Available
1. Runtime → Change runtime type
2. Select **T4 GPU** (free) or **A100** (paid)
3. Save and reconnect

### Training Too Slow
Use Layer 8 for quick testing (30 min vs 6 hours)

### Import Errors
```python
# Reinstall dependencies
!pip install --upgrade torch transformers transformer-lens wandb datasets einops jaxtyping
```

## 📚 Repository Structure

```
matryoshka_transcoder/
├── src/
│   ├── models/
│   │   └── sae.py                    # ✅ Updated: Nested groups
│   ├── scripts/
│   │   ├── train_gemma_layer17_...py # Training scripts
│   │   ├── train_gemma_layer12_...py
│   │   └── train_gemma_layer8_...py
│   └── eval/
│       ├── evaluate_interpretability_standalone.py
│       └── compare_interpretability.py
├── colab_train.py                    # ✅ New: Automated training
├── QUICK_START_COLAB.md              # ✅ New: Quick reference
├── COLAB_TRAINING.md                 # ✅ New: Detailed guide
└── NESTED_GROUPS_SUMMARY.md          # ✅ New: Architecture docs
```

## ✅ Verification

To verify nested groups are working correctly:

```python
# After training starts, check logs for:
# "Matryoshka group configuration - NESTED GROUPS"
# "NESTED: Always start from 0"

# Or check in code:
from src.models.sae import MatryoshkaTranscoder
print(transcoder.group_indices)  # Should show cumulative sums
```

## 🎓 Learning Resources

- **Original Paper**: [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)
- **SAE Bench**: https://github.com/adamkarvonen/SAEBench
- **Anthropic Research**: https://transformer-circuits.pub/

## 📞 Support

- **Quick Reference**: `QUICK_START_COLAB.md`
- **Detailed Guide**: `COLAB_TRAINING.md`
- **Architecture Explanation**: `NESTED_GROUPS_SUMMARY.md`
- **Issues**: GitHub repository issues tab

## 🎉 Ready to Train!

Your nested groups implementation is complete and ready for training on Google Colab!

**Recommended command**:
```python
!python colab_train.py --layer 17
```

This will automatically:
1. ✅ Clone repository
2. ✅ Install dependencies
3. ✅ Check GPU
4. ✅ Train with nested groups
5. ✅ Save to Google Drive
6. ✅ Provide next steps

**Happy training! 🚀**

