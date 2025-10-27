# Training Matryoshka Transcoder on Google Colab

This guide shows you how to train the Matryoshka Transcoder (with **nested groups**) on Google Colab.

## Architecture Update: Nested Groups

The implementation now uses **nested groups** matching the original Matryoshka paper:

- **Group 0**: Features 0-1151 (1,152 features)
- **Group 1**: Features 0-3455 (3,456 features - includes Group 0)
- **Group 2**: Features 0-8063 (8,064 features - includes Groups 0+1)
- **Group 3**: Features 0-18431 (18,432 features - all features)

Each group builds hierarchically on previous groups, enabling adaptive complexity.

## Setup on Google Colab

### 1. Clone the Repository

```python
# Clone your repository
!git clone https://github.com/YOUR_USERNAME/matryoshka_sae.git
%cd matryoshka_sae
```

### 2. Install Dependencies

```python
# Install required packages
!pip install torch transformers transformer-lens wandb datasets
!pip install einops jaxtyping
```

### 3. Login to Weights & Biases (Optional)

```python
import wandb
wandb.login()
```

### 4. Check GPU Availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## Training Commands

### Layer 17 Training (Recommended - 15k steps)

```bash
!python src/scripts/train_gemma_layer17_with_warmup_decay_samples.py
```

**Configuration:**
- **Layer**: 17 (deeper layer for analysis)
- **Dictionary size**: 18,432 (8x expansion)
- **Group sizes**: [1152, 2304, 4608, 10368] (NESTED)
- **Top-k**: 96 features per sample
- **Training tokens**: 15M (~15k steps)
- **Learning rate**: 4e-4 with warmup + decay
- **Features**: Activation sample collection enabled

### Layer 12 Training (Alternative - 20k steps)

```bash
!python src/scripts/train_gemma_layer12_with_warmup_decay_samples.py
```

**Configuration:**
- **Layer**: 12 (mid-layer)
- **Dictionary size**: 36,864 (16x expansion)
- **Group sizes**: [2304, 4608, 9216, 20736] (NESTED)
- **Top-k**: 96 features
- **Training tokens**: 20M (~20k steps)

### Layer 8 Training (Quick Test - 1k steps)

```bash
!python src/scripts/train_gemma_layer8_with_warmup_decay_samples.py
```

**Configuration:**
- **Layer**: 8 (early layer)
- **Dictionary size**: 18,432 (8x expansion)
- **Training tokens**: 1M (~1k steps)
- **Good for testing setup**

## Monitoring Training

### View W&B Dashboard

Training metrics are automatically logged to Weights & Biases:

```python
# Your W&B dashboard URL will be printed during training
# https://wandb.ai/YOUR_USERNAME/PROJECT_NAME
```

**Key Metrics to Monitor:**
- `loss`: Total training loss
- `l2_loss`: Reconstruction error
- `l0_norm`: Sparsity (number of active features)
- `fvu`: Fraction of Variance Unexplained (lower is better)
- `dead_features_percentage`: % of inactive features (target: <10%)
- `learning_rate`: LR schedule visualization

### View Logs in Colab

```python
# Training progress shows in the notebook output
# Watch for:
# - Loss decreasing
# - FVU < 0.1 (good reconstruction)
# - Dead features < 10%
```

## Accessing Results

### Download Checkpoints

```python
# Checkpoints are saved in: checkpoints/transcoder/gemma-2-2b/
# Download them to your local machine or Google Drive

from google.colab import files

# Download final checkpoint
!zip -r final_checkpoint.zip checkpoints/transcoder/gemma-2-2b/

# Download to local machine
files.download('final_checkpoint.zip')
```

### Mount Google Drive (Save Checkpoints)

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy checkpoints to Drive
!cp -r checkpoints/transcoder/gemma-2-2b/ /content/drive/MyDrive/matryoshka_checkpoints/
```

## Nested Groups Explanation

### Before (Disjoint Groups):
```python
# Each group was separate
reconstruction = bias + group0 + group1 + group2 + group3
# Where each group had distinct features
```

### After (Nested Groups):
```python
# Each group includes all previous groups
# Group 0: [features 0-1151]
# Group 1: [features 0-3455]  <- includes Group 0
# Group 2: [features 0-8063]  <- includes Groups 0+1
# Group 3: [features 0-18431] <- all features

# Reconstruction uses cumulative features
reconstruction = bias + features[0:end_idx] @ W_dec[0:end_idx]
```

**Benefits:**
- Hierarchical feature learning (coarse → fine)
- Adaptive complexity based on task
- Each group is a complete model on its own
- Matches original Matryoshka paper design

## Troubleshooting

### Out of Memory Error

```python
# Reduce batch size in training script
cfg["batch_size"] = 512  # instead of 1024
cfg["model_batch_size"] = 2  # instead of 4
```

### CUDA Error

```python
# Ensure GPU runtime is enabled
# Runtime → Change runtime type → T4 GPU (or better)
```

### Training Too Slow

```python
# Use shorter training run for testing
cfg["num_tokens"] = int(1e6)  # 1M tokens instead of 15M
```

## Expected Training Time

| GPU Type | Layer 17 (15k steps) | Layer 12 (20k steps) | Layer 8 (1k steps) |
|----------|---------------------|---------------------|-------------------|
| T4 (Free)| ~6-8 hours          | ~8-10 hours         | ~30-45 min        |
| A100     | ~2-3 hours          | ~3-4 hours          | ~10-15 min        |
| V100     | ~3-4 hours          | ~4-5 hours          | ~15-20 min        |

## Next Steps After Training

1. **Evaluate Interpretability**:
   ```bash
   !python src/eval/evaluate_interpretability_standalone.py \
       --checkpoint checkpoints/transcoder/gemma-2-2b/YOUR_MODEL \
       --layer 17 \
       --batches 1000
   ```

2. **Compare with Google's Transcoder**:
   ```bash
   !python src/eval/compare_interpretability.py \
       --ours_checkpoint checkpoints/transcoder/gemma-2-2b/YOUR_MODEL \
       --google_dir /path/to/gemma_scope \
       --layer 17
   ```

3. **Analyze Activation Samples**:
   ```bash
   !python src/utils/analyze_activation_samples.py \
       checkpoints/transcoder/gemma-2-2b/YOUR_MODEL_activation_samples/
   ```

## Questions?

Check the main README.md or open an issue on GitHub.

