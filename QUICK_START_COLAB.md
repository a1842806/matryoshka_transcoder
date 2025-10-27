# Quick Start: Google Colab Training

## 🚀 Copy-Paste Commands for Google Colab

### Step 1: Setup (Run Once)

```python
# Clone repository
!git clone https://github.com/a1842806/matryoshka_transcoder.git
%cd matryoshka_transcoder

# Switch to the nested groups branch
!git checkout interpretability-evaluation

# Install dependencies
!pip install torch transformers transformer-lens wandb datasets einops jaxtyping -q

# Check GPU
import torch
print(f"✓ CUDA available: {torch.cuda.is_available()}")
print(f"✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Login to W&B (optional - for metric tracking)
import wandb
wandb.login()
```

### Step 2: Train Model (Choose One)

#### Option A: Layer 17 (Recommended - ~6 hours on T4)
```python
!python src/scripts/train_gemma_layer17_with_warmup_decay_samples.py
```
- **Best for**: Final training, research quality
- **Features**: 15k steps, activation samples, optimized hyperparameters
- **Output**: High-quality nested group transcoder

#### Option B: Layer 8 (Quick Test - ~30 min on T4)
```python
!python src/scripts/train_gemma_layer8_with_warmup_decay_samples.py
```
- **Best for**: Testing setup, quick experiments
- **Features**: 1k steps, faster iteration
- **Output**: Proof of concept model

#### Option C: Layer 12 (Alternative - ~8 hours on T4)
```python
!python src/scripts/train_gemma_layer12_with_warmup_decay_samples.py
```
- **Best for**: Mid-layer analysis
- **Features**: 20k steps, comprehensive training

### Step 3: Monitor Progress

Your W&B dashboard link will appear in the output:
```
W&B dashboard: https://wandb.ai/YOUR_USERNAME/PROJECT_NAME
```

Watch these metrics:
- ✅ `loss` decreasing
- ✅ `fvu` < 0.1 (good reconstruction)
- ✅ `dead_features_percentage` < 10%
- ✅ `learning_rate` following warmup+decay schedule

### Step 4: Save Results to Google Drive

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy checkpoints
!cp -r checkpoints/transcoder/gemma-2-2b/ /content/drive/MyDrive/matryoshka_checkpoints/

# Copy activation samples
!cp -r checkpoints/transcoder/gemma-2-2b/*_activation_samples/ /content/drive/MyDrive/matryoshka_samples/

print("✓ Saved to Google Drive!")
```

---

## 📊 What Changed: Nested Groups

### Before (Your Old Implementation - Disjoint):
```python
Group 0: features [0-1151]       (1,152 features)
Group 1: features [1152-3455]    (2,304 features)
Group 2: features [3456-8063]    (4,608 features)
Group 3: features [8064-18431]   (10,368 features)
```
Each group was **separate and independent**.

### After (New Implementation - Nested):
```python
Group 0: features [0-1151]       (1,152 features)
Group 1: features [0-3455]       (3,456 features) ← includes Group 0
Group 2: features [0-8063]       (8,064 features) ← includes Groups 0+1
Group 3: features [0-18431]      (18,432 features) ← all features
```
Each group **includes all previous groups** (hierarchical nesting).

**Why this matters:**
- ✨ Matches original Matryoshka paper design
- ✨ Hierarchical feature learning (coarse → fine)
- ✨ Each group is a complete model
- ✨ Adaptive complexity based on task

---

## 🔧 Troubleshooting

### Out of Memory?
```python
# Edit the training script before running:
# Reduce batch_size from 1024 to 512
# Reduce model_batch_size from 4 to 2
```

### Need GPU?
1. Runtime → Change runtime type
2. Select T4 GPU (free) or A100 (paid)
3. Click Save

### Training Too Slow?
Use Layer 8 script for quick testing (30 minutes vs 6 hours)

---

## 📈 After Training

### Evaluate Your Model
```python
!python src/eval/evaluate_interpretability_standalone.py \
    --checkpoint checkpoints/transcoder/gemma-2-2b/YOUR_MODEL_FOLDER \
    --layer 17 \
    --batches 500
```

### Analyze Activation Samples
```python
!python src/utils/analyze_activation_samples.py \
    checkpoints/transcoder/gemma-2-2b/YOUR_MODEL_activation_samples/
```

---

## ✅ Success Checklist

- [ ] GPU is available (T4 or better)
- [ ] Repository cloned and on `interpretability-evaluation` branch
- [ ] Dependencies installed
- [ ] W&B logged in (optional)
- [ ] Training script running
- [ ] Metrics visible on W&B dashboard
- [ ] Checkpoints saved to Google Drive

---

**Need help?** Check `COLAB_TRAINING.md` for detailed documentation.

