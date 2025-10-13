# Project Structure Guide

This document explains the organized project structure and how to navigate it.

## Directory Structure

```
matryoshka_sae/
├── src/                           # Source code (organized by purpose)
│   ├── models/                    # Model definitions
│   │   ├── sae.py                # SAE architectures (including Matryoshka)
│   │   ├── activation_store.py   # Activation collection for SAEs
│   │   └── transcoder_activation_store.py  # Activation collection for transcoders
│   │
│   ├── scripts/                   # Training & evaluation scripts
│   │   ├── main.py               # Train standard SAEs
│   │   ├── train_gpt2_simple.py  # Simple GPT-2 transcoder training
│   │   ├── train_gpt2_transcoder.py  # Full GPT-2 transcoder training
│   │   ├── train_gemma_example.py    # Gemma-2 examples
│   │   ├── train_matryoshka_transcoder.py  # Advanced transcoder training
│   │   ├── run_evaluation.py     # Evaluate trained models
│   │   └── evaluate_features.py  # Feature quality evaluation
│   │
│   ├── training/                  # Training loops & logic
│   │   └── training.py           # Core training functions
│   │
│   └── utils/                     # Utilities & configuration
│       ├── config.py             # Configuration management
│       ├── logs.py               # W&B logging & checkpointing
│       └── utils.py              # Helper functions
│
├── checkpoints/                   # Saved model checkpoints (organized)
│   ├── sae/                      # Sparse Autoencoder checkpoints
│   │   ├── gpt2-small/
│   │   ├── gpt2-medium/
│   │   ├── gemma-2-2b/
│   │   └── gemma-2-9b/
│   │
│   └── transcoder/               # Transcoder checkpoints
│       ├── gpt2-small/
│       ├── gpt2-medium/
│       ├── gemma-2-2b/
│       └── gemma-2-9b/
│
├── docs/                          # Documentation
│   ├── PROJECT_STRUCTURE.md      # This file
│   ├── FEATURE_EVALUATION_GUIDE.md  # Feature evaluation guide
│   ├── FVU_AND_GEMMA_GUIDE.md    # FVU metrics & Gemma support
│   ├── MATRYOSHKA_TRANSCODER.md  # Transcoder documentation
│   └── TRANSCODER_QUICKSTART.md  # Quick start guide
│
├── config/                        # Configuration files
│   └── default.json              # Default training config
│
├── logs/                          # Training logs
│   └── *.out                     # Output logs
│
├── metrics/                       # Evaluation metrics
│   ├── sae/
│   └── transcoder/
│
├── wandb/                         # W&B local files
│
└── README.md                      # Main project README
```

## Key Files

### Model Definitions (`src/models/`)

| File | Purpose |
|------|---------|
| `sae.py` | Transcoder model definitions: `BaseAutoencoder`, `MatryoshkaTranscoder` |
| `transcoder_activation_store.py` | Collects paired activations (source→target) for transcoder training |

### Training Scripts (`src/scripts/`)

| File | Purpose | When to Use |
|------|---------|-------------|
| `train_gpt2_simple.py` | Simple GPT-2 transcoder | Quick testing, learning |
| `train_gpt2_transcoder.py` | Full GPT-2 transcoder | Production training |
| `train_gemma_example.py` | Gemma-2 examples | Training on Gemma models |
| `train_matryoshka_transcoder.py` | Advanced multi-transcoder | Multiple transcoders in parallel |
| `run_evaluation.py` | Evaluate trained models | After training |
| `evaluate_features.py` | Feature quality metrics | Deep feature analysis |

### Training Logic (`src/training/`)

| File | Purpose |
|------|---------|
| `training.py` | Core training loops: `train_transcoder()`, `train_transcoder_group_seperate_wandb()`, etc. |

### Utilities (`src/utils/`)

| File | Purpose |
|------|---------|
| `config.py` | Config management, model-specific settings, FVU computation |
| `logs.py` | W&B logging, checkpoint saving |
| `utils.py` | Helper functions |

## Usage Patterns

### Training a Model

```bash
# From project root
cd /path/to/matryoshka_sae

# Train GPT-2 Small transcoder (simple)
python src/scripts/train_gpt2_simple.py

# Train with full W&B logging
python src/scripts/train_gpt2_transcoder.py

# Train Gemma-2
python src/scripts/train_gemma_example.py
```

### Evaluating a Model

```bash
# Automatic: finds most recent checkpoint
python src/scripts/run_evaluation.py

# Specify checkpoint
python src/scripts/run_evaluation.py checkpoints/transcoder/gpt2-small/model_name_9764/
```

### Loading a Checkpoint

```python
import torch
import json
from src.models.sae import MatryoshkaTranscoder

# Load config
with open("checkpoints/transcoder/gpt2-small/model_9764/config.json") as f:
    cfg = json.load(f)

# Load model
transcoder = MatryoshkaTranscoder(cfg)
transcoder.load_state_dict(
    torch.load("checkpoints/transcoder/gpt2-small/model_9764/sae.pt")
)
```

### Importing Modules

Since code is organized in `src/`, scripts use relative imports:

```python
# From src/scripts/ (e.g., train_gpt2_simple.py)
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.sae import MatryoshkaTranscoder
from models.transcoder_activation_store import TranscoderActivationsStore
from training.training import train_transcoder
from utils.config import get_default_cfg
```

## Checkpoint Organization

### Structure

```
checkpoints/{model_type}/{base_model}/{checkpoint_name}_{step}/
```

**Example:**
```
checkpoints/transcoder/gpt2-small/gpt2-small_blocks.8.hook_resid_mid_to_blocks.8.hook_mlp_out_12288_matryoshka-transcoder_64_0.0003_9764/
├── config.json          # Training configuration
├── sae.pt              # Model weights
└── evaluation/         # Optional: evaluation results
    └── results.json
```

### Model Type Detection

Automatically determined from `sae_type` in config:
- Contains "transcoder" or "clt" → `transcoder/`
- Otherwise → `sae/`

### Finding Checkpoints

```bash
# List all GPT-2 transcoder checkpoints
ls checkpoints/transcoder/gpt2-small/

# Find most recent
ls -t checkpoints/transcoder/gpt2-small/ | head -1

# Find all checkpoints at step 5000
find checkpoints -name "*_5000"
```

## Configuration

### Default Config (`config/default.json`)

Contains default training parameters. Can be loaded and customized:

```python
from utils.config import get_default_cfg

cfg = get_default_cfg()
cfg["model_name"] = "gemma-2-2b"
cfg["dict_size"] = 36864
```

### Model-Specific Configs

Automatically detected in `utils/config.py`:

```python
from utils.config import get_model_config

config = get_model_config("gemma-2-2b")
# Returns: {"act_size": 2304, "n_layers": 26}
```

## W&B Integration

### Project Names

Training scripts use organized W&B projects:
- SAEs: `"sparse_autoencoders"`
- Transcoders: `"gpt2-matryoshka-transcoder"`, `"gemma-matryoshka-transcoder"`

### Artifacts

Checkpoints are saved as W&B artifacts:
- Automatic upload on checkpoint save
- Linked to training run
- Can be downloaded from W&B dashboard

## Common Tasks

### 1. Train a GPT-2 Transcoder

```bash
python src/scripts/train_gpt2_simple.py
```

### 2. Evaluate Feature Quality

```bash
python src/scripts/run_evaluation.py
```

### 3. Compare Multiple Checkpoints

```bash
# Evaluate all checkpoints
for ckpt in checkpoints/transcoder/gpt2-small/*/; do
    python src/scripts/run_evaluation.py "$ckpt"
done
```

### 4. Load and Use a Trained Transcoder

```python
from src.models.sae import MatryoshkaTranscoder
import torch

# Load checkpoint
transcoder = MatryoshkaTranscoder(cfg)
transcoder.load_state_dict(torch.load("checkpoints/.../sae.pt"))
transcoder.eval()

# Encode source activations
sparse_features = transcoder.encode(source_activations)

# Decode to target
target_reconstruction = transcoder.decode(sparse_features)
```

### 5. Add a New Training Script

1. Create file in `src/scripts/`
2. Add path setup:
   ```python
   import sys
   import os
   sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
   ```
3. Import from `models`, `training`, `utils`
4. Use training functions from `training.training`

## Best Practices

1. **Run scripts from project root**: `python src/scripts/train_gpt2_simple.py`
2. **Use organized checkpoints**: Let code handle checkpoint organization
3. **Don't move checkpoints manually**: Breaks W&B tracking
4. **Evaluate after training**: Run `run_evaluation.py` to assess quality
5. **Use W&B**: All training scripts have W&B integration

## Migration from Old Structure

If you have old files in the root directory, you can organize them:

```bash
# Old structure (flat):
# - sae.py
# - config.py
# - training.py
# - checkpoints/model_name_step/

# New structure (organized):
# - src/models/sae.py
# - src/utils/config.py
# - src/training/training.py
# - checkpoints/transcoder/gpt2-small/model_name_step/

# Move files
mkdir -p src/models src/scripts src/training src/utils
mv sae.py activation_store.py transcoder_activation_store.py src/models/
mv training.py src/training/
mv config.py logs.py utils.py src/utils/
mv train_*.py main.py evaluate_features.py run_evaluation.py src/scripts/

# Organize checkpoints
mkdir -p checkpoints/sae checkpoints/transcoder
# Move manually based on type
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'models'`:
```python
# Add this at the top of your script
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
```

### Checkpoint Not Found

Make sure you're using the full organized path:
```
checkpoints/{model_type}/{base_model}/{checkpoint_name}/
```

### Wrong Working Directory

Always run scripts from project root:
```bash
cd /path/to/matryoshka_sae
python src/scripts/train_gpt2_simple.py  # ✅ Correct
cd src/scripts && python train_gpt2_simple.py  # ❌ Wrong
```

## Related Documentation

- **Main README**: `../README.md` - Project overview
- **Feature Evaluation**: `FEATURE_EVALUATION_GUIDE.md` - Evaluation metrics
- **FVU & Gemma**: `FVU_AND_GEMMA_GUIDE.md` - FVU metrics and Gemma support
- **Transcoders**: `MATRYOSHKA_TRANSCODER.md` - Transcoder architecture
- **Checkpoints**: `../checkpoints/README.md` - Checkpoint organization

## Quick Reference

| Task | Command |
|------|---------|
| Train GPT-2 transcoder | `python src/scripts/train_gpt2_simple.py` |
| Train Gemma-2 | `python src/scripts/train_gemma_example.py` |
| Evaluate model | `python src/scripts/run_evaluation.py` |
| Train SAE | `python src/scripts/main.py` |
| Find checkpoints | `ls checkpoints/transcoder/gpt2-small/` |
| Load checkpoint | See "Loading a Checkpoint" above |

---

For questions or issues, refer to the main README or documentation in `docs/`.

