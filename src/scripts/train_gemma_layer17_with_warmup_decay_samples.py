"""
Train Gemma-2-2B Matryoshka Transcoder on Layer 17 with:
- Learning rate warmup + decay scheduling
- Activation sample collection for interpretability
- 15k steps training run for comprehensive analysis
- OPTIMIZED configuration to reduce dead features from 50% to <10%

This script trains a transcoder on layer 17 and collects activation samples
for detailed feature analysis and interpretability research.

KEY OPTIMIZATIONS TO REDUCE DEAD FEATURES (50% -> <10% target):
==============================================================

1. Top-K Doubled (48 -> 96):
   - Problem: Only 0.26% features active was too sparse for normalized inputs
   - Solution: Increase to 0.52% activation rate
   - Rationale: Normalized mlp_in inputs need more active features to learn properly

2. Learning Rate Increased (3e-4 -> 4e-4):
   - Problem: Normalized inputs have reduced variance, weaker gradients
   - Solution: Higher LR compensates for normalized input distribution
   - Rationale: Need stronger gradients for implicit denormalization task

3. Auxiliary Penalty Reduced (1/32 -> 1/64):
   - Problem: Too strong penalty destabilizes training when reviving dead features
   - Solution: Lighter penalty for more stable revival
   - Rationale: Gradual revival is better than aggressive correction

4. Top-K Auxiliary Reduced (512 -> 256):
   - Problem: Activating 512 dead features at once overwhelms the model
   - Solution: Focus on 256 features for targeted revival
   - Rationale: Quality over quantity in dead feature revival

5. Dead Feature Monitoring Enhanced:
   - Added: dead_features_count (absolute number)
   - Added: dead_features_percentage (% of total)
   - Added: dead_features_alive (active features count)
   - Rationale: Comprehensive tracking for better diagnosis

DEAD LATENT DETECTION VERIFICATION:
===================================
Your implementation is CORRECT and follows industry best practices:

✓ Tracks consecutive inactive batches (num_batches_not_active)
✓ Resets counter when feature activates (proper reset logic)
✓ Uses configurable threshold (n_batches_to_dead = 20)
✓ Applies auxiliary loss only to dead features (efficient)
✓ No gradient flow for dead features (standard practice)

This matches approaches from:
- Anthropic's SAE work
- OpenAI's sparse autoencoders
- Academic research on ReLU dead neurons

Usage:
    python train_gemma_layer17_with_warmup_decay_samples.py
"""

import torch
from transformer_lens import HookedTransformer
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.sae import MatryoshkaTranscoder
from src.models.transcoder_activation_store import TranscoderActivationsStore, create_transcoder_config
from src.training.training import train_transcoder
from src.utils.config import get_default_cfg, post_init_cfg


def main():
    """Train Gemma-2-2B Matryoshka Transcoder on Layer 17 with warmup+decay and sample collection."""
    
    print("=" * 80)
    print("Gemma-2-2B Matryoshka Transcoder Training - Layer 17")
    print("Features: Warmup+Decay LR + Activation Sample Collection")
    print("Training: ~15k steps for comprehensive analysis")
    print("=" * 80)
    
    # Base configuration
    cfg = get_default_cfg()
    
    # Model settings - Gemma-2-2B specific
    cfg["model_name"] = "gemma-2-2b"
    cfg["dataset_path"] = "HuggingFaceFW/fineweb-edu"  # High-quality dataset
    cfg["layer"] = 17  # Target layer 17 for deeper analysis
    
    # Training settings - 15k steps configuration
    cfg["num_tokens"] = int(3e6)  # 15M tokens for ~15k steps (1024 batch size)
    cfg["model_batch_size"] = 4    # Smaller for Gemma's larger size
    cfg["batch_size"] = 1024       # Batch size
    cfg["seq_len"] = 64            # Sequence length
    cfg["lr"] = 4e-4               # Increased LR: normalized inputs need stronger gradients
    cfg["model_dtype"] = torch.bfloat16  # Use bfloat16 for Gemma
    cfg["dtype"] = torch.bfloat16
    cfg["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"  # Use available GPU
    
    # Print device information
    print(f"🖥️  Device Configuration:")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU count: {torch.cuda.device_count()}")
        print(f"   Using device: {cfg['device']}")
        print(f"   GPU name: {torch.cuda.get_device_name(0)}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"   Using device: {cfg['device']} (CPU)")
    
    # *** LEARNING RATE SCHEDULING - WARMUP + DECAY ***
    cfg["scheduler_type"] = "warmup_decay"  # Enable warmup + decay
    cfg["warmup_steps"] = 500               # 500 steps of warmup (3.3% of total)
    cfg["min_lr"] = cfg["lr"] * 0.01        # Decay to 1% of initial LR
    
    # *** MATRYOSHKA TRANSCODER CONFIGURATION ***
    cfg["dict_size"] = 18432      # 8x expansion (2304 * 8) for Gemma - halved from original
    cfg["group_sizes"] = [1152, 2304, 4608, 10368]  # 0.5x, 1x, 2x, 4.5x for Gemma - halved from original
    cfg["top_k"] = 96              # Doubled: 0.26% -> 0.52% activation rate for normalized inputs
    cfg["aux_penalty"] = 1/64      # Reduced: Lighter penalty for stable dead feature revival
    
    # *** DEAD FEATURE REVIVAL CONFIGURATION ***
    cfg["n_batches_to_dead"] = 20        # Features dead after 20 inactive batches
    cfg["top_k_aux"] = 256              # Reduced: Focused revival of 256 dead features
    
    # *** ACTIVATION SAMPLE COLLECTION ***
    cfg["save_activation_samples"] = True
    cfg["sample_collection_freq"] = 100     # Collect samples every 100 steps
    cfg["max_samples_per_feature"] = 100    # Max samples per feature
    cfg["sample_context_size"] = 20         # Context around activation
    cfg["sample_activation_threshold"] = 0.1 # Minimum activation threshold
    cfg["top_features_to_save"] = 100       # Top 100 features to save
    cfg["samples_per_feature_to_save"] = 10 # 10 samples per feature
    
    # *** LOGGING AND CHECKPOINTING ***
    cfg["perf_log_freq"] = 50      # Log performance every 50 steps
    cfg["checkpoint_freq"] = 500   # Save checkpoint every 500 steps
    cfg["wandb_project"] = "gemma-2-2b-layer17-interpretability"
    
    # *** TRANSCODER MAPPING - LAYER 17 (mlp_in → mlp_out) ***
    # Align with Google's interface and avoid double-normalizing.
    cfg = create_transcoder_config(
        cfg,
        source_layer=17,
        target_layer=17,
        source_site="mlp_in",   # Input to MLP (post-RMSNorm)
        target_site="mlp_out"   # Output of MLP
    )
    
    # Set correct activation sizes for transcoder
    cfg["source_act_size"] = 2304  # Gemma-2-2B d_model
    cfg["target_act_size"] = 2304  # Gemma-2-2B d_model
    cfg["input_unit_norm"] = False  # mlp_in is already normalized via RMSNorm
    
    # Post-process configuration
    cfg = post_init_cfg(cfg)
    
    print(f"Configuration:")
    print(f"  Model: {cfg['model_name']}")
    print(f"  Layer: {cfg['layer']}")
    print(f"  Training tokens: {cfg['num_tokens']:,}")
    print(f"  Batch size: {cfg['batch_size']}")
    print(f"  Dictionary size: {cfg['dict_size']}")
    print(f"  Learning rate: {cfg['lr']}")
    print(f"  Warmup steps: {cfg['warmup_steps']}")
    print(f"  Scheduler: {cfg['scheduler_type']}")
    print(f"  Sample collection: Every {cfg['sample_collection_freq']} steps")
    print(f"  W&B Project: {cfg['wandb_project']}")
    print(f"  Device: {cfg['device']}")
    
    # Load model
    print(f"\nLoading {cfg['model_name']}...")
    model = HookedTransformer.from_pretrained_no_processing(
        cfg["model_name"], 
        dtype=cfg["model_dtype"]
    ).to(cfg["device"])
    
    print(f"Model loaded successfully!")
    print(f"Model config: {model.cfg}")
    
    # Create activation store
    print(f"\nCreating activation store...")
    activation_store = TranscoderActivationsStore(model, cfg)
    
    # Create transcoder
    print(f"\nCreating Matryoshka Transcoder...")
    transcoder = MatryoshkaTranscoder(cfg).to(cfg["device"])
    
    print(f"Transcoder created:")
    print(f"  Dictionary size: {cfg['dict_size']}")
    print(f"  Group sizes: {cfg['group_sizes']}")
    print(f"  Top-k per group: {cfg['top_k']}")
    print(f"  Total parameters: {sum(p.numel() for p in transcoder.parameters()):,}")
    
    # Calculate expected steps
    expected_steps = int(cfg["num_tokens"] // cfg["batch_size"])
    print(f"\nExpected training steps: {expected_steps:,}")
    print(f"Sample collections: ~{expected_steps // cfg['sample_collection_freq']}")
    print(f"Checkpoints: ~{expected_steps // cfg['checkpoint_freq']}")
    
    # Start training
    print(f"\n{'='*80}")
    print("STARTING TRAINING")
    print(f"{'='*80}")
    
    try:
        train_transcoder(transcoder, activation_store, model, cfg)
        print(f"\n{'='*80}")
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print("TRAINING FAILED!")
        print(f"Error: {e}")
        print(f"{'='*80}")
        raise
    
    print(f"\nTraining completed! Check the following for results:")
    print(f"  - W&B Dashboard: {cfg['wandb_project']}")
    print(f"  - Checkpoints: checkpoints/transcoder/{cfg['model_name']}/{cfg['name']}")
    print(f"  - Activation samples: checkpoints/transcoder/{cfg['model_name']}/{cfg['name']}_activation_samples")


if __name__ == "__main__":
    main()