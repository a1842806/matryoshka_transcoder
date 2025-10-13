"""
Train Gemma-2-2B Matryoshka Transcoder on Layer 17 with:
- Learning rate warmup + decay scheduling
- Activation sample collection for interpretability
- 15k steps training run for comprehensive analysis

This script trains a transcoder on layer 17 and collects activation samples
for detailed feature analysis and interpretability research.

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
    cfg["num_tokens"] = int(15e6)  # 15M tokens for ~15k steps (1024 batch size)
    cfg["model_batch_size"] = 4    # Smaller for Gemma's larger size
    cfg["batch_size"] = 1024       # Batch size
    cfg["seq_len"] = 64            # Sequence length
    cfg["lr"] = 3e-4               # Initial learning rate
    cfg["model_dtype"] = torch.bfloat16  # Use bfloat16 for Gemma
    cfg["dtype"] = torch.bfloat16
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    
    # *** LEARNING RATE SCHEDULING - WARMUP + DECAY ***
    cfg["scheduler_type"] = "warmup_decay"  # Enable warmup + decay
    cfg["warmup_steps"] = 500               # 500 steps of warmup (3.3% of total)
    cfg["min_lr"] = cfg["lr"] * 0.01        # Decay to 1% of initial LR
    
    # *** MATRYOSHKA TRANSCODER CONFIGURATION ***
    cfg["dict_size"] = 4608        # 2x activation size for good reconstruction
    cfg["group_sizes"] = [1152, 2304, 1152]  # Matryoshka groups: 0.5x, 1x, 0.5x
    cfg["top_k"] = 48              # Active features per group
    cfg["aux_penalty"] = 1/32      # Auxiliary loss weight
    
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
    
    # *** TRANSCODER MAPPING - LAYER 17 ***
    # Source: resid_mid (after attention, before MLP)
    # Target: mlp_out (after MLP)
    cfg = create_transcoder_config(
        cfg,
        source_layer=17,
        target_layer=17,
        source_site="resid_mid",  # After attention, before MLP
        target_site="mlp_out"     # After MLP
    )
    
    # Set correct activation sizes for transcoder
    cfg["source_act_size"] = 2304  # Gemma-2-2B d_model
    cfg["target_act_size"] = 2304  # Gemma-2-2B d_model
    
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
