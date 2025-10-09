"""
Train Gemma-2-2B Matryoshka Transcoder on Layer 8 with:
- Learning rate warmup + decay scheduling
- Activation sample collection for interpretability
- Demonstration that features store activation samples from training data

This script demonstrates the complete interpretability pipeline:
1. Train transcoder with proper learning rate scheduling
2. Collect activation samples during training
3. Show that features do store activation samples from training data
4. Enable analysis of learned features

Usage:
    python train_gemma_layer8_with_warmup_decay_samples.py
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
    """Train Gemma-2-2B Matryoshka Transcoder on Layer 8 with warmup+decay and sample collection."""
    
    print("=" * 80)
    print("Gemma-2-2B Matryoshka Transcoder Training - Layer 8")
    print("Features: Warmup+Decay LR + Activation Sample Collection")
    print("=" * 80)
    
    # Base configuration
    cfg = get_default_cfg()
    
    # Model settings - Gemma-2-2B specific
    cfg["model_name"] = "gemma-2-2b"
    cfg["dataset_path"] = "HuggingFaceFW/fineweb-edu"  # High-quality alternative to OpenWebText
    cfg["layer"] = 8  # Target layer 8 as requested
    
    # Training settings - optimized for interpretability research
    cfg["num_tokens"] = int(5e6)   # 5M tokens for faster demo
    cfg["model_batch_size"] = 4    # Smaller for Gemma's larger size
    cfg["batch_size"] = 1024       # Reasonable batch size
    cfg["seq_len"] = 64            # Sequence length
    cfg["lr"] = 3e-4               # Initial learning rate
    cfg["model_dtype"] = torch.bfloat16  # Use bfloat16 for Gemma
    cfg["dtype"] = torch.bfloat16
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    
    # *** LEARNING RATE SCHEDULING - WARMUP + DECAY ***
    cfg["scheduler_type"] = "warmup_decay"  # Enable warmup + decay
    cfg["warmup_steps"] = 200               # 200 steps of warmup
    cfg["min_lr"] = cfg["lr"] * 0.01        # Decay to 1% of initial LR
    
    # Matryoshka Transcoder architecture - optimized for Gemma-2-2B
    cfg["sae_type"] = "matryoshka-transcoder"
    cfg["dict_size"] = 18432      # 8x expansion (2304 * 8) for Gemma
    cfg["group_sizes"] = [2304, 4608, 9216, 2304]  # 1x, 2x, 4x, 1x for Gemma
    cfg["top_k"] = 48             # Moderate for layer 8
    cfg["l1_coeff"] = 0.0         # Using TopK, not L1
    cfg["aux_penalty"] = 1/32
    cfg["n_batches_to_dead"] = 20
    cfg["top_k_aux"] = 256
    
    # *** ACTIVATION SAMPLE COLLECTION - ANTHROPIC STYLE ***
    cfg["save_activation_samples"] = True        # Enable sample collection
    cfg["sample_collection_freq"] = 100          # Collect every 100 steps (more frequent for demo)
    cfg["max_samples_per_feature"] = 100         # Store top 100 samples per feature
    cfg["sample_context_size"] = 20              # 20 tokens of context
    cfg["sample_activation_threshold"] = 0.1     # Only activations > 0.1
    cfg["top_features_to_save"] = 100            # Save top 100 most active features
    cfg["samples_per_feature_to_save"] = 10      # Save 10 examples per feature
    
    # Configure cross-layer mapping: resid_mid -> mlp_out on layer 8
    # This learns how the MLP transforms features at layer 8
    cfg = create_transcoder_config(
        cfg,
        source_layer=8,          # Layer 8 as requested
        target_layer=8,          # Same layer
        source_site="resid_mid", # After attention, before MLP
        target_site="mlp_out"    # Output of MLP
    )
    
    # W&B settings
    cfg["wandb_project"] = "gemma-2-2b-layer8-interpretability"
    cfg["checkpoint_freq"] = 500    # Save every 500 steps
    cfg["perf_log_freq"] = 100      # Log performance every 100 steps
    
    # Print configuration
    print("\n" + "=" * 80)
    print("Configuration:")
    print("=" * 80)
    print(f"Model: {cfg['model_name']}")
    print(f"Layer: {cfg['layer']}")
    print(f"Device: {cfg['device']}")
    print(f"Dataset: {cfg['dataset_path']}")
    print(f"Source: {cfg['source_hook_point']}")
    print(f"Target: {cfg['target_hook_point']}")
    print(f"Dictionary size: {cfg['dict_size']:,}")
    print(f"Group sizes: {cfg['group_sizes']}")
    print(f"Top-K: {cfg['top_k']}")
    print(f"Training tokens: {cfg['num_tokens']:,.0f}")
    print(f"Expected steps: ~{cfg['num_tokens'] // cfg['batch_size']:,}")
    print(f"Batch size: {cfg['batch_size']}")
    print(f"Initial learning rate: {cfg['lr']}")
    print(f"Min learning rate: {cfg['min_lr']}")
    print(f"Warmup steps: {cfg['warmup_steps']}")
    print(f"Scheduler: {cfg['scheduler_type']}")
    print(f"W&B project: {cfg['wandb_project']}")
    print(f"Run name: {cfg['name']}")
    
    print("\n" + "=" * 80)
    print("Activation Sample Collection Settings:")
    print("=" * 80)
    print(f"Sample collection enabled: {cfg['save_activation_samples']}")
    print(f"Collection frequency: every {cfg['sample_collection_freq']} steps")
    print(f"Max samples per feature: {cfg['max_samples_per_feature']}")
    print(f"Context size: {cfg['sample_context_size']} tokens")
    print(f"Activation threshold: {cfg['sample_activation_threshold']}")
    print(f"Top features to save: {cfg['top_features_to_save']}")
    print(f"Samples per feature to save: {cfg['samples_per_feature_to_save']}")
    print("=" * 80)
    
    # Load model
    print(f"\n📥 Loading {cfg['model_name']}...")
    try:
        model = HookedTransformer.from_pretrained_no_processing(
            cfg["model_name"]
        ).to(cfg["model_dtype"]).to(cfg["device"])
        print(f"✓ Model loaded successfully")
        print(f"  - Activation size: {model.cfg.d_model}")
        print(f"  - Number of layers: {model.cfg.n_layers}")
        print(f"  - Vocabulary size: {model.cfg.d_vocab}")
        print(f"  - Model dtype: {model.cfg.dtype}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("Make sure you have access to Gemma-2-2B model")
        return
    
    # Create transcoder activation store
    print(f"\n📊 Creating transcoder activation store...")
    try:
        activation_store = TranscoderActivationsStore(model, cfg)
        print(f"✓ Activation store created")
        print(f"  - Source hook: {cfg['source_hook_point']}")
        print(f"  - Target hook: {cfg['target_hook_point']}")
        print(f"  - Context size: {activation_store.context_size}")
    except Exception as e:
        print(f"❌ Failed to create activation store: {e}")
        return
    
    # Create Matryoshka Transcoder
    print(f"\n🏗️  Initializing Matryoshka Transcoder...")
    try:
        transcoder = MatryoshkaTranscoder(cfg)
        
        # Count parameters
        total_params = sum(p.numel() for p in transcoder.parameters())
        trainable_params = sum(p.numel() for p in transcoder.parameters() if p.requires_grad)
        
        print(f"✓ Transcoder initialized")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - Groups: {len(cfg['group_sizes'])}")
        print(f"  - Group sizes: {cfg['group_sizes']}")
        print(f"  - Source act size: {transcoder.source_act_size}")
        print(f"  - Target act size: {transcoder.target_act_size}")
    except Exception as e:
        print(f"❌ Failed to initialize transcoder: {e}")
        return
    
    # Training
    print("\n" + "=" * 80)
    print("🚀 Starting training with interpretability features...")
    print("=" * 80)
    print("Training features:")
    print("  ✓ Learning rate warmup + decay scheduling")
    print("  ✓ Activation sample collection")
    print("  ✓ Comprehensive logging")
    print("  ✓ Automatic checkpointing")
    print()
    print("Metrics tracked:")
    print("  - Loss: Total training loss")
    print("  - L2: Reconstruction error")
    print("  - L0: Number of active features")
    print("  - FVU: Fraction of Variance Unexplained (0=perfect, <0.1=good)")
    print("  - Dead: Number of inactive features")
    print("  - Learning Rate: Current LR (with warmup/decay)")
    print()
    print("W&B dashboard will show:")
    print("  - Training metrics over time")
    print("  - Learning rate schedule")
    print("  - Performance metrics (CE degradation, recovery)")
    print("  - Group-specific FVU (fvu_min, fvu_max, fvu_mean)")
    print("  - Model checkpoints as artifacts")
    print()
    print("Activation samples will be saved to:")
    print(f"  - checkpoints/transcoder/{cfg['model_name']}/{cfg['name']}_activation_samples/")
    print("=" * 80)
    print()
    
    try:
        train_transcoder(transcoder, activation_store, model, cfg)
        print("\n" + "=" * 80)
        print("✅ Training completed successfully!")
        print("=" * 80)
        print(f"\nCheckpoints saved to: checkpoints/transcoder/{cfg['model_name']}/{cfg['name']}_*/")
        print(f"Activation samples saved to: checkpoints/transcoder/{cfg['model_name']}/{cfg['name']}_activation_samples/")
        print(f"W&B dashboard: https://wandb.ai/{cfg.get('wandb_entity', 'your-entity')}/{cfg['wandb_project']}")
        print("\nYou can now:")
        print("  1. View training metrics on W&B dashboard")
        print("  2. Load the checkpoint for inference")
        print("  3. Analyze learned features using activation samples")
        print("  4. Generate interpretability reports")
        print("  5. Use the transcoder for cross-layer feature mapping")
        print()
        print("To analyze activation samples:")
        print("  python src/utils/analyze_activation_samples.py checkpoints/transcoder/gemma-2-2b/*/activation_samples/")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print("Partial checkpoints and activation samples may be saved in checkpoints/")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Check dependencies
    print("Checking dependencies...")
    
    # Check if wandb is installed
    try:
        import wandb
        print("✓ Weights & Biases is installed")
    except ImportError:
        print("❌ Weights & Biases not found!")
        print("Install it with: pip install wandb")
        print("Then login with: wandb login")
        exit(1)
    
    # Check if transformer_lens is installed
    try:
        from transformer_lens import HookedTransformer
        print("✓ TransformerLens is installed")
    except ImportError:
        print("❌ TransformerLens not found!")
        print("Install it with: pip install transformer_lens")
        exit(1)
    
    # Check if torch is available
    try:
        import torch
        print(f"✓ PyTorch is installed (version: {torch.__version__})")
        if torch.cuda.is_available():
            print(f"✓ CUDA is available (device count: {torch.cuda.device_count()})")
        else:
            print("⚠️  CUDA not available, will use CPU (slower)")
    except ImportError:
        print("❌ PyTorch not found!")
        print("Install it with: pip install torch")
        exit(1)
    
    print()
    main()
