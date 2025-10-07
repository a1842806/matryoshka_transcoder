"""
Train Gemma-2-2B Matryoshka Transcoder on Layer 12 with Learning Rate Warmup.

This script trains a Matryoshka transcoder with linear warmup scheduling.
Designed to run on GPU 0 for comparison with standard training.

Usage:
    CUDA_VISIBLE_DEVICES=0 python train_gemma_layer12_transcoder_warmup.py
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
    """Train Gemma-2-2B Matryoshka Transcoder on Layer 12 with Warmup."""
    
    print("=" * 80)
    print("Gemma-2-2B Matryoshka Transcoder Training - Layer 12 (WARMUP ONLY)")
    print("=" * 80)
    
    # Base configuration
    cfg = get_default_cfg()
    
    # Model settings - Gemma-2-2B specific
    cfg["model_name"] = "gemma-2-2b"
    cfg["dataset_path"] = "HuggingFaceFW/fineweb-edu"
    cfg["layer"] = 12
    
    # Training settings - optimized for 10k steps
    cfg["num_tokens"] = int(1e7)  # 10M tokens for 10k steps (1000 tokens per step)
    cfg["model_batch_size"] = 4
    cfg["batch_size"] = 1024
    cfg["seq_len"] = 64
    cfg["lr"] = 3e-4              # Base learning rate
    cfg["model_dtype"] = torch.bfloat16
    cfg["dtype"] = torch.bfloat16
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Learning Rate Scheduling - WARMUP ONLY
    cfg["scheduler_type"] = "warmup_only"
    cfg["warmup_steps"] = 1000    # 10% of total steps for warmup
    cfg["warmup_lr"] = 1e-6       # Start with very low learning rate
    
    # Matryoshka Transcoder architecture
    cfg["sae_type"] = "matryoshka-transcoder"
    cfg["dict_size"] = 36864
    cfg["group_sizes"] = [2304, 4608, 9216, 20736]
    cfg["top_k"] = 96
    cfg["l1_coeff"] = 0.0
    cfg["aux_penalty"] = 1/32
    cfg["n_batches_to_dead"] = 20
    cfg["top_k_aux"] = 512
    
    # Configure cross-layer mapping: resid_mid -> mlp_out on layer 12
    cfg = create_transcoder_config(
        cfg,
        source_layer=12,
        target_layer=12,
        source_site="resid_mid",
        target_site="mlp_out"
    )
    
    # W&B settings - distinct project for comparison
    cfg["wandb_project"] = "gemma-2-2b-layer12-transcoder-warmup"
    cfg["checkpoint_freq"] = 1000
    cfg["perf_log_freq"] = 500
    
    # Print configuration
    print("\n" + "=" * 80)
    print("Configuration:")
    print("=" * 80)
    print(f"Model: {cfg['model_name']}")
    print(f"Layer: {cfg['layer']}")
    print(f"Device: {cfg['device']}")
    print(f"Dataset: {cfg['dataset_path']}")
    print(f"Scheduler: {cfg['scheduler_type']}")
    print(f"Warmup steps: {cfg['warmup_steps']}")
    print(f"Base LR: {cfg['lr']}")
    print(f"Warmup LR: {cfg['warmup_lr']}")
    print(f"Source: {cfg['source_hook_point']}")
    print(f"Target: {cfg['target_hook_point']}")
    print(f"Dictionary size: {cfg['dict_size']:,}")
    print(f"Group sizes: {cfg['group_sizes']}")
    print(f"Top-K: {cfg['top_k']}")
    print(f"Training tokens: {cfg['num_tokens']:,.0f}")
    print(f"Expected steps: ~{cfg['num_tokens'] // cfg['batch_size']:,}")
    print(f"Batch size: {cfg['batch_size']}")
    print(f"W&B project: {cfg['wandb_project']}")
    print(f"Run name: {cfg['name']}")
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
    print("🚀 Starting training with WARMUP scheduling...")
    print("=" * 80)
    print("Learning Rate Schedule:")
    print(f"  - Steps 0-{cfg['warmup_steps']}: Linear warmup from {cfg['warmup_lr']:.1e} to {cfg['lr']:.1e}")
    print(f"  - Steps {cfg['warmup_steps']}+: Fixed learning rate {cfg['lr']:.1e}")
    print("")
    print("Metrics tracked:")
    print("  - Loss: Total training loss")
    print("  - L2: Reconstruction error")
    print("  - L0: Number of active features")
    print("  - FVU: Fraction of Variance Unexplained")
    print("  - Dead: Number of inactive features")
    print("  - LR: Current learning rate")
    print("\nW&B dashboard will show:")
    print("  - Training metrics over time")
    print("  - Learning rate schedule")
    print("  - Performance comparison with standard run")
    print("=" * 80)
    print()
    
    try:
        train_transcoder(transcoder, activation_store, model, cfg)
        print("\n" + "=" * 80)
        print("✅ Training completed successfully!")
        print("=" * 80)
        print(f"\nCheckpoints saved to: checkpoints/{cfg['name']}_*/")
        print(f"W&B dashboard: https://wandb.ai/{cfg.get('wandb_entity', 'your-entity')}/{cfg['wandb_project']}")
        print("\nCompare with standard run on W&B dashboard!")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Check dependencies
    print("Checking dependencies...")
    
    try:
        import wandb
        print("✓ Weights & Biases is installed")
    except ImportError:
        print("❌ Weights & Biases not found!")
        print("Install it with: pip install wandb")
        exit(1)
    
    try:
        from transformer_lens import HookedTransformer
        print("✓ TransformerLens is installed")
    except ImportError:
        print("❌ TransformerLens not found!")
        exit(1)
    
    try:
        import torch
        print(f"✓ PyTorch is installed (version: {torch.__version__})")
        if torch.cuda.is_available():
            print(f"✓ CUDA is available (device count: {torch.cuda.device_count()})")
            print(f"✓ Using device: cuda")
        else:
            print("⚠️  CUDA not available, will use CPU (slower)")
    except ImportError:
        print("❌ PyTorch not found!")
        exit(1)
    
    print()
    main()
