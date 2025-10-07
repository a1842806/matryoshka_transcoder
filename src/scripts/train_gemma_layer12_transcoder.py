"""
Train Gemma-2-2B Matryoshka Transcoder on Layer 12 with OpenWebText.

This script trains a Matryoshka transcoder that learns cross-layer feature
transformations in Gemma-2-2B, specifically focusing on layer 12 with 10k steps.

Usage:
    python train_gemma_layer12_transcoder.py
"""

import torch
from transformer_lens import HookedTransformer
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.sae import MatryoshkaTranscoder
from src.models.transcoder_activation_store import TranscoderActivationsStore, create_transcoder_config
from src.training.training import train_transcoder
from src.utils.config import get_default_cfg, post_init_cfg


def main():
    """Train Gemma-2-2B Matryoshka Transcoder on Layer 12."""
    
    print("=" * 80)
    print("Gemma-2-2B Matryoshka Transcoder Training - Layer 12")
    print("=" * 80)
    
    # Base configuration
    cfg = get_default_cfg()
    
    # Model settings - Gemma-2-2B specific
    cfg["model_name"] = "gemma-2-2b"
    cfg["dataset_path"] = "HuggingFaceFW/fineweb-edu"  # High-quality alternative to OpenWebText
    cfg["layer"] = 12  # Target layer 12 as requested
    
    # Training settings - optimized for 10k steps
    cfg["num_tokens"] = int(1e7)  # 10M tokens for 10k steps (1000 tokens per step)
    cfg["model_batch_size"] = 4   # Smaller for Gemma's larger size
    cfg["batch_size"] = 1024      # Reasonable batch size
    cfg["seq_len"] = 64           # Sequence length
    cfg["lr"] = 3e-4              # Learning rate
    cfg["model_dtype"] = torch.bfloat16  # Use bfloat16 for Gemma
    cfg["dtype"] = torch.bfloat16
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Matryoshka Transcoder architecture - optimized for Gemma-2-2B
    cfg["sae_type"] = "matryoshka-transcoder"
    cfg["dict_size"] = 36864      # 16x expansion (2304 * 16) for Gemma
    cfg["group_sizes"] = [2304, 4608, 9216, 20736]  # 1x, 2x, 4x, 9x for Gemma
    cfg["top_k"] = 96             # Higher for larger model
    cfg["l1_coeff"] = 0.0         # Using TopK, not L1
    cfg["aux_penalty"] = 1/32
    cfg["n_batches_to_dead"] = 20
    cfg["top_k_aux"] = 512
    
    # Configure cross-layer mapping: resid_mid -> mlp_out on layer 12
    # This learns how the MLP transforms features at layer 12
    cfg = create_transcoder_config(
        cfg,
        source_layer=12,         # Layer 12 as requested
        target_layer=12,         # Same layer
        source_site="resid_mid", # After attention, before MLP
        target_site="mlp_out"    # Output of MLP
    )
    
    # W&B settings
    cfg["wandb_project"] = "gemma-2-2b-layer12-transcoder"
    cfg["checkpoint_freq"] = 1000   # Save every 1000 steps
    cfg["perf_log_freq"] = 500      # Log performance every 500 steps
    
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
    print(f"Learning rate: {cfg['lr']}")
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
    print("🚀 Starting training...")
    print("=" * 80)
    print("Metrics tracked:")
    print("  - Loss: Total training loss")
    print("  - L2: Reconstruction error")
    print("  - L0: Number of active features")
    print("  - FVU: Fraction of Variance Unexplained (0=perfect, <0.1=good)")
    print("  - Dead: Number of inactive features")
    print("\nW&B dashboard will show:")
    print("  - Training metrics over time")
    print("  - Performance metrics (CE degradation, recovery)")
    print("  - Group-specific FVU (fvu_min, fvu_max, fvu_mean)")
    print("  - Model checkpoints as artifacts")
    print("=" * 80)
    print()
    
    try:
        train_transcoder(transcoder, activation_store, model, cfg)
        print("\n" + "=" * 80)
        print("✅ Training completed successfully!")
        print("=" * 80)
        print(f"\nCheckpoints saved to: checkpoints/{cfg['name']}_*/")
        print(f"W&B dashboard: https://wandb.ai/{cfg.get('wandb_entity', 'your-entity')}/{cfg['wandb_project']}")
        print("\nYou can now:")
        print("  1. View training metrics on W&B dashboard")
        print("  2. Load the checkpoint for inference")
        print("  3. Analyze learned features")
        print("  4. Use the transcoder for cross-layer feature mapping")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print("Partial checkpoints may be saved in checkpoints/")
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
