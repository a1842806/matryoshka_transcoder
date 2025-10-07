"""
Train GPT-2 Matryoshka Transcoder on Layer 12 with OpenWebText.

This script trains a transcoder that learns cross-layer feature transformations
on layer 12 of GPT-2, using the OpenWebText dataset for 10,000 steps.

Usage:
    python train_gpt2_layer12_transcoder.py
"""

import torch
from transformer_lens import HookedTransformer
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.sae import MatryoshkaTranscoder
from models.transcoder_activation_store import TranscoderActivationsStore, create_transcoder_config
from training.training import train_transcoder
from utils.config import get_default_cfg, post_init_cfg


def main():
    """Train GPT-2 Matryoshka Transcoder on Layer 12 with OpenWebText."""
    
    print("=" * 80)
    print("GPT-2 Layer 12 Matryoshka Transcoder Training")
    print("Dataset: OpenWebText | Steps: 10,000 | Layer: 12")
    print("=" * 80)
    
    # Base configuration
    cfg = get_default_cfg()
    
    # Model settings
    cfg["model_name"] = "gpt2-small"
    cfg["dataset_path"] = "openwebtext"  # OpenWebText dataset
    
    # Training settings for 10,000 steps
    cfg["num_tokens"] = int(1.28e6)  # ~10,000 steps with batch_size=128
    cfg["model_batch_size"] = 8     # Smaller for stability
    cfg["batch_size"] = 128         # Adjusted for 10k steps
    cfg["seq_len"] = 128
    cfg["lr"] = 3e-4
    cfg["model_dtype"] = torch.float32
    cfg["dtype"] = torch.float32
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Matryoshka Transcoder architecture
    cfg["sae_type"] = "matryoshka-transcoder"
    cfg["dict_size"] = 12288  # 16x expansion (768 * 16)
    cfg["group_sizes"] = [768, 1536, 3072, 6912]  # Growing groups (1x, 2x, 4x, 9x)
    cfg["top_k"] = 64  # Number of active features per batch
    cfg["l1_coeff"] = 0.0  # Using TopK, not L1
    cfg["aux_penalty"] = 1/32
    cfg["n_batches_to_dead"] = 20
    cfg["top_k_aux"] = 512
    
    # Configure cross-layer mapping for layer 11 (last layer of GPT-2 Small)
    # Learn how layer 11's MLP transforms features
    cfg = create_transcoder_config(
        cfg,
        source_layer=11,     # Layer 11 (last layer of GPT-2 Small)
        target_layer=11,     # Same layer
        source_site="resid_mid",  # After attention, before MLP
        target_site="mlp_out"     # Output of MLP
    )
    
    # W&B settings
    cfg["wandb_project"] = "gpt2-layer12-matryoshka-transcoder"
    cfg["checkpoint_freq"] = 1000   # Save every 1000 steps
    cfg["perf_log_freq"] = 500      # Log performance every 500 steps
    
    # Print configuration
    print("\n" + "=" * 80)
    print("Configuration:")
    print("=" * 80)
    print(f"Model: {cfg['model_name']}")
    print(f"Layer: 12")
    print(f"Dataset: {cfg['dataset_path']}")
    print(f"Device: {cfg['device']}")
    print(f"Source: {cfg['source_hook_point']}")
    print(f"Target: {cfg['target_hook_point']}")
    print(f"Dictionary size: {cfg['dict_size']:,}")
    print(f"Group sizes: {cfg['group_sizes']}")
    print(f"Top-K: {cfg['top_k']}")
    print(f"Training tokens: {cfg['num_tokens']:,.0f}")
    print(f"Batch size: {cfg['batch_size']}")
    print(f"Learning rate: {cfg['lr']}")
    print(f"Steps: ~{cfg['num_tokens'] // cfg['batch_size']:,}")
    print(f"W&B project: {cfg['wandb_project']}")
    print(f"Run name: {cfg['name']}")
    print("=" * 80)
    
    # Load model
    print(f"\n📥 Loading {cfg['model_name']}...")
    model = HookedTransformer.from_pretrained_no_processing(
        cfg["model_name"]
    ).to(cfg["model_dtype"]).to(cfg["device"])
    print(f"✓ Model loaded successfully")
    print(f"  - Activation size: {model.cfg.d_model}")
    print(f"  - Number of layers: {model.cfg.n_layers}")
    print(f"  - Vocabulary size: {model.cfg.d_vocab}")
    print(f"  - Layer 12 exists: {12 < model.cfg.n_layers}")
    
    # Create transcoder activation store
    print(f"\n📊 Creating transcoder activation store...")
    activation_store = TranscoderActivationsStore(model, cfg)
    print(f"✓ Activation store created")
    print(f"  - Source hook: {cfg['source_hook_point']}")
    print(f"  - Target hook: {cfg['target_hook_point']}")
    print(f"  - Context size: {activation_store.context_size}")
    
    # Create Matryoshka Transcoder
    print(f"\n🏗️  Initializing Matryoshka Transcoder...")
    transcoder = MatryoshkaTranscoder(cfg)
    
    # Count parameters
    total_params = sum(p.numel() for p in transcoder.parameters())
    trainable_params = sum(p.numel() for p in transcoder.parameters() if p.requires_grad)
    
    print(f"✓ Transcoder initialized")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Groups: {len(cfg['group_sizes'])}")
    print(f"  - Group sizes: {cfg['group_sizes']}")
    
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
        print("  4. Use the transcoder for cross-layer analysis")
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
    
    print()
    main()
