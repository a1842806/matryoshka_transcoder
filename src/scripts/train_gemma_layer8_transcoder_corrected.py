"""
Train Gemma-2-2B Matryoshka Transcoder on Layer 8 with CORRECT hooks.

This script addresses the reviewer's feedback by using the correct hooks:
- Source: ln2.hook_normalized (post-RMSNorm) + RMSNorm scaling
- Target: hook_mlp_out (MLP output contribution)

This captures the TRUE MLP transformation, not just a pre-norm transformation.

Usage:
    python train_gemma_layer8_transcoder_corrected.py
"""

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.sae import MatryoshkaTranscoder
from src.models.transcoder_activation_store import TranscoderActivationsStore
from src.training.training import train_transcoder
from src.utils.config import get_default_cfg, create_gemma_mlp_transcoder_config


def main():
    """Train Gemma-2-2B Matryoshka Transcoder on Layer 8 with CORRECT hooks."""
    
    print("=" * 80)
    print("Gemma-2-2B Matryoshka Transcoder Training - Layer 8 (CORRECTED)")
    print("=" * 80)
    print("✅ Using ln2.hook_normalized + RMSNorm scaling (actual MLP input)")
    print("✅ Target: hook_mlp_out (MLP output contribution)")
    print("✅ This captures the TRUE MLP transformation")
    print("=" * 80)
    
    # Base configuration
    cfg = get_default_cfg()
    
    # Model settings - Gemma-2-2B specific
    cfg["model_name"] = "google/gemma-2-2b"
    cfg["dataset_path"] = "c4"  # Use C4 dataset (more reliable)
    cfg["layer"] = 8  # Target layer 8 as requested
    
    # Training settings - optimized for 10k steps
    cfg["num_tokens"] = int(1e7)  # 10M tokens for 10k steps (1000 tokens per step)
    cfg["model_batch_size"] = 4   # Smaller for Gemma's larger size
    cfg["batch_size"] = 1024      # Reasonable batch size
    cfg["seq_len"] = 64           # Sequence length
    cfg["lr"] = 1e-4              # Learning rate (reduced for stability)
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
    
    # Create CORRECT transcoder configuration for layer 8
    print(f"\n🔧 Creating CORRECT transcoder configuration for layer 8...")
    print(f"   Layer: 8")
    print(f"   Source: ln2.hook_normalized (post-RMSNorm) + scaling")
    print(f"   Target: hook_mlp_out (MLP output)")
    
    transcoder_cfg = create_gemma_mlp_transcoder_config(
        cfg, 
        layer=8, 
        use_ln2_normalized=True  # CORRECT approach
    )
    
    # W&B settings
    transcoder_cfg["wandb_project"] = "gemma-2-2b-layer8-transcoder-corrected"
    transcoder_cfg["checkpoint_freq"] = 1000   # Save every 1000 steps
    transcoder_cfg["perf_log_freq"] = 500      # Log performance every 500 steps
    
    # Print configuration
    print("\n" + "=" * 80)
    print("Configuration:")
    print("=" * 80)
    print(f"Model: {transcoder_cfg['model_name']}")
    print(f"Layer: {transcoder_cfg['layer']}")
    print(f"Device: {transcoder_cfg['device']}")
    print(f"Dataset: {transcoder_cfg['dataset_path']}")
    print(f"Source: {transcoder_cfg['source_hook_point']}")
    print(f"Target: {transcoder_cfg['target_hook_point']}")
    print(f"RMSNorm scaling: {transcoder_cfg.get('apply_rmsnorm_scaling', False)}")
    print(f"Dictionary size: {transcoder_cfg['dict_size']:,}")
    print(f"Group sizes: {transcoder_cfg['group_sizes']}")
    print(f"Top-K: {transcoder_cfg['top_k']}")
    print(f"Training tokens: {transcoder_cfg['num_tokens']:,.0f}")
    print(f"Expected steps: ~{transcoder_cfg['num_tokens'] // transcoder_cfg['batch_size']:,}")
    print(f"Batch size: {transcoder_cfg['batch_size']}")
    print(f"Learning rate: {transcoder_cfg['lr']}")
    print(f"W&B project: {transcoder_cfg['wandb_project']}")
    print(f"Run name: {transcoder_cfg['name']}")
    print("=" * 80)
    
    # Load model with proper configuration
    print(f"\n📥 Loading {transcoder_cfg['model_name']}...")
    try:
        # Load model with standard configuration
        model = HookedTransformer.from_pretrained(
            transcoder_cfg["model_name"],
            device=transcoder_cfg["device"]
        ).to(transcoder_cfg["model_dtype"])
        
        print(f"✓ Model loaded successfully")
        print(f"  - Activation size: {model.cfg.d_model}")
        print(f"  - Number of layers: {model.cfg.n_layers}")
        print(f"  - Vocabulary size: {model.cfg.d_vocab}")
        print(f"  - Model dtype: {model.cfg.dtype}")
        print(f"  - RMSNorm enabled: {hasattr(model.blocks[0].ln2, 'w')}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("Make sure you have access to Gemma-2-2B model")
        return
    
    # Create transcoder activation store
    print(f"\n📊 Creating transcoder activation store...")
    try:
        activation_store = TranscoderActivationsStore(model, transcoder_cfg)
        print(f"✓ Activation store created")
        print(f"  - Source hook: {transcoder_cfg['source_hook_point']}")
        print(f"  - Target hook: {transcoder_cfg['target_hook_point']}")
        print(f"  - RMSNorm scaling: {transcoder_cfg.get('apply_rmsnorm_scaling', False)}")
        print(f"  - Context size: {activation_store.context_size}")
        
    except Exception as e:
        print(f"❌ Failed to create activation store: {e}")
        print("This might be due to dataset loading issues.")
        return
    
    # Create Matryoshka Transcoder
    print(f"\n🏗️  Initializing Matryoshka Transcoder...")
    try:
        transcoder = MatryoshkaTranscoder(transcoder_cfg)
        
        # Count parameters
        total_params = sum(p.numel() for p in transcoder.parameters())
        trainable_params = sum(p.numel() for p in transcoder.parameters() if p.requires_grad)
        
        print(f"✓ Transcoder initialized")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - Groups: {len(transcoder_cfg['group_sizes'])}")
        print(f"  - Group sizes: {transcoder_cfg['group_sizes']}")
        print(f"  - Source act size: {transcoder.source_act_size}")
        print(f"  - Target act size: {transcoder.target_act_size}")
        
    except Exception as e:
        print(f"❌ Failed to initialize transcoder: {e}")
        return
    
    # Training
    print("\n" + "=" * 80)
    print("🚀 Starting training with CORRECT hooks...")
    print("=" * 80)
    print("This transcoder learns the TRUE MLP transformation:")
    print("  ln2_normalized * ln2.w → mlp_out")
    print("")
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
        train_transcoder(transcoder, activation_store, model, transcoder_cfg)
        print("\n" + "=" * 80)
        print("✅ Training completed successfully!")
        print("=" * 80)
        print(f"\nCheckpoints saved to: checkpoints/{transcoder_cfg['name']}_*/")
        print(f"W&B dashboard: https://wandb.ai/{transcoder_cfg.get('wandb_entity', 'your-entity')}/{transcoder_cfg['wandb_project']}")
        print("\nThis transcoder now captures the TRUE MLP transformation!")
        print("You can now:")
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
