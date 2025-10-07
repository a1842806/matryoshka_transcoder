"""
Example script for training Matryoshka SAEs and Transcoders on Gemma-2-2B.

This demonstrates:
1. Using Gemma-2-2B with proper hook names
2. Setting appropriate activation sizes for Gemma
3. Training with FVU metrics
"""

import torch
import copy
from transformer_lens import HookedTransformer
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.sae import GlobalBatchTopKMatryoshkaSAE, BatchTopKSAE, MatryoshkaTranscoder
from models.activation_store import ActivationsStore
from models.transcoder_activation_store import TranscoderActivationsStore, create_transcoder_config
from training.training import train_sae, train_transcoder
from utils.config import get_default_cfg, post_init_cfg, get_model_config


def train_gemma_matryoshka_sae():
    """
    Train a Matryoshka SAE on Gemma-2-2B.
    
    Gemma-2-2B specifications:
    - act_size: 2304 (automatically detected)
    - n_layers: 26
    - Hook names: Uses standard TransformerLens format
    """
    # Base configuration
    cfg = get_default_cfg()
    
    # Gemma-2-2B specific settings
    cfg["model_name"] = "gemma-2-2b"
    cfg["layer"] = 8  # Middle layer
    cfg["site"] = "resid_pre"
    cfg["dataset_path"] = "HuggingFaceFW/fineweb-edu"  # High-quality dataset
    
    # Training settings
    cfg["num_tokens"] = 1e8  # 100M tokens
    cfg["model_batch_size"] = 4  # Smaller for Gemma's larger size
    cfg["batch_size"] = 2048
    cfg["seq_len"] = 64
    cfg["lr"] = 3e-4
    cfg["model_dtype"] = torch.bfloat16
    cfg["dtype"] = torch.bfloat16
    cfg["device"] = "cuda"
    
    # Matryoshka SAE settings
    cfg["sae_type"] = "global-matryoshka-topk"
    cfg["dict_size"] = 36864  # 16x expansion (2304 * 16)
    cfg["group_sizes"] = [2304, 4608, 9216, 20736]  # 1x, 2x, 4x, 9x
    cfg["top_k"] = 96  # Higher for larger model
    cfg["l1_coeff"] = 0.0
    cfg["aux_penalty"] = 1/32
    cfg["n_batches_to_dead"] = 20
    cfg["top_k_aux"] = 512
    
    # Logging
    cfg["wandb_project"] = "gemma-2-2b-matryoshka-sae"
    cfg["checkpoint_freq"] = 5000
    cfg["perf_log_freq"] = 1000
    
    # Post-process config (auto-detects act_size and sets hook_point)
    cfg = post_init_cfg(cfg)
    
    # Verify configuration
    model_config = get_model_config(cfg["model_name"])
    print(f"Model: {cfg['model_name']}")
    print(f"  Activation size: {cfg['act_size']}")
    print(f"  Hook point: {cfg['hook_point']}")
    print(f"  Dictionary size: {cfg['dict_size']}")
    print(f"  Group sizes: {cfg['group_sizes']}")
    print(f"  Top-K: {cfg['top_k']}")
    
    # Load model
    print(f"\nLoading {cfg['model_name']}...")
    model = HookedTransformer.from_pretrained_no_processing(
        cfg["model_name"]
    ).to(cfg["model_dtype"]).to(cfg["device"])
    
    # Create activation store
    print("Creating activation store...")
    activation_store = ActivationsStore(model, cfg)
    
    # Create Matryoshka SAE
    print("Initializing Matryoshka SAE...")
    sae = GlobalBatchTopKMatryoshkaSAE(cfg)
    
    # Train
    print("\nStarting training...")
    print("Metrics tracked: Loss, L2, L0, FVU (Fraction of Variance Unexplained)")
    train_sae(sae, activation_store, model, cfg)
    print("Training complete!")


def train_gemma_transcoder():
    """
    Train a Matryoshka Transcoder on Gemma-2-2B.
    
    This example learns how the MLP transforms features in Gemma-2-2B.
    """
    # Base configuration
    cfg = get_default_cfg()
    
    # Gemma-2-2B specific settings
    cfg["model_name"] = "gemma-2-2b"
    cfg["dataset_path"] = "HuggingFaceFW/fineweb-edu"
    cfg["num_tokens"] = 5e7  # 50M tokens
    cfg["model_batch_size"] = 4
    cfg["batch_size"] = 1024
    cfg["seq_len"] = 64
    cfg["lr"] = 3e-4
    cfg["model_dtype"] = torch.bfloat16
    cfg["dtype"] = torch.bfloat16
    cfg["device"] = "cuda"
    
    # Matryoshka Transcoder settings
    cfg["sae_type"] = "matryoshka-transcoder"
    cfg["dict_size"] = 18432  # 8x expansion
    cfg["group_sizes"] = [2304, 4608, 6912, 4608]  # Varying sizes
    cfg["top_k"] = 64
    cfg["l1_coeff"] = 0.0
    cfg["aux_penalty"] = 1/32
    cfg["n_batches_to_dead"] = 20
    cfg["top_k_aux"] = 512
    
    # Cross-layer mapping: resid_mid -> mlp_out (MLP transformation)
    cfg = create_transcoder_config(
        cfg,
        source_layer=8,
        target_layer=8,
        source_site="resid_mid",  # After attention, before MLP
        target_site="mlp_out"     # Output of MLP
    )
    
    cfg["wandb_project"] = "gemma-2-2b-matryoshka-transcoder"
    cfg["checkpoint_freq"] = 5000
    cfg["perf_log_freq"] = 1000
    
    print(f"Model: {cfg['model_name']}")
    print(f"Transcoder: {cfg['source_hook_point']} → {cfg['target_hook_point']}")
    print(f"Dictionary size: {cfg['dict_size']}")
    print(f"Group sizes: {cfg['group_sizes']}")
    
    # Load model
    print(f"\nLoading {cfg['model_name']}...")
    model = HookedTransformer.from_pretrained_no_processing(
        cfg["model_name"]
    ).to(cfg["model_dtype"]).to(cfg["device"])
    
    # Create transcoder activation store
    print("Creating transcoder activation store...")
    activation_store = TranscoderActivationsStore(model, cfg)
    
    # Create transcoder
    print("Initializing Matryoshka Transcoder...")
    transcoder = MatryoshkaTranscoder(cfg)
    
    # Train
    print("\nStarting training...")
    print("Metrics tracked: Loss, L2, L0, FVU (Fraction of Variance Unexplained)")
    train_transcoder(transcoder, activation_store, model, cfg)
    print("Training complete!")


def compare_gemma_vs_gpt2():
    """
    Compare configurations for Gemma-2-2B vs GPT-2.
    
    Shows the differences in activation sizes and dictionary sizes.
    """
    print("=" * 70)
    print("Model Configuration Comparison")
    print("=" * 70)
    
    models = ["gpt2-small", "gemma-2-2b"]
    
    for model_name in models:
        model_config = get_model_config(model_name)
        act_size = model_config["act_size"]
        
        print(f"\n{model_name}:")
        print(f"  Activation size: {act_size}")
        print(f"  Number of layers: {model_config['n_layers']}")
        print(f"  Typical dictionary size (16x): {act_size * 16}")
        print(f"  Typical group sizes (1x, 2x, 4x, 9x):")
        print(f"    [{act_size}, {act_size*2}, {act_size*4}, {act_size*9}]")
        print(f"  Typical top-k (dict_size/200): {(act_size * 16) // 200}")
    
    print("\n" + "=" * 70)
    print("Key differences:")
    print("  - Gemma-2-2B has 3x larger activations than GPT-2")
    print("  - Gemma-2-2B requires 3x larger dictionaries")
    print("  - Gemma-2-2B needs smaller batch sizes (memory)")
    print("  - Hook names are handled automatically by TransformerLens")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "compare"
    
    if mode == "sae":
        print("Training Gemma-2-2B Matryoshka SAE...")
        train_gemma_matryoshka_sae()
    elif mode == "transcoder":
        print("Training Gemma-2-2B Matryoshka Transcoder...")
        train_gemma_transcoder()
    elif mode == "compare":
        compare_gemma_vs_gpt2()
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python train_gemma_example.py [sae|transcoder|compare]")
        print("  sae: Train Matryoshka SAE on Gemma-2-2B")
        print("  transcoder: Train Matryoshka Transcoder on Gemma-2-2B")
        print("  compare: Show configuration comparison (default)")

