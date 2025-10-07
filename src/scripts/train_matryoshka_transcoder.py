"""
Example script for training Matryoshka Transcoders.

This script demonstrates how to train cross-layer feature transformers
that learn hierarchical mappings between different parts of a transformer.
"""

import torch
import copy
from transformer_lens import HookedTransformer
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.sae import MatryoshkaTranscoder
from models.transcoder_activation_store import TranscoderActivationsStore, create_transcoder_config
from training.training import train_transcoder, train_transcoder_group_seperate_wandb
from utils.config import get_default_cfg


def train_single_matryoshka_transcoder():
    """
    Train a single Matryoshka Transcoder.
    
    Example: Learn how features transform from resid_mid to mlp_out
    within the same layer (understanding MLP transformations).
    """
    # Base configuration
    cfg = get_default_cfg()
    cfg["model_name"] = "gpt2-small"
    cfg["dataset_path"] = "Skylion007/openwebtext"
    cfg["num_tokens"] = 1e7  # 10M tokens for quick test
    cfg["model_batch_size"] = 8
    cfg["batch_size"] = 1024
    cfg["seq_len"] = 64
    cfg["lr"] = 3e-4
    cfg["model_dtype"] = torch.bfloat16
    cfg["dtype"] = torch.bfloat16
    cfg["device"] = "cuda"
    
    # Matryoshka Transcoder specific settings
    cfg["sae_type"] = "matryoshka-transcoder"
    cfg["dict_size"] = 12288  # 16x expansion
    cfg["group_sizes"] = [768, 1536, 3072, 6912]  # Growing groups (1x, 2x, 4x, 9x)
    cfg["top_k"] = 64  # Number of active features per batch
    cfg["l1_coeff"] = 0.0  # Using TopK, not L1
    cfg["aux_penalty"] = 1/32
    cfg["n_batches_to_dead"] = 20
    cfg["top_k_aux"] = 512
    
    # Cross-layer mapping: resid_mid -> mlp_out (same layer)
    cfg = create_transcoder_config(
        cfg,
        source_layer=8,
        target_layer=8,
        source_site="resid_mid",  # After attention, before MLP
        target_site="mlp_out"     # Output of MLP
    )
    
    cfg["wandb_project"] = "matryoshka-transcoder"
    cfg["checkpoint_freq"] = 1000
    cfg["perf_log_freq"] = 500
    
    # Load model
    print(f"Loading model: {cfg['model_name']}")
    model = HookedTransformer.from_pretrained_no_processing(
        cfg["model_name"]
    ).to(cfg["model_dtype"]).to(cfg["device"])
    
    # Create activation store for paired activations
    print("Creating transcoder activation store...")
    activation_store = TranscoderActivationsStore(model, cfg)
    
    # Create Matryoshka Transcoder
    print("Initializing Matryoshka Transcoder...")
    transcoder = MatryoshkaTranscoder(cfg)
    
    print(f"Transcoder architecture:")
    print(f"  Source: {cfg['source_hook_point']}")
    print(f"  Target: {cfg['target_hook_point']}")
    print(f"  Dictionary size: {cfg['dict_size']}")
    print(f"  Group sizes: {cfg['group_sizes']}")
    print(f"  Top-K: {cfg['top_k']}")
    
    # Train
    print("\nStarting training...")
    train_transcoder(transcoder, activation_store, model, cfg)
    print("Training complete!")


def train_multiple_matryoshka_transcoders():
    """
    Train multiple Matryoshka Transcoders with different configurations.
    
    This explores different cross-layer mappings:
    1. resid_mid -> mlp_out (MLP transformation)
    2. resid_pre -> attn_out (Attention transformation)
    3. resid_pre -> resid_post (Full layer transformation)
    """
    # Base configuration
    base_cfg = get_default_cfg()
    base_cfg["model_name"] = "gpt2-small"
    base_cfg["dataset_path"] = "Skylion007/openwebtext"
    base_cfg["num_tokens"] = 1e7
    base_cfg["model_batch_size"] = 8
    base_cfg["batch_size"] = 1024
    base_cfg["seq_len"] = 64
    base_cfg["lr"] = 3e-4
    base_cfg["model_dtype"] = torch.bfloat16
    base_cfg["dtype"] = torch.bfloat16
    base_cfg["device"] = "cuda"
    base_cfg["sae_type"] = "matryoshka-transcoder"
    base_cfg["l1_coeff"] = 0.0
    base_cfg["aux_penalty"] = 1/32
    base_cfg["n_batches_to_dead"] = 20
    base_cfg["top_k_aux"] = 512
    base_cfg["wandb_project"] = "matryoshka-transcoder-multi"
    base_cfg["checkpoint_freq"] = 1000
    base_cfg["perf_log_freq"] = 500
    
    # Load model once (shared across all transcoders)
    print(f"Loading model: {base_cfg['model_name']}")
    model = HookedTransformer.from_pretrained_no_processing(
        base_cfg["model_name"]
    ).to(base_cfg["model_dtype"]).to(base_cfg["device"])
    
    # Define different transcoder configurations
    transcoder_configs = [
        {
            "name": "MLP_Transcoder",
            "source_layer": 8,
            "target_layer": 8,
            "source_site": "resid_mid",
            "target_site": "mlp_out",
            "dict_size": 12288,
            "group_sizes": [768, 1536, 3072, 6912],
            "top_k": 64,
        },
        {
            "name": "Attention_Transcoder",
            "source_layer": 8,
            "target_layer": 8,
            "source_site": "resid_pre",
            "target_site": "attn_out",
            "dict_size": 12288,
            "group_sizes": [768, 1536, 3072, 6912],
            "top_k": 64,
        },
        {
            "name": "FullLayer_Transcoder",
            "source_layer": 8,
            "target_layer": 8,
            "source_site": "resid_pre",
            "target_site": "resid_post",
            "dict_size": 18432,
            "group_sizes": [1152, 2304, 4608, 10368],
            "top_k": 96,
        },
    ]
    
    transcoders = []
    activation_stores = []
    cfgs = []
    
    for tc_cfg in transcoder_configs:
        cfg = copy.deepcopy(base_cfg)
        cfg["dict_size"] = tc_cfg["dict_size"]
        cfg["group_sizes"] = tc_cfg["group_sizes"]
        cfg["top_k"] = tc_cfg["top_k"]
        
        cfg = create_transcoder_config(
            cfg,
            source_layer=tc_cfg["source_layer"],
            target_layer=tc_cfg["target_layer"],
            source_site=tc_cfg["source_site"],
            target_site=tc_cfg["target_site"]
        )
        
        print(f"\nCreating {tc_cfg['name']}:")
        print(f"  {cfg['source_hook_point']} -> {cfg['target_hook_point']}")
        print(f"  Dict size: {cfg['dict_size']}, Groups: {cfg['group_sizes']}")
        
        # Create activation store
        activation_store = TranscoderActivationsStore(model, cfg)
        activation_stores.append(activation_store)
        
        # Create transcoder
        transcoder = MatryoshkaTranscoder(cfg)
        transcoders.append(transcoder)
        cfgs.append(cfg)
    
    # Train all transcoders with separate W&B runs
    print("\n" + "="*70)
    print("Starting training for all transcoders...")
    print("="*70)
    train_transcoder_group_seperate_wandb(transcoders, activation_stores, model, cfgs)
    print("\nAll training complete!")


def train_cross_layer_transcoder():
    """
    Train a transcoder that maps between DIFFERENT layers.
    
    Example: Learn how features transform from layer 8 to layer 9.
    """
    cfg = get_default_cfg()
    cfg["model_name"] = "gpt2-small"
    cfg["dataset_path"] = "Skylion007/openwebtext"
    cfg["num_tokens"] = 1e7
    cfg["model_batch_size"] = 8
    cfg["batch_size"] = 1024
    cfg["seq_len"] = 64
    cfg["lr"] = 3e-4
    cfg["model_dtype"] = torch.bfloat16
    cfg["dtype"] = torch.bfloat16
    cfg["device"] = "cuda"
    cfg["sae_type"] = "matryoshka-cross-layer-transcoder"
    cfg["dict_size"] = 12288
    cfg["group_sizes"] = [768, 1536, 3072, 6912]
    cfg["top_k"] = 64
    cfg["l1_coeff"] = 0.0
    cfg["aux_penalty"] = 1/32
    cfg["n_batches_to_dead"] = 20
    cfg["top_k_aux"] = 512
    
    # Cross-layer mapping: layer 8 -> layer 9
    cfg = create_transcoder_config(
        cfg,
        source_layer=8,
        target_layer=9,
        source_site="resid_post",  # Output of layer 8
        target_site="resid_pre"    # Input to layer 9
    )
    
    cfg["wandb_project"] = "matryoshka-cross-layer-transcoder"
    cfg["checkpoint_freq"] = 1000
    cfg["perf_log_freq"] = 500
    
    print(f"Training cross-layer Matryoshka Transcoder")
    print(f"  From: Layer {cfg['source_layer']} ({cfg['source_site']})")
    print(f"  To: Layer {cfg['target_layer']} ({cfg['target_site']})")
    
    # Load model
    model = HookedTransformer.from_pretrained_no_processing(
        cfg["model_name"]
    ).to(cfg["model_dtype"]).to(cfg["device"])
    
    # Create activation store
    activation_store = TranscoderActivationsStore(model, cfg)
    
    # Create transcoder
    transcoder = MatryoshkaTranscoder(cfg)
    
    # Train
    print("\nStarting training...")
    train_transcoder(transcoder, activation_store, model, cfg)
    print("Training complete!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "single"
    
    if mode == "single":
        print("Training single Matryoshka Transcoder...")
        train_single_matryoshka_transcoder()
    elif mode == "multiple":
        print("Training multiple Matryoshka Transcoders...")
        train_multiple_matryoshka_transcoders()
    elif mode == "cross_layer":
        print("Training cross-layer Matryoshka Transcoder...")
        train_cross_layer_transcoder()
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python train_matryoshka_transcoder.py [single|multiple|cross_layer]")
        print("  single: Train one transcoder (resid_mid -> mlp_out)")
        print("  multiple: Train multiple transcoders in parallel")
        print("  cross_layer: Train transcoder between different layers")

