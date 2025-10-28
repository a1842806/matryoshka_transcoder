"""Minimal training script for Gemma-2-2B Layer 8 - 16k steps."""

import torch
import sys
import os
from transformer_lens import HookedTransformer

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.sae import MatryoshkaTranscoder
from src.models.transcoder_activation_store import TranscoderActivationsStore, create_transcoder_config
from src.training.training import train_transcoder
from src.utils.config import get_default_cfg, post_init_cfg

def main():
    cfg = get_default_cfg()
    
    # Model settings
    cfg["model_name"] = "gemma-2-2b"
    cfg["dataset_path"] = "HuggingFaceFW/fineweb-edu"
    cfg["layer"] = 8
    
    # Training settings for 16k steps
    cfg["num_tokens"] = int(16e6)  # 16M tokens for ~16k steps
    cfg["model_batch_size"] = 4
    cfg["batch_size"] = 1024
    cfg["seq_len"] = 64
    cfg["lr"] = 4e-4
    cfg["model_dtype"] = torch.bfloat16
    cfg["dtype"] = torch.bfloat16
    cfg["device"] = "cuda:1"
    
    # Learning rate scheduling
    cfg["scheduler_type"] = "warmup_decay"
    cfg["warmup_steps"] = 1000
    cfg["min_lr"] = cfg["lr"] * 0.01
    
    # Matryoshka settings
    cfg["dict_size"] = 18432
    cfg["prefix_sizes"] = [1152, 2073, 3732, 6718, 4757]
    cfg["top_k"] = 96
    cfg["aux_penalty"] = 1/64
    cfg["n_batches_to_dead"] = 20
    cfg["top_k_aux"] = 256
    
    # Activation sample collection
    cfg["save_activation_samples"] = True
    cfg["sample_collection_freq"] = 100
    cfg["max_samples_per_feature"] = 100
    cfg["sample_context_size"] = 20
    cfg["sample_activation_threshold"] = 0.1
    cfg["top_features_to_save"] = 100
    cfg["samples_per_feature_to_save"] = 10
    
    # Logging
    cfg["perf_log_freq"] = 50
    cfg["checkpoint_freq"] = 500
    cfg["wandb_project"] = "gemma-2-2b-layer8-16k-steps"
    cfg["experiment_description"] = "16k-steps-layer8"

    # Create transcoder config
    cfg = create_transcoder_config(
        cfg,
        source_layer=8,
        target_layer=8,
        source_site="mlp_in",
        target_site="mlp_out"
    )
    
    cfg["source_act_size"] = 2304
    cfg["target_act_size"] = 2304
    cfg["input_unit_norm"] = False
    
    cfg = post_init_cfg(cfg)

    print("=" * 80)
    print("🚀 STARTING TRAINING - Gemma-2-2B Layer 8 (16k steps)")
    print("=" * 80)
    print(f"Model: {cfg['model_name']}")
    print(f"Layer: {cfg['layer']}")
    print(f"Device: {cfg['device']}")
    print(f"Training tokens: {cfg['num_tokens']:,}")
    print(f"Expected steps: ~{cfg['num_tokens'] // cfg['batch_size']:,}")
    print(f"Dictionary size: {cfg['dict_size']:,}")
    print(f"Prefix sizes: {cfg['prefix_sizes']}")
    print(f"Learning rate: {cfg['lr']}")
    print(f"W&B project: {cfg['wandb_project']}")
    print("=" * 80)

    model = HookedTransformer.from_pretrained_no_processing(
        cfg["model_name"], 
        dtype=cfg["model_dtype"]
    ).to(cfg["device"])
    
    activation_store = TranscoderActivationsStore(model, cfg)
    transcoder = MatryoshkaTranscoder(cfg).to(cfg["device"])
    
    try:
        train_transcoder(transcoder, activation_store, model, cfg)
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("Results saved to organized structure:")
        print("  - Checkpoint: results/gemma_2_2b/layer8/[date]_16k-steps-layer8/")
        print("  - Activation samples: results/gemma_2_2b/layer8/[date]_16k-steps-layer8/activation_samples/")
        print("  - W&B dashboard: https://wandb.ai/[entity]/gemma-2-2b-layer8-16k-steps")
        print("=" * 80)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
