"""
Train Gemma-2-2B Matryoshka Transcoder on Layer 17 with warmup+decay and sample collection.

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

    cfg = get_default_cfg()
    
    cfg["model_name"] = "gemma-2-2b"
    cfg["dataset_path"] = "HuggingFaceFW/fineweb-edu"
    cfg["layer"] = 17
    
    cfg["num_tokens"] = int(3e6)
    cfg["model_batch_size"] = 4
    cfg["batch_size"] = 1024
    cfg["seq_len"] = 64
    cfg["lr"] = 4e-4
    cfg["model_dtype"] = torch.bfloat16
    cfg["dtype"] = torch.bfloat16
    cfg["device"] = "cuda:1"
    
    cfg["scheduler_type"] = "warmup_decay"
    cfg["warmup_steps"] = 500
    cfg["min_lr"] = cfg["lr"] * 0.01
    
    cfg["dict_size"] = 18432
    cfg["group_sizes"] = [1152, 2304, 4608, 10368]
    cfg["top_k"] = 96
    cfg["aux_penalty"] = 1/64
    
    cfg["n_batches_to_dead"] = 20
    cfg["top_k_aux"] = 256
    
    cfg["save_activation_samples"] = True
    cfg["sample_collection_freq"] = 100
    cfg["max_samples_per_feature"] = 100
    cfg["sample_context_size"] = 20
    cfg["sample_activation_threshold"] = 0.1
    cfg["top_features_to_save"] = 100
    cfg["samples_per_feature_to_save"] = 10
    
    cfg["perf_log_freq"] = 50
    cfg["checkpoint_freq"] = 500
    cfg["wandb_project"] = "gemma-2-2b-layer17-interpretability"
    
    cfg = create_transcoder_config(
        cfg,
        source_layer=17,
        target_layer=17,
        source_site="mlp_in",
        target_site="mlp_out"
    )
    
    cfg["source_act_size"] = 2304
    cfg["target_act_size"] = 2304
    cfg["input_unit_norm"] = False
    
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

    print(f"\nLoading {cfg['model_name']}...")
    model = HookedTransformer.from_pretrained_no_processing(
        cfg["model_name"], 
        dtype=cfg["model_dtype"]
    ).to(cfg["device"])
    
    print(f"Model loaded successfully!")
    print(f"Model config: {model.cfg}")

    print(f"\nCreating activation store...")
    activation_store = TranscoderActivationsStore(model, cfg)

    print(f"\nCreating Matryoshka Transcoder...")
    transcoder = MatryoshkaTranscoder(cfg).to(cfg["device"])
    
    print(f"Transcoder created:")
    print(f"  Dictionary size: {cfg['dict_size']}")
    print(f"  Group sizes: {cfg['group_sizes']}")
    print(f"  Top-k per group: {cfg['top_k']}")
    print(f"  Total parameters: {sum(p.numel() for p in transcoder.parameters()):,}")

    expected_steps = int(cfg["num_tokens"] // cfg["batch_size"])
    print(f"\nExpected training steps: {expected_steps:,}")
    print(f"Sample collections: ~{expected_steps // cfg['sample_collection_freq']}")
    print(f"Checkpoints: ~{expected_steps // cfg['checkpoint_freq']}")

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
    