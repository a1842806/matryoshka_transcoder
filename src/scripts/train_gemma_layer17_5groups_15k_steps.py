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



GROUP_SIZES_5_NESTED = 

def main():
    cfg = get_default_cfg()
    
    # Model settings
    cfg["model_name"] = "gemma-2-2b"
    cfg["dataset_path"] = "HuggingFaceFW/fineweb-edu"
    cfg["layer"] = 17
    
    # Training settings for 15k steps
    cfg["num_tokens"] = int(15e6)  # 15M tokens for ~15k steps
    cfg["model_batch_size"] = 4
    cfg["batch_size"] = 1024
    cfg["seq_len"] = 64
    cfg["lr"] = 4e-4
    cfg["model_dtype"] = torch.bfloat16
    cfg["dtype"] = torch.bfloat16
    cfg["device"] = "cuda:1"
    
    # Learning rate scheduling
    cfg["scheduler_type"] = "warmup_decay"
    cfg["warmup_steps"] = 1000  # 1000 steps warmup for 15k total
    cfg["min_lr"] = cfg["lr"] * 0.01
    
    cfg["dict_size"] = 18432
    cfg["prefix_sizes"] = [1152, 2073, 3732, 6718, 4757]
    
    print(f"5 Prefixes: {cfg['prefix_sizes']} (total: {sum(cfg['prefix_sizes']):,})")
    
    # Matryoshka-specific settings
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
    
    # Logging and checkpointing
    cfg["perf_log_freq"] = 50
    cfg["checkpoint_freq"] = 500
    cfg["wandb_project"] = "gemma-2-2b-layer17-5groups-15k"
    
    # Create transcoder config
    cfg = create_transcoder_config(
        cfg,
        source_layer=17,
        target_layer=17,
        source_site="mlp_in",
        target_site="mlp_out"
    )
    
    # Ensure correct activation sizes
    cfg["source_act_size"] = 2304
    cfg["target_act_size"] = 2304
    cfg["input_unit_norm"] = False
    
    cfg = post_init_cfg(cfg)

    model = HookedTransformer.from_pretrained_no_processing(
        cfg["model_name"], 
        dtype=cfg["model_dtype"]
    ).to(cfg["device"])
    
    activation_store = TranscoderActivationsStore(model, cfg)
    transcoder = MatryoshkaTranscoder(cfg).to(cfg["device"])
    expected_steps = int(cfg["num_tokens"] // cfg["batch_size"])
    
    try:
        train_transcoder(transcoder, activation_store, model, cfg)
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
