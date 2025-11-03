"""
TRAINING SCRIPT TEMPLATE FOR MATRYOSHKA SAE

IMPORTANT: This is a TEMPLATE file with detailed comments for reference.
When creating specific training scripts, REMOVE ALL COMMENTS and keep only the clean code.
The specific training scripts should be clean and minimal like train_gemma_layer17_5groups_15k_steps.py

This template shows all available configuration options with explanations.

RESULTS ORGANIZATION:
- Training results are automatically saved to organized structure: results/{model}/{layerX}/{steps}/
- No need to manually manage checkpoints or activation samples
- Use view_activation_samples.py to browse and view results
- Use src/utils/cleanup_results.py to migrate legacy checkpoints if needed
"""

import torch, sys, os
from transformer_lens import HookedTransformer

# Add the parent directory to the path to import project modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.sae import MatryoshkaTranscoder
from src.models.transcoder_activation_store import TranscoderActivationsStore, create_transcoder_config
from src.training.training import train_transcoder
from src.utils.config import get_default_cfg, post_init_cfg

def main():
    # Initialize configuration with default values
    cfg = get_default_cfg()
    
    # ============================================================================
    # MODEL CONFIGURATION
    # ============================================================================
    
    # Model to train on - options: "gemma-2-2b", "gemma-2-9b", etc.
    cfg["model_name"] = "gemma-2-2b"
    
    # Dataset for training - options: "HuggingFaceFW/fineweb-edu", custom datasets
    cfg["dataset_path"] = "HuggingFaceFW/fineweb-edu"
    
    # Layer to extract activations from (0-indexed)
    cfg["layer"] = 17
    
    # ============================================================================
    # TRAINING HYPERPARAMETERS
    # ============================================================================
    
    # Total number of tokens to process during training
    # Calculate steps as: num_tokens // batch_size
    # Example: 15M tokens / 1024 batch_size = ~14.6k steps
    cfg["num_tokens"] = int(15e6)  # 15M tokens for ~15k steps
    
    # Number of sequences processed in parallel by the model
    # Higher = more GPU memory usage, faster training
    cfg["model_batch_size"] = 4
    
    # Number of activation vectors per batch
    # Higher = more stable gradients, more GPU memory usage
    cfg["batch_size"] = 1024
    
    # Length of each sequence (context window)
    # Affects memory usage: longer = more memory
    cfg["seq_len"] = 128
    
    # Learning rate for optimizer
    # Typical range: 1e-5 to 1e-3
    cfg["lr"] = 4e-4
    
    # Data type for model weights (affects memory and speed)
    # Options: torch.float32, torch.bfloat16, torch.float16
    cfg["model_dtype"] = torch.bfloat16
    
    # Data type for SAE training (can differ from model dtype)
    cfg["dtype"] = torch.bfloat16
    
    # Device to run training on
    # Options: "cuda:0", "cuda:1", "cpu"
    cfg["device"] = "cuda"
    
    # ============================================================================
    # LEARNING RATE SCHEDULING
    # ============================================================================
    
    # Type of learning rate scheduler
    # Options: "constant", "warmup_decay", "cosine", "linear"
    cfg["scheduler_type"] = "warmup_decay"
    
    # Number of steps to gradually increase learning rate from 0
    # Should be ~5-10% of total training steps
    cfg["warmup_steps"] = 1000  # 1000 steps warmup for 15k total
    
    # Minimum learning rate after decay
    # Usually 1-10% of initial learning rate
    cfg["min_lr"] = cfg["lr"] * 0.01
    
    # ============================================================================
    # SAE ARCHITECTURE CONFIGURATION
    # ============================================================================
    
    # Total dictionary size (number of features)
    # Larger = more capacity, more parameters, longer training
    cfg["dict_size"] = 18432
    
    # Matryoshka prefix sizes following paper methodology
    # Paper uses fixed, exponentially-spaced prefixes ending at full dictionary
    # Key principles from Matryoshka Transcoder paper:
    # 1. Fixed prefixes > random prefixes (better eval metrics on Gemma-2-2B)
    # 2. Geometric spacing: cover "coarse → fine" scales without redundancy
    # 3. 4-6 prefixes is the sweet spot
    # 4. First prefix: ~3-6% of D (sufficient for broad features)
    # 5. Last prefix: equals full dictionary size
    # 
    # This implements the "doubling strategy" from paper appendix (cleaner ratios)
    # For dict_size=18432: [D/8, D/4, D/2, D/2+D/4, D] = [2304, 4608, 9216, 13824, 18432]
    # Alternative "paper-style" (start ~D/32): [576, 1728, 4032, 8640, 18432]
    # 
    # Paper's main Gemma-2-2B run (D=65536): [2048, 6144, 14336, 30720, 65536]
    cfg["prefix_sizes"] = [2304, 4608, 9216, 13824, 18432]
    
    # ============================================================================
    # MATRYOSHKA-SPECIFIC HYPERPARAMETERS
    # ============================================================================
    
    # Number of top features to use for reconstruction loss
    # Higher = more features used, less sparse
    cfg["top_k"] = 96
    
    # Weight for auxiliary loss (encourages feature diversity)
    # Typical range: 1/32 to 1/128
    cfg["aux_penalty"] = 1/64
    
    # Number of batches before marking features as "dead"
    # Features that don't activate are reset
    cfg["n_batches_to_dead"] = 20
    
    # Number of top features for auxiliary loss
    # Usually 2-4x larger than top_k
    cfg["top_k_aux"] = 256
    
    # ============================================================================
    # ACTIVATION SAMPLE COLLECTION (for analysis)
    # ============================================================================
    
    # Whether to collect activation samples during training
    # Useful for understanding what features learn
    cfg["save_activation_samples"] = True
    
    # How often to collect samples (in steps)
    cfg["sample_collection_freq"] = 100
    
    # Maximum samples to collect per feature
    cfg["max_samples_per_feature"] = 100
    
    # Context size around each activation sample
    cfg["sample_context_size"] = 20
    
    # Minimum activation value to save a sample
    cfg["sample_activation_threshold"] = 0.1
    
    # Number of top features to save samples for
    cfg["top_features_to_save"] = 100
    
    # Number of samples to save per feature
    cfg["samples_per_feature_to_save"] = 10
    
    # ============================================================================
    # LOGGING AND CHECKPOINTING
    # ============================================================================
    
    # How often to log performance metrics (in steps)
    cfg["perf_log_freq"] = 50
    
    # How often to save model checkpoints (in steps)
    cfg["checkpoint_freq"] = 500
    
    # Weights & Biases project name for experiment tracking
    cfg["wandb_project"] = "gemma-2-2b-layer17-5groups-15k"
    
    # Experiment description for clean results organization
    # This will be used in the organized directory structure: results/model/layer/date_description/
    cfg["experiment_description"] = "15k-steps-5groups"
    
    # ============================================================================
    # TRANSCODER CONFIGURATION
    # ============================================================================
    
    # Create transcoder configuration for layer-to-layer reconstruction
    cfg = create_transcoder_config(
        cfg,
        source_layer=17,      # Layer to extract activations from
        target_layer=17,       # Layer to reconstruct (usually same as source)
        source_site="mlp_in",  # Activation site: "mlp_in", "mlp_out", "resid_pre", "resid_post"
        target_site="mlp_out"  # Target site to reconstruct
    )
    
    # ============================================================================
    # ACTIVATION SIZE CONFIGURATION
    # ============================================================================
    
    # Size of input activations (depends on model and layer)
    # For Gemma-2-2b MLP layers: 2304
    cfg["source_act_size"] = 2304
    
    # Size of target activations (usually same as source)
    cfg["target_act_size"] = 2304
    
    # Whether to normalize input activations to unit norm
    # Usually False for MLP activations
    cfg["input_unit_norm"] = False
    
    # ============================================================================
    # FINALIZE CONFIGURATION
    # ============================================================================
    
    # Post-process configuration (validates settings, sets derived values)
    cfg = post_init_cfg(cfg)

    # ============================================================================
    # MODEL AND DATA LOADING
    # ============================================================================
    
    # Load the transformer model
    model = HookedTransformer.from_pretrained_no_processing(
        cfg["model_name"], 
        dtype=cfg["model_dtype"]
    ).to(cfg["device"])
    
    # Create activation store for data loading
    activation_store = TranscoderActivationsStore(model, cfg)
    
    # Initialize the Matryoshka SAE
    transcoder = MatryoshkaTranscoder(cfg).to(cfg["device"])
    
    # Calculate expected number of training steps
    expected_steps = int(cfg["num_tokens"] // cfg["batch_size"])
    print(f"Expected training steps: {expected_steps}")
    
    # ============================================================================
    # TRAINING EXECUTION
    # ============================================================================
    
    # Training will automatically save results to organized structure:
    # results/gemma-2-2b/layer17/15000/
    # ├── checkpoints/final.pt
    # ├── config.json  (includes completed_at timestamp)
    # ├── metrics.json
    # └── activation_samples/
    #     ├── collection_summary.json
    #     └── feature_*.json
    
    try:
        # Start training (results automatically saved to clean structure)
        train_transcoder(transcoder, activation_store, model, cfg)
    except Exception as e:
        print(f"Training error: {e}")
        raise

if __name__ == "__main__":
    main()
