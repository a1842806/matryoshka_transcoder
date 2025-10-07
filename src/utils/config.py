import transformer_lens.utils as utils
import torch 


def get_model_config(model_name):
    """
    Get model-specific configuration including correct hook names.
    
    Gemma-2 models use different hook naming conventions than GPT-2.
    TransformerLens handles this, but we need to set proper defaults.
    """
    model_configs = {
        "gpt2": {"act_size": 768, "n_layers": 12},
        "gpt2-medium": {"act_size": 1024, "n_layers": 24},
        "gpt2-large": {"act_size": 1280, "n_layers": 36},
        "gpt2-xl": {"act_size": 1600, "n_layers": 48},
        "gemma-2-2b": {"act_size": 2304, "n_layers": 26},
        "gemma-2-9b": {"act_size": 3584, "n_layers": 42},
        "gemma-2-27b": {"act_size": 4608, "n_layers": 46},
    }
    
    # Check for model family
    for model_key, config in model_configs.items():
        if model_key in model_name.lower():
            return config
    
    # Default fallback
    return {"act_size": 768, "n_layers": 12}


def get_hook_name(site, layer, model_name):
    """
    Get correct hook name for the model.
    
    TransformerLens uses different conventions for different models:
    - GPT-2: blocks.{layer}.hook_{site}
    - Gemma-2: blocks.{layer}.hook_{site} (same, but Gemma has different sites)
    
    Available sites:
    - resid_pre: residual stream before the layer
    - attn_out: output of attention (before adding to residual)
    - resid_mid: residual stream after attention, before MLP
    - mlp_out: output of MLP (before adding to residual)
    - resid_post: residual stream after the layer
    - ln2_normalized: RMSNorm output (post-norm, pre-scale) - CORRECT for Gemma-2 MLP input
    - mlp_in: pre-norm stream (only if use_hook_mlp_in=True)
    """
    # Handle special cases for Gemma-2 models
    if "gemma" in model_name.lower():
        if site == "ln2_normalized":
            return f"blocks.{layer}.ln2.hook_normalized"
        elif site == "mlp_in":
            return f"blocks.{layer}.hook_mlp_in"
    
    # TransformerLens handles the naming automatically for standard sites
    return utils.get_act_name(site, layer)


def get_default_cfg():
    default_cfg = {
        "seed": 49,
        "batch_size": 4096,
        "lr": 3e-4,
        "num_tokens": int(1e9),
        "l1_coeff": 0,
        "beta1": 0.9,
        "beta2": 0.99,
        "max_grad_norm": 100000,
        "seq_len": 128,
        "dtype": torch.float32,
        "model_dtype": torch.float32,
        "model_name": "gpt2-small",
        "site": "resid_pre",
        "layer": 8,
        "act_size": 768,
        "dict_size": 12288,
        "device": "cuda:0",
        "model_batch_size": 512,
        "num_batches_in_buffer": 10,
        "dataset_path": "Skylion007/openwebtext",
        "wandb_project": "sparse_autoencoders",
        "input_unit_norm": True,
        "perf_log_freq": 1000,
        "sae_type": "topk",
        "checkpoint_freq": 10000,
        "n_batches_to_dead": 20,

        # (Batch)TopKSAE specific
        "top_k": 32,
        "top_k_aux": 512,
        "aux_penalty": (1/32),
        
        # for jumprelu
        "bandwidth": 0.001,
        
        # Activation sample collection (Anthropic-style interpretability)
        "save_activation_samples": False,  # Enable to collect high-activation samples
        "sample_collection_freq": 1000,     # Collect samples every N steps
        "max_samples_per_feature": 100,     # Max samples to store per feature
        "sample_context_size": 20,          # Tokens of context around activation
        "sample_activation_threshold": 0.1, # Minimum activation to consider
        "top_features_to_save": 100,        # Number of top features to save samples for
        "samples_per_feature_to_save": 10,  # Number of samples to save per feature
        "store_tokens_for_samples": False,  # Store tokens in activation buffer
    }
    default_cfg = post_init_cfg(default_cfg)
    return default_cfg

def post_init_cfg(cfg):
    """
    Post-process configuration to add derived fields.
    Also applies model-specific defaults if not already set.
    """
    # Get model-specific defaults
    model_config = get_model_config(cfg["model_name"])
    
    # Set act_size from model config if not explicitly provided
    if "act_size" not in cfg or cfg["act_size"] is None:
        cfg["act_size"] = model_config["act_size"]
    
    # Set hook point using proper naming
    cfg["hook_point"] = get_hook_name(cfg["site"], cfg["layer"], cfg["model_name"])
    
    # Create descriptive name
    cfg["name"] = f"{cfg['model_name']}_{cfg['hook_point']}_{cfg['dict_size']}_{cfg['sae_type']}_{cfg['top_k']}_{cfg['lr']}"
    
    return cfg


def create_gemma_mlp_transcoder_config(base_cfg, layer):
    """
    Create transcoder configuration for Gemma-2 MLP transformation.
    
    Uses resid_mid -> mlp_out transformation (working approach).
    
    Args:
        base_cfg: Base configuration dictionary
        layer: Layer number to target
    
    Returns:
        Updated configuration dictionary with proper hooks
    """
    cfg = base_cfg.copy()
    
    # Set layer information
    cfg["source_layer"] = layer
    cfg["target_layer"] = layer
    cfg["layer"] = layer
    
    # Use resid_mid (working approach)
    cfg["source_site"] = "resid_mid"
    cfg["target_site"] = "mlp_out"
    cfg["source_hook_point"] = f"blocks.{layer}.hook_resid_mid"
    cfg["target_hook_point"] = f"blocks.{layer}.hook_mlp_out"
    
    # Update name
    cfg["name"] = (
        f"{cfg['model_name']}_residmid_to_mlpout_{layer}_"
        f"{cfg['dict_size']}_{cfg['sae_type']}_{cfg['top_k']}_{cfg['lr']}"
    )
    
    # Set hook_point for compatibility
    cfg["hook_point"] = cfg["source_hook_point"]
    
    return cfg


def compute_fvu(x_original, x_reconstruct):
    """
    Compute Fraction of Variance Unexplained (FVU).
    
    FVU = Var(residuals) / Var(original)
        = MSE(original, reconstruct) / Var(original)
    
    Lower FVU is better (0 = perfect reconstruction, 1 = no better than mean)
    
    Args:
        x_original: Original input tensor
        x_reconstruct: Reconstructed tensor
    
    Returns:
        FVU as a scalar tensor
    """
    # Compute residuals
    residuals = x_original.float() - x_reconstruct.float()
    
    # Variance of residuals (MSE)
    var_residuals = residuals.pow(2).mean()
    
    # Variance of original data
    var_original = x_original.float().var()
    
    # FVU = Var(residuals) / Var(original)
    # Add small epsilon to avoid division by zero
    fvu = var_residuals / (var_original + 1e-8)
    
    return fvu