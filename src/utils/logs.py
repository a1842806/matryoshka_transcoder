import wandb
import torch
from functools import partial
import os
import json

def init_wandb(cfg):
    return wandb.init(project=cfg["wandb_project"], name=cfg["name"], config=cfg, reinit=True)

def log_wandb(output, step, wandb_run, index=None):
    log_dict = {
        k: v.item() if isinstance(v, torch.Tensor) and v.dim() == 0 else v
        for k, v in output.items() if isinstance(v, (int, float)) or 
        (isinstance(v, torch.Tensor) and v.dim() == 0)
    }

    if index is not None:
        log_dict = {f"{k}_{index}": v for k, v in log_dict.items()}

    wandb_run.log(log_dict, step=step)

# Hooks for model performance evaluation
def reconstr_hook(activation, hook, sae_out):
    return sae_out

def zero_abl_hook(activation, hook):
    return torch.zeros_like(activation)

def mean_abl_hook(activation, hook):
    return activation.mean([0, 1]).expand_as(activation)

@torch.no_grad()
def log_model_performance(wandb_run, step, model, activations_store, transcoder, index=None, batch_tokens=None):
    """
    Log transcoder performance metrics.
    Note: This function is simplified for transcoder-only usage.
    """
    if batch_tokens is None:
        # For transcoders, we need paired activations
        source_batch, target_batch = activations_store.next_batch()
    else:
        # Get paired activations from tokens
        source_batch, target_batch = activations_store.get_paired_activations(batch_tokens)
    
    # Get transcoder output
    transcoder_output = transcoder(source_batch, target_batch)
    
    # Log basic metrics
    log_dict = {
        "performance/l2_loss": transcoder_output["l2_loss"].item(),
        "performance/fvu": transcoder_output["fvu"].item(),
        "performance/l0_norm": transcoder_output["l0_norm"].item(),
        "performance/aux_loss": transcoder_output["aux_loss"].item(),
    }

    if index is not None:
        log_dict = {f"{k}_{index}": v for k, v in log_dict.items()}

    wandb_run.log(log_dict, step=step)

def save_checkpoint(wandb_run, model, cfg, step):
    """
    Save checkpoint with organized structure:
    checkpoints/{model_type}/{base_model}/{name}_{step}/
    """
    # Determine model type (sae or transcoder)
    sae_type = cfg.get("sae_type", "topk")
    if "transcoder" in sae_type or "clt" in sae_type:
        model_type = "transcoder"
    else:
        model_type = "sae"
    
    # Get base model name (e.g., "gpt2-small", "gemma-2-2b")
    base_model = cfg["model_name"]
    
    # Create organized directory structure
    save_dir = f"checkpoints/{model_type}/{base_model}/{cfg['name']}_{step}"
    os.makedirs(save_dir, exist_ok=True)

    # Save model state
    model_path = os.path.join(save_dir, "sae.pt")
    torch.save(model.state_dict(), model_path)

    # Prepare config for JSON serialization
    json_safe_cfg = {}
    for key, value in cfg.items():
        if isinstance(value, (int, float, str, bool, type(None))):
            json_safe_cfg[key] = value
        elif isinstance(value, (torch.dtype, type)):
            json_safe_cfg[key] = str(value)
        else:
            json_safe_cfg[key] = str(value)

    # Save config
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(json_safe_cfg, f, indent=4)

    # Create and log artifact
    artifact = wandb.Artifact(
        name=f"{cfg['name']}_{step}",
        type="model",
        description=f"Model checkpoint at step {step}",
    )
    artifact.add_file(model_path)
    artifact.add_file(config_path)
    wandb_run.log_artifact(artifact)

    print(f"Model and config saved to {save_dir}")

