import wandb
import torch

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
