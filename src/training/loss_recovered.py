import torch
from transformer_lens import HookedTransformer
from typing import Any


@torch.no_grad()
def compute_loss_recovered_for_transcoder(
    model: HookedTransformer,
    transcoder: Any,
    activation_store: Any,
    source_hook_name: str,
    target_hook_name: str,
    n_batches: int = 5,
) -> dict[str, float]:
    model.eval()
    transcoder.eval()
    
    ce_losses_without_transcoder = []
    ce_losses_with_transcoder = []
    ce_losses_with_ablation = []
    
    for _ in range(n_batches):
        batch_tokens = activation_store.get_batch_tokens()
        batch_tokens = batch_tokens.to(model.cfg.device)
        
        source_activations_cache = {}
        
        def cache_source_activations_hook(activations: torch.Tensor, hook: Any) -> torch.Tensor:
            source_activations_cache[hook.name] = activations.detach().clone()
            return activations
        
        with model.hooks([(source_hook_name, cache_source_activations_hook)]):
            _, original_ce_loss = model(
                batch_tokens, 
                return_type="both", 
                loss_per_token=True
            )
        
        ce_losses_without_transcoder.append(original_ce_loss.mean().item())
        
        source_activations = source_activations_cache[source_hook_name]
        batch_size_actual, seq_len = source_activations.shape[:2]
        source_flat = source_activations.reshape(-1, source_activations.shape[-1])
        
        if source_flat.shape[-1] != transcoder.config["source_act_size"]:
            raise ValueError(
                f"Source activation size mismatch: expected {transcoder.config['source_act_size']}, "
                f"got {source_flat.shape[-1]}"
            )
        
        transcoder_prediction_flat = transcoder.decode(transcoder.encode(source_flat))
        target_dim = transcoder.config["target_act_size"]
        transcoder_prediction = transcoder_prediction_flat.reshape(
            batch_size_actual, seq_len, target_dim
        )
        transcoder_prediction = transcoder_prediction.to(model.cfg.device)
        
        def replace_with_transcoder_hook(activations: torch.Tensor, hook: Any) -> torch.Tensor:
            return transcoder_prediction.to(activations.device, dtype=activations.dtype)
        
        with model.hooks([(target_hook_name, replace_with_transcoder_hook)]):
            _, ce_loss_with_transcoder = model(
                batch_tokens,
                return_type="both",
                loss_per_token=True
            )
        ce_losses_with_transcoder.append(ce_loss_with_transcoder.mean().item())
        
        def replace_with_ablation_hook(activations: torch.Tensor, hook: Any) -> torch.Tensor:
            return torch.zeros_like(activations)
        
        with model.hooks([(target_hook_name, replace_with_ablation_hook)]):
            _, ce_loss_with_ablation = model(
                batch_tokens,
                return_type="both",
                loss_per_token=True
            )
        ce_losses_with_ablation.append(ce_loss_with_ablation.mean().item())
    
    H_org = sum(ce_losses_without_transcoder) / len(ce_losses_without_transcoder)
    H_phi = sum(ce_losses_with_transcoder) / len(ce_losses_with_transcoder)
    H_0 = sum(ce_losses_with_ablation) / len(ce_losses_with_ablation)
    
    denominator = H_0 - H_org
    if abs(denominator) < 1e-8:
        loss_recovered = 0.0
    else:
        loss_recovered = (H_0 - H_phi) / denominator
    
    return {
        "ce_loss_without_transcoder": H_org,
        "ce_loss_with_transcoder": H_phi,
        "ce_loss_with_ablation": H_0,
        "loss_recovered": loss_recovered,
    }
