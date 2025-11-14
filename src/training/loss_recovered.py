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
    
    device = next(model.parameters()).device
    
    for _ in range(n_batches):
        batch_tokens = activation_store.get_batch_tokens()
        batch_tokens = batch_tokens.to(device)
        
        model.reset_hooks()
        _, original_ce_loss = model(
            batch_tokens, 
            return_type="both", 
            loss_per_token=True
        )
        ce_losses_without_transcoder.append(original_ce_loss.mean().item())
        
        layer_num = source_hook_name.split(".")[1] if "." in source_hook_name else str(model.cfg.n_layers - 1)
        resid_mid_hook_name = f"blocks.{layer_num}.hook_resid_mid"
        resid_post_hook_name = f"blocks.{layer_num}.hook_resid_post"
        
        source_activations_storage = [None]
        
        def cache_source_hook(activations: torch.Tensor, hook: Any) -> torch.Tensor:
            source_activations_storage[0] = activations.detach().clone()
            return activations
        
        model.reset_hooks()
        with model.hooks([(source_hook_name, cache_source_hook)]):
            _ = model(batch_tokens)
        
        if source_activations_storage[0] is None:
            raise RuntimeError("Source activations not cached")
        
        source_acts = source_activations_storage[0]
        batch_size_actual, seq_len = source_acts.shape[:2]
        source_flat = source_acts.reshape(-1, source_acts.shape[-1])
        
        if source_flat.shape[-1] != transcoder.config["source_act_size"]:
            raise ValueError(
                f"Source activation size mismatch: expected {transcoder.config['source_act_size']}, "
                f"got {source_flat.shape[-1]}"
            )
        
        with torch.no_grad():
            transcoder_device = next(transcoder.parameters()).device
            source_flat_device = source_flat.to(transcoder_device)
            transcoder_prediction_flat = transcoder.decode(transcoder.encode(source_flat_device))
            target_dim = transcoder.config["target_act_size"]
            transcoder_prediction = transcoder_prediction_flat.reshape(
                batch_size_actual, seq_len, target_dim
            )
            transcoder_prediction = transcoder_prediction.to(device)
        
        resid_mid_storage = [None]
        def cache_resid_mid_hook(activations: torch.Tensor, hook: Any) -> torch.Tensor:
            resid_mid_storage[0] = activations.detach().clone()
            return activations
        
        def replace_with_transcoder_hook(activations: torch.Tensor, hook: Any) -> torch.Tensor:
            if resid_mid_storage[0] is not None:
                return resid_mid_storage[0].to(activations.device, dtype=activations.dtype) + transcoder_prediction.to(activations.device, dtype=activations.dtype)
            return activations
        
        def replace_with_ablation_hook(activations: torch.Tensor, hook: Any) -> torch.Tensor:
            return torch.zeros_like(activations)
        
        model.reset_hooks()
        with model.hooks([
            (resid_mid_hook_name, cache_resid_mid_hook),
            (resid_post_hook_name, replace_with_transcoder_hook)
        ]):
            _, ce_loss_with_transcoder = model(
                batch_tokens,
                return_type="both",
                loss_per_token=True
            )
        ce_losses_with_transcoder.append(ce_loss_with_transcoder.mean().item())
        
        model.reset_hooks()
        resid_mid_storage[0] = None
        with model.hooks([
            (resid_mid_hook_name, cache_resid_mid_hook),
            (resid_post_hook_name, replace_with_ablation_hook)
        ]):
            _, ce_loss_with_ablation = model(
                batch_tokens,
                return_type="both",
                loss_per_token=True
            )
        ce_losses_with_ablation.append(ce_loss_with_ablation.mean().item())
        
        model.reset_hooks()
    
    H_org = sum(ce_losses_without_transcoder) / len(ce_losses_without_transcoder)
    H_phi = sum(ce_losses_with_transcoder) / len(ce_losses_with_transcoder)
    H_0 = sum(ce_losses_with_ablation) / len(ce_losses_with_ablation)
    
    if H_0 < H_org:
        # print(f"\n⚠️  CRITICAL: Ablation loss ({H_0:.4f}) < Original loss ({H_org:.4f}).")
        # print("This indicates the hook replacement is not working correctly.")
        # print("Expected: H_0 (ablation) >= H_org (original)")
        return {
            "ce_loss_without_transcoder": H_org,
            "ce_loss_with_transcoder": H_phi,
            "ce_loss_with_ablation": H_0,
            "loss_recovered": 0.0,
            "error": "Ablation loss < Original loss - hook replacement failed"
        }
    
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
