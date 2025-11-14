"""Evaluate Loss Recovered metric on a trained transcoder checkpoint."""

import os
import sys
import json
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.sae import MatryoshkaTranscoder
from src.models.transcoder_activation_store import TranscoderActivationsStore
from src.training.loss_recovered import compute_loss_recovered_for_transcoder
from src.utils.config import post_init_cfg


def load_checkpoint(checkpoint_path: str, config_path: str, device: torch.device):
    """Load transcoder from checkpoint and config."""
    with open(config_path, "r") as f:
        cfg = json.load(f)
    
    for key in ["dtype", "model_dtype"]:
        if key in cfg and isinstance(cfg[key], str):
            cfg[key] = getattr(torch, cfg[key].replace("torch.", ""))
    
    cfg["device"] = device
    
    cfg = post_init_cfg(cfg)
    
    transcoder = MatryoshkaTranscoder(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=str(device))
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        transcoder.load_state_dict(checkpoint["model_state_dict"])
    else:
        transcoder.load_state_dict(checkpoint)
    
    transcoder.eval()
    transcoder.to(device)
    
    return transcoder, cfg


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Loss Recovered metric on a trained transcoder")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file (e.g., results/gemma-2-2b/layer8/16000/checkpoints/final.pt)")
    parser.add_argument("--config", type=str, help="Path to config file (default: checkpoint_dir/config.json)")
    parser.add_argument("--n-batches", type=int, default=10, help="Number of batches to evaluate over (default: 10)")
    parser.add_argument("--output", type=str, help="Path to save results JSON (optional)")
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = checkpoint_path.parent / "config.json"
        if not config_path.exists():
            config_path = checkpoint_path.parent.parent / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Loading config: {config_path}")
    
    device = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)
    
    transcoder, cfg = load_checkpoint(str(checkpoint_path), str(config_path), device)
    
    print(f"Model: {cfg['model_name']}")
    print(f"Layer: {cfg['layer']}")
    
    model = HookedTransformer.from_pretrained_no_processing(cfg["model_name"], dtype=cfg["model_dtype"])
    model = model.to(device)
    model.eval()
    
    activation_store = TranscoderActivationsStore(model, cfg)
    
    from src.utils.config import get_hook_name
    
    source_site = cfg.get("source_site", "mlp_in")
    target_site = cfg.get("target_site", "mlp_out")
    
    source_hook_name = cfg.get("source_hook_point")
    if source_hook_name and "hook_mlp_in" in source_hook_name and "gemma" in cfg['model_name'].lower():
        source_hook_name = get_hook_name("mlp_in", cfg['layer'], cfg['model_name'])
    
    target_hook_name = cfg.get("target_hook_point") or get_hook_name(target_site, cfg['layer'], cfg['model_name'])
    
    if not source_hook_name:
        source_hook_name = get_hook_name(source_site, cfg['layer'], cfg['model_name'])
    
    resid_post_hook_name = f"blocks.{cfg['layer']}.hook_resid_post"
    print(f"\nNote: For transcoders, consider using resid_post hook ({resid_post_hook_name})")
    print(f"      instead of mlp_out hook ({target_hook_name}) for more accurate Loss Recovered.")
    
    print(f"\nComputing Loss Recovered metric over {args.n_batches} batches...")
    print(f"Source hook: {source_hook_name}")
    print(f"Target hook: {target_hook_name}")
    
    results = compute_loss_recovered_for_transcoder(
        model=model,
        transcoder=transcoder,
        activation_store=activation_store,
        source_hook_name=source_hook_name,
        target_hook_name=target_hook_name,
        n_batches=args.n_batches,
    )
    
    print("\n" + "="*60)
    print("Loss Recovered Results:")
    print("="*60)
    print(f"Loss Recovered Score: {results['loss_recovered']:.4f}")
    print(f"H_org (original):     {results['ce_loss_without_transcoder']:.4f}")
    print(f"H_phi (transcoder):   {results['ce_loss_with_transcoder']:.4f}")
    print(f"H_0 (ablation):       {results['ce_loss_with_ablation']:.4f}")
    print("="*60)
    
    if results['loss_recovered'] > 1.0:
        print("\n⚠️  Warning: Loss Recovered > 1.0 (transcoder performs better than original)")
    elif results['loss_recovered'] < 0.0:
        print("\n⚠️  Warning: Loss Recovered < 0.0 (transcoder performs worse than ablation)")
    elif results['loss_recovered'] > 0.9:
        print("\n✓ Excellent: Loss Recovered > 0.9 (transcoder preserves model behavior very well)")
    elif results['loss_recovered'] > 0.7:
        print("\n✓ Good: Loss Recovered > 0.7 (transcoder preserves model behavior well)")
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()

