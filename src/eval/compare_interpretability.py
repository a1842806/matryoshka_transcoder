"""
Comprehensive Interpretability Comparison Script.

Compare your Matryoshka Transcoder against Google's Gemma Scope transcoder
using trusted interpretability metrics:

1. Absorption Score (SAE Bench standard)
2. Feature Utilization (Dead neurons)
3. Sparsity Analysis
4. Reconstruction Quality

Usage:
    python compare_interpretability.py \
        --ours_checkpoint checkpoints/transcoder/gemma-2-2b/my_model_15000 \
        --google_dir /path/to/gemma_scope_2b \
        --layer 17 \
        --dataset HuggingFaceFW/fineweb-edu \
        --batches 1000 \
        --device cuda:0
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.sae import MatryoshkaTranscoder
from src.models.transcoder_activation_store import TranscoderActivationsStore, create_transcoder_config
from src.adapters.gemma_scope_transcoder import GemmaScopeTranscoderAdapter
from src.utils.config import get_default_cfg
from src.eval.interpretability_metrics import (
    InterpretabilityEvaluator,
    InterpretabilityMetrics,
    save_metrics,
    print_metrics_summary
)


def load_our_transcoder(
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype
) -> MatryoshkaTranscoder:
    """Load our Matryoshka transcoder from checkpoint."""
    print(f"\n📥 Loading our transcoder from: {checkpoint_path}")
    
    # Load config
    config_path = os.path.join(checkpoint_path, "config.json")
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    
    # Fix data types
    if isinstance(cfg.get("dtype"), str):
        try:
            cfg["dtype"] = getattr(torch, cfg["dtype"].split(".")[-1])
        except:
            cfg["dtype"] = dtype
    
    if isinstance(cfg.get("model_dtype"), str):
        try:
            cfg["model_dtype"] = getattr(torch, cfg["model_dtype"].split(".")[-1])
        except:
            cfg["model_dtype"] = dtype
    
    # Fix prefix_sizes (handles both old group_sizes and new prefix_sizes naming)
    if isinstance(cfg.get("prefix_sizes"), str):
        try:
            cfg["prefix_sizes"] = json.loads(cfg["prefix_sizes"])
        except:
            import ast
            cfg["prefix_sizes"] = ast.literal_eval(cfg["prefix_sizes"])
    elif isinstance(cfg.get("group_sizes"), str):
        # Legacy support for old naming
        try:
            cfg["prefix_sizes"] = json.loads(cfg["group_sizes"])
        except:
            import ast
            cfg["prefix_sizes"] = ast.literal_eval(cfg["group_sizes"])
    
    # Ensure numeric fields are correct type
    for key in ["dict_size", "top_k", "top_k_aux", "n_batches_to_dead"]:
        if key in cfg:
            cfg[key] = int(cfg[key])
    
    for key in ["lr", "aux_penalty", "min_lr"]:
        if key in cfg and isinstance(cfg[key], str) and "/" in cfg[key]:
            num, den = cfg[key].split("/")
            cfg[key] = float(num) / float(den)
        elif key in cfg:
            cfg[key] = float(cfg[key])
    
    cfg["device"] = device
    cfg["dtype"] = dtype
    
    # Create transcoder
    transcoder = MatryoshkaTranscoder(cfg).to(device=device, dtype=dtype)
    
    # Load state dict
    state_path = os.path.join(checkpoint_path, "sae.pt")
    transcoder.load_state_dict(torch.load(state_path, map_location=device))
    transcoder.eval()
    
    print(f"✓ Transcoder loaded successfully")
    print(f"  - Dictionary size: {cfg['dict_size']:,}")
    print(f"  - Prefix sizes: {cfg['prefix_sizes']}")
    print(f"  - Top-k: {cfg['top_k']}")
    
    return transcoder


def load_google_transcoder(
    google_dir: str,
    layer: int,
    device: torch.device,
    dtype: torch.dtype
) -> GemmaScopeTranscoderAdapter:
    """Load Google's Gemma Scope transcoder."""
    print(f"\n📥 Loading Google's transcoder from: {google_dir}")
    
    transcoder = GemmaScopeTranscoderAdapter(
        layer=layer,
        repo_dir=google_dir,
        device=device,
        dtype=dtype
    )
    
    print(f"✓ Google transcoder loaded successfully")
    print(f"  - Layer: {layer}")
    print(f"  - Weight shape: {transcoder.W.shape}")
    
    return transcoder


@torch.no_grad()
def evaluate_model_on_batches(
    transcoder,
    activation_store: TranscoderActivationsStore,
    evaluator: InterpretabilityEvaluator,
    num_batches: int,
    is_google: bool = False,
) -> InterpretabilityMetrics:
    """
    Evaluate a transcoder over multiple batches.
    
    Args:
        transcoder: Model to evaluate
        activation_store: Source of activation pairs
        evaluator: Interpretability evaluator
        num_batches: Number of batches to evaluate
        is_google: If True, transcoder is Google's (no sparse features)
        
    Returns:
        Aggregated InterpretabilityMetrics
    """
    print(f"\n🔍 Evaluating over {num_batches} batches...")
    
    # Accumulators for reconstruction metrics
    total_mse = 0.0
    total_mae = 0.0
    total_fvu = 0.0
    total_cos = 0.0
    total_samples = 0
    
    # For sparsity (only for ours)
    all_l0_values = []
    
    # Collect activations for absorption score
    all_activations = []
    
    for i in tqdm(range(num_batches), desc="Evaluating"):
        source_batch, target_batch = activation_store.next_batch()
        source_batch = source_batch.to(transcoder.W.device if is_google else transcoder.W_dec.device)
        target_batch = target_batch.to(source_batch.device)
        
        batch_size = source_batch.shape[0]
        
        if is_google:
            # Google transcoder: direct mapping, no sparse features
            reconstruction = transcoder(source_batch)
            
            # Reconstruction metrics
            mse = F.mse_loss(reconstruction, target_batch).item()
            mae = (reconstruction - target_batch).abs().mean().item()
            
            # FVU
            residuals = reconstruction - target_batch
            var_res = residuals.pow(2).mean().item()
            var_orig = target_batch.var().item()
            fvu = var_res / (var_orig + 1e-8)
            
            # Cosine similarity
            cos = F.cosine_similarity(
                reconstruction.flatten(),
                target_batch.flatten(),
                dim=0
            ).item()
            
        else:
            # Our transcoder: has sparse features
            sparse_features = transcoder.encode(source_batch)
            reconstruction = transcoder.decode(sparse_features)
            
            # Reconstruction metrics
            mse = F.mse_loss(reconstruction, target_batch).item()
            mae = (reconstruction - target_batch).abs().mean().item()
            
            # FVU
            residuals = reconstruction - target_batch
            var_res = residuals.pow(2).mean().item()
            var_orig = target_batch.var().item()
            fvu = var_res / (var_orig + 1e-8)
            
            # Cosine similarity
            cos = F.cosine_similarity(
                reconstruction.flatten(),
                target_batch.flatten(),
                dim=0
            ).item()
            
            # L0 sparsity
            l0_per_sample = (sparse_features > 0).sum(dim=-1).float()
            all_l0_values.extend(l0_per_sample.cpu().tolist())
            
            # Collect activations for later analysis
            all_activations.append(sparse_features.cpu())
        
        # Accumulate
        total_mse += mse * batch_size
        total_mae += mae * batch_size
        total_fvu += fvu * batch_size
        total_cos += cos * batch_size
        total_samples += batch_size
    
    # Compute final metrics
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples
    avg_fvu = total_fvu / total_samples
    avg_cos = total_cos / total_samples
    
    # Absorption score
    if not is_google:
        feature_vectors = transcoder.W_dec.detach()
        absorption_score, absorption_details = evaluator.compute_absorption_score(
            feature_vectors,
            return_details=True
        )
        absorption_pairs = absorption_details["pairs_above_threshold"]
        
        # Feature utilization
        util_stats = evaluator.compute_feature_utilization(transcoder.num_batches_not_active)
        
        # Sparsity stats
        l0_tensor = torch.tensor(all_l0_values)
        mean_l0 = l0_tensor.mean().item()
        median_l0 = l0_tensor.median().item()
        l0_std = l0_tensor.std().item()
        
        # Activation stats
        all_activations_cat = torch.cat(all_activations, dim=0)
        max_activation = all_activations_cat.max().item()
        mean_activation = all_activations_cat[all_activations_cat > 0].mean().item() if (all_activations_cat > 0).any() else 0.0
        
    else:
        # Google transcoder doesn't have sparse features
        absorption_score = float('nan')
        absorption_pairs = 0
        util_stats = {
            "dead_count": 0,
            "dead_percentage": 0.0,
            "alive_count": 0,
            "total_features": 0,
        }
        mean_l0 = float('nan')
        median_l0 = float('nan')
        l0_std = float('nan')
        max_activation = float('nan')
        mean_activation = float('nan')
    
    return InterpretabilityMetrics(
        absorption_score=absorption_score,
        absorption_pairs_count=absorption_pairs,
        absorption_threshold=evaluator.absorption_threshold,
        dead_features_count=util_stats["dead_count"],
        dead_features_percentage=util_stats["dead_percentage"],
        alive_features_count=util_stats["alive_count"],
        total_features=util_stats["total_features"],
        mean_l0=mean_l0,
        median_l0=median_l0,
        l0_std=l0_std,
        mse=avg_mse,
        mae=avg_mae,
        fvu=avg_fvu,
        cosine_similarity=avg_cos,
        max_feature_activation=max_activation,
        mean_feature_activation=mean_activation,
    )


def create_comparison_report(
    ours_metrics: InterpretabilityMetrics,
    google_metrics: InterpretabilityMetrics,
    output_dir: str
):
    """Create detailed comparison report."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comparison dict
    comparison = {
        "summary": {
            "interpretation": "Lower is better for FVU and Absorption. Higher is better for cosine similarity and alive features."
        },
        "ours": ours_metrics.to_dict(),
        "google": google_metrics.to_dict(),
        "comparison": {
            "fvu_difference": ours_metrics.fvu - google_metrics.fvu,
            "fvu_winner": "ours" if ours_metrics.fvu < google_metrics.fvu else "google",
            "absorption_difference": ours_metrics.absorption_score - google_metrics.absorption_score if not np.isnan(ours_metrics.absorption_score) else None,
            "absorption_winner": "ours" if ours_metrics.absorption_score < google_metrics.absorption_score else "google" if not np.isnan(ours_metrics.absorption_score) else "N/A (Google has no sparse features)",
            "cosine_sim_difference": ours_metrics.cosine_similarity - google_metrics.cosine_similarity,
            "cosine_sim_winner": "ours" if ours_metrics.cosine_similarity > google_metrics.cosine_similarity else "google",
        }
    }
    
    # Save JSON
    json_path = os.path.join(output_dir, "comparison.json")
    with open(json_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n✅ Comparison saved to: {json_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("📊 COMPARISON SUMMARY")
    print("="*80)
    
    print("\n🎯 RECONSTRUCTION QUALITY (FVU - Lower is Better)")
    print(f"  Ours:   {ours_metrics.fvu:.4f}")
    print(f"  Google: {google_metrics.fvu:.4f}")
    print(f"  Winner: {'✨ OURS ✨' if ours_metrics.fvu < google_metrics.fvu else '🏆 GOOGLE'} (Δ = {abs(ours_metrics.fvu - google_metrics.fvu):.4f})")
    
    if not np.isnan(ours_metrics.absorption_score):
        print("\n📊 INTERPRETABILITY (Absorption - Lower is Better)")
        print(f"  Ours:   {ours_metrics.absorption_score:.4f}")
        print(f"  Google: N/A (no sparse features)")
        print(f"  Note: Only our model has interpretable sparse features")
    
    print("\n✨ SPARSITY (Our Model Only)")
    print(f"  Mean L0: {ours_metrics.mean_l0:.1f}")
    print(f"  Dead features: {ours_metrics.dead_features_percentage:.1f}%")
    
    print("\n🎭 COSINE SIMILARITY (Higher is Better)")
    print(f"  Ours:   {ours_metrics.cosine_similarity:.4f}")
    print(f"  Google: {google_metrics.cosine_similarity:.4f}")
    print(f"  Winner: {'✨ OURS ✨' if ours_metrics.cosine_similarity > google_metrics.cosine_similarity else '🏆 GOOGLE'} (Δ = {abs(ours_metrics.cosine_similarity - google_metrics.cosine_similarity):.4f})")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare interpretability metrics between our transcoder and Google's"
    )
    parser.add_argument("--ours_checkpoint", type=str, required=True,
                       help="Path to our transcoder checkpoint directory")
    parser.add_argument("--google_dir", type=str, required=True,
                       help="Path to Google Gemma Scope transcoder directory")
    parser.add_argument("--layer", type=int, default=17,
                       help="Layer to evaluate (default: 17)")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu",
                       help="Dataset for evaluation")
    parser.add_argument("--batches", type=int, default=1000,
                       help="Number of batches to evaluate")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size for evaluation")
    parser.add_argument("--seq_len", type=int, default=64,
                       help="Sequence length")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                       help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                       choices=["float16", "bfloat16", "float32"],
                       help="Data type")
    parser.add_argument("--output_dir", type=str, default="analysis_results/interpretability_comparison",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    device = torch.device(args.device)
    dtype = dtype_map[args.dtype]
    
    print("="*80)
    print("🔬 INTERPRETABILITY COMPARISON")
    print("="*80)
    print(f"Layer: {args.layer}")
    print(f"Device: {device}")
    print(f"Batches: {args.batches}")
    print(f"Dataset: {args.dataset}")
    
    # Load model
    print(f"\n📥 Loading Gemma-2-2B model...")
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=device, dtype=dtype)
    print("✓ Model loaded")
    
    # Create activation store config
    cfg = get_default_cfg()
    cfg["model_name"] = "gemma-2-2b"
    cfg["device"] = device
    cfg["dtype"] = dtype
    cfg["dataset_path"] = args.dataset
    cfg["batch_size"] = args.batch_size
    cfg["seq_len"] = args.seq_len
    cfg["model_batch_size"] = 2
    cfg["num_batches_in_buffer"] = 1
    
    cfg = create_transcoder_config(
        cfg,
        source_layer=args.layer,
        target_layer=args.layer,
        source_site="mlp_in",
        target_site="mlp_out"
    )
    
    # Ensure correct sizes
    cfg["source_act_size"] = 2304
    cfg["target_act_size"] = 2304
    cfg["act_size"] = 2304
    
    # Create activation store
    print(f"\n📊 Creating activation store...")
    activation_store = TranscoderActivationsStore(model, cfg)
    print("✓ Activation store created")
    
    # Load transcoders
    our_transcoder = load_our_transcoder(args.ours_checkpoint, device, dtype)
    google_transcoder = load_google_transcoder(args.google_dir, args.layer, device, dtype)
    
    # Create evaluator
    evaluator = InterpretabilityEvaluator(
        absorption_threshold=0.9,
        absorption_subset_size=8192,
        dead_threshold=20
    )
    
    # Evaluate our transcoder
    print("\n" + "="*80)
    print("📊 EVALUATING OUR TRANSCODER")
    print("="*80)
    ours_metrics = evaluate_model_on_batches(
        our_transcoder,
        activation_store,
        evaluator,
        args.batches,
        is_google=False
    )
    print_metrics_summary(ours_metrics, "Our Matryoshka Transcoder")
    
    # Evaluate Google's transcoder
    print("\n" + "="*80)
    print("📊 EVALUATING GOOGLE'S TRANSCODER")
    print("="*80)
    google_metrics = evaluate_model_on_batches(
        google_transcoder,
        activation_store,
        evaluator,
        args.batches,
        is_google=True
    )
    print_metrics_summary(google_metrics, "Google Gemma Scope Transcoder")
    
    # Create comparison report
    create_comparison_report(ours_metrics, google_metrics, args.output_dir)
    
    # Save individual metrics
    save_metrics(ours_metrics, os.path.join(args.output_dir, "ours_metrics.json"))
    save_metrics(google_metrics, os.path.join(args.output_dir, "google_metrics.json"))
    
    print(f"\n✅ All results saved to: {args.output_dir}")


if __name__ == "__main__":
    import numpy as np
    main()

