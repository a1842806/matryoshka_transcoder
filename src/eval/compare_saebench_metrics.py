"""
Compare SAEBench Metrics: Matryoshka vs Google's Transcoder

This script runs the simplified SAEBench evaluation on both:
1. Your Matryoshka transcoder (layer 8)
2. Google's Gemma Scope transcoder (layer 8)

And provides a side-by-side comparison of interpretability metrics.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any
import numpy as np

import torch
from transformer_lens import HookedTransformer

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.sae import MatryoshkaTranscoder
from src.models.transcoder_activation_store import TranscoderActivationsStore, create_transcoder_config
from src.adapters.gemma_scope_transcoder import GemmaScopeTranscoderAdapter
from src.utils.config import get_default_cfg
from src.eval.eval_saebench_simple import SimpleSAEBenchEvaluator, SimpleSAEBenchMetrics


def load_matryoshka_transcoder(checkpoint_path: str, device: torch.device, dtype: torch.dtype):
    """Load Matryoshka transcoder from checkpoint."""
    print(f"\n📥 Loading Matryoshka transcoder from: {checkpoint_path}")
    
    config_path = os.path.join(checkpoint_path, "config.json")
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    
    # Fix data types
    if isinstance(cfg.get("dtype"), str):
        try:
            cfg["dtype"] = getattr(torch, cfg["dtype"].split(".")[-1])
        except:
            cfg["dtype"] = dtype
    
    # Fix prefix_sizes
    if isinstance(cfg.get("prefix_sizes"), str):
        try:
            cfg["prefix_sizes"] = json.loads(cfg["prefix_sizes"])
        except:
            import ast
            cfg["prefix_sizes"] = ast.literal_eval(cfg["prefix_sizes"])
    
    # Ensure numeric fields
    for key in ["dict_size", "top_k"]:
        if key in cfg:
            cfg[key] = int(cfg[key])
    
    cfg["device"] = device
    cfg["dtype"] = dtype
    
    # Create transcoder
    transcoder = MatryoshkaTranscoder(cfg).to(device=device, dtype=dtype)
    
    # Load state dict
    state_path = os.path.join(checkpoint_path, "checkpoint.pt")
    transcoder.load_state_dict(torch.load(state_path, map_location=device))
    transcoder.eval()
    
    print(f"✓ Matryoshka transcoder loaded successfully")
    print(f"  - Dictionary size: {cfg['dict_size']:,}")
    print(f"  - Prefix sizes: {cfg['prefix_sizes']}")
    print(f"  - Top-k: {cfg['top_k']}")
    
    return transcoder, cfg


def load_google_transcoder(google_dir: str, layer: int, device: torch.device, dtype: torch.dtype):
    """Load Google's Gemma Scope transcoder."""
    print(f"\n📥 Loading Google's transcoder from: {google_dir}")
    
    # Find the best L0 match (closest to our transcoder's L0)
    l0_dirs = [d for d in os.listdir(google_dir) if d.startswith("average_l0_")]
    if not l0_dirs:
        raise FileNotFoundError(f"No L0 directories found in {google_dir}")
    
    # Sort by L0 value and pick the closest to our expected L0 (~71)
    l0_values = []
    for d in l0_dirs:
        try:
            l0_val = int(d.split("_")[-1])
            l0_values.append((l0_val, d))
        except:
            continue
    
    l0_values.sort(key=lambda x: x[0])
    target_l0 = 71  # Our transcoder's mean L0
    best_l0_dir = min(l0_values, key=lambda x: abs(x[0] - target_l0))[1]
    
    print(f"  - Selected L0 directory: {best_l0_dir} (target: {target_l0})")
    
    transcoder = GemmaScopeTranscoderAdapter(
        layer=layer,
        repo_dir=os.path.join(google_dir, best_l0_dir),
        device=device,
        dtype=dtype
    )
    
    print(f"✓ Google transcoder loaded successfully")
    print(f"  - Layer: {layer}")
    print(f"  - Weight shape: {transcoder.W.shape}")
    
    return transcoder


class GoogleTranscoderWrapper:
    """Wrapper to make Google's transcoder compatible with our evaluator."""
    
    def __init__(self, google_transcoder):
        self.transcoder = google_transcoder
        self.dict_size = 16384  # Google's transcoder size
    
    def encode(self, x):
        # Google's transcoder is just a linear mapping, no sparse features
        return self.transcoder(x)
    
    def decode(self, x):
        # For Google's transcoder, encode and decode are the same
        return x
    
    @property
    def W_dec(self):
        # Return the weight matrix as "decoder" for compatibility
        return self.transcoder.W.T  # Transpose to match expected shape


def print_comparison_results(
    matryoshka_metrics: SimpleSAEBenchMetrics,
    google_metrics: SimpleSAEBenchMetrics
):
    """Print side-by-side comparison of results."""
    print("\n" + "="*100)
    print("📊 SAEBENCH METRICS COMPARISON: MATRYOSHKA vs GOOGLE")
    print("="*100)
    
    print(f"\n🎯 RECONSTRUCTION QUALITY")
    print("-"*100)
    print(f"{'Metric':<25} {'Matryoshka':<15} {'Google':<15} {'Winner':<15}")
    print("-"*100)
    
    # FVU (lower is better)
    fvu_winner = "Matryoshka" if matryoshka_metrics.fvu < google_metrics.fvu else "Google"
    print(f"{'FVU (lower=better)':<25} {matryoshka_metrics.fvu:<15.6f} {google_metrics.fvu:<15.6f} {fvu_winner:<15}")
    
    # MSE (lower is better)
    mse_winner = "Matryoshka" if matryoshka_metrics.mse < google_metrics.mse else "Google"
    print(f"{'MSE (lower=better)':<25} {matryoshka_metrics.mse:<15.6f} {google_metrics.mse:<15.6f} {mse_winner:<15}")
    
    # MAE (lower is better)
    mae_winner = "Matryoshka" if matryoshka_metrics.mae < google_metrics.mae else "Google"
    print(f"{'MAE (lower=better)':<25} {matryoshka_metrics.mae:<15.6f} {google_metrics.mae:<15.6f} {mae_winner:<15}")
    
    # Cosine similarity (higher is better)
    cos_winner = "Matryoshka" if matryoshka_metrics.cosine_similarity > google_metrics.cosine_similarity else "Google"
    print(f"{'Cosine Similarity':<25} {matryoshka_metrics.cosine_similarity:<15.6f} {google_metrics.cosine_similarity:<15.6f} {cos_winner:<15}")
    
    print(f"\n📊 FEATURE REDUNDANCY (Simplified Absorption)")
    print("-"*100)
    print(f"{'Metric':<25} {'Matryoshka':<15} {'Google':<15} {'Winner':<15}")
    print("-"*100)
    
    # Redundancy (lower is better)
    red_winner = "Matryoshka" if matryoshka_metrics.feature_redundancy < google_metrics.feature_redundancy else "Google"
    print(f"{'Redundancy Score':<25} {matryoshka_metrics.feature_redundancy:<15.6f} {google_metrics.feature_redundancy:<15.6f} {red_winner:<15}")
    print(f"{'Similar Pairs':<25} {matryoshka_metrics.similar_pairs_count:<15,} {google_metrics.similar_pairs_count:<15,} {'N/A':<15}")
    
    print(f"\n✨ SPARSITY (Matryoshka Only - Google has no sparse features)")
    print("-"*100)
    print(f"{'Metric':<25} {'Matryoshka':<15} {'Google':<15}")
    print("-"*100)
    print(f"{'Mean L0':<25} {matryoshka_metrics.mean_l0:<15.1f} {'N/A (dense)':<15}")
    print(f"{'Median L0':<25} {matryoshka_metrics.median_l0:<15.1f} {'N/A (dense)':<15}")
    print(f"{'Std L0':<25} {matryoshka_metrics.l0_std:<15.1f} {'N/A (dense)':<15}")
    
    print(f"\n🔧 FEATURE UTILIZATION")
    print("-"*100)
    print(f"{'Metric':<25} {'Matryoshka':<15} {'Google':<15}")
    print("-"*100)
    print(f"{'Dead Features':<25} {matryoshka_metrics.dead_features_count:<15,} {'N/A (dense)':<15}")
    print(f"{'Alive Features':<25} {matryoshka_metrics.alive_features_count:<15,} {'N/A (dense)':<15}")
    print(f"{'Dead %':<25} {matryoshka_metrics.dead_features_percentage:<15.1f} {'N/A (dense)':<15}")
    
    print(f"\n📋 MODEL INFO")
    print("-"*100)
    print(f"{'Metric':<25} {'Matryoshka':<15} {'Google':<15}")
    print("-"*100)
    print(f"{'Dictionary Size':<25} {matryoshka_metrics.dict_size:<15,} {google_metrics.dict_size:<15,}")
    print(f"{'Layer':<25} {matryoshka_metrics.layer:<15} {google_metrics.layer:<15}")
    print(f"{'Model Type':<25} {'Sparse (Matryoshka)':<15} {'Dense (Linear)':<15}")
    
    print("\n" + "="*100)
    print("🎯 SUMMARY")
    print("="*100)
    
    # Count wins
    matryoshka_wins = 0
    google_wins = 0
    
    if matryoshka_metrics.fvu < google_metrics.fvu:
        matryoshka_wins += 1
    else:
        google_wins += 1
    
    if matryoshka_metrics.mse < google_metrics.mse:
        matryoshka_wins += 1
    else:
        google_wins += 1
    
    if matryoshka_metrics.mae < google_metrics.mae:
        matryoshka_wins += 1
    else:
        google_wins += 1
    
    if matryoshka_metrics.cosine_similarity > google_metrics.cosine_similarity:
        matryoshka_wins += 1
    else:
        google_wins += 1
    
    if matryoshka_metrics.feature_redundancy < google_metrics.feature_redundancy:
        matryoshka_wins += 1
    else:
        google_wins += 1
    
    print(f"Reconstruction Quality: {'Matryoshka' if matryoshka_wins > google_wins else 'Google'} ({matryoshka_wins}-{google_wins})")
    print(f"Feature Diversity: {'Matryoshka' if matryoshka_metrics.feature_redundancy < google_metrics.feature_redundancy else 'Google'}")
    print(f"Sparsity: Matryoshka only (Google is dense)")
    print(f"Interpretability: Matryoshka has sparse, interpretable features")
    
    print("\n" + "="*100)


def main():
    parser = argparse.ArgumentParser(
        description="Compare SAEBench metrics between Matryoshka and Google transcoders"
    )
    parser.add_argument("--matryoshka_checkpoint", type=str, required=True,
                       help="Path to Matryoshka transcoder checkpoint")
    parser.add_argument("--google_dir", type=str, required=True,
                       help="Path to Google Gemma Scope transcoder directory")
    parser.add_argument("--layer", type=int, default=8,
                       help="Layer to evaluate")
    parser.add_argument("--device", type=str, default="cuda:1",
                       help="Device")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                       choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output_dir", type=str, default="analysis_results/saebench_comparison",
                       help="Output directory")
    parser.add_argument("--batches", type=int, default=50,
                       help="Number of batches to evaluate")
    
    args = parser.parse_args()
    
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    device = torch.device(args.device)
    dtype = dtype_map[args.dtype]
    
    print("="*100)
    print("🔬 SAEBENCH COMPARISON: MATRYOSHKA vs GOOGLE")
    print("="*100)
    print(f"Matryoshka checkpoint: {args.matryoshka_checkpoint}")
    print(f"Google directory: {args.google_dir}")
    print(f"Layer: {args.layer}")
    print(f"Device: {device}")
    print(f"Batches: {args.batches}")
    
    # Load model
    print(f"\n📥 Loading Gemma-2-2B...")
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=device, dtype=dtype)
    print("✓ Model loaded")
    
    # Load transcoders
    matryoshka_transcoder, matryoshka_cfg = load_matryoshka_transcoder(
        args.matryoshka_checkpoint, device, dtype
    )
    
    google_transcoder = load_google_transcoder(
        args.google_dir, args.layer, device, dtype
    )
    
    # Wrap Google transcoder for compatibility
    google_wrapper = GoogleTranscoderWrapper(google_transcoder)
    
    # Create activation store
    print(f"\n📊 Creating activation store...")
    store_cfg = get_default_cfg()
    store_cfg["model_name"] = "gemma-2-2b"
    store_cfg["device"] = device
    store_cfg["dtype"] = dtype
    store_cfg["dataset_path"] = "HuggingFaceFW/fineweb-edu"
    store_cfg["batch_size"] = 256
    store_cfg["seq_len"] = 64
    store_cfg["model_batch_size"] = 2
    store_cfg["num_batches_in_buffer"] = 1
    
    store_cfg = create_transcoder_config(
        store_cfg,
        source_layer=args.layer,
        target_layer=args.layer,
        source_site="mlp_in",
        target_site="mlp_out"
    )
    
    store_cfg["source_act_size"] = 2304
    store_cfg["target_act_size"] = 2304
    store_cfg["act_size"] = 2304
    
    activation_store = TranscoderActivationsStore(model, store_cfg)
    print("✓ Activation store created")
    
    # Run evaluations
    evaluator = SimpleSAEBenchEvaluator(num_batches=args.batches)
    
    print(f"\n🔬 Evaluating Matryoshka transcoder...")
    matryoshka_metrics = evaluator.evaluate(
        matryoshka_transcoder, activation_store, args.layer, matryoshka_cfg
    )
    
    print(f"\n🔬 Evaluating Google transcoder...")
    google_metrics = evaluator.evaluate(
        google_wrapper, activation_store, args.layer, {"dict_size": 16384}
    )
    
    # Print comparison
    print_comparison_results(matryoshka_metrics, google_metrics)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {
        "matryoshka": matryoshka_metrics.to_dict(),
        "google": google_metrics.to_dict(),
        "comparison": {
            "fvu_winner": "matryoshka" if matryoshka_metrics.fvu < google_metrics.fvu else "google",
            "mse_winner": "matryoshka" if matryoshka_metrics.mse < google_metrics.mse else "google",
            "mae_winner": "matryoshka" if matryoshka_metrics.mae < google_metrics.mae else "google",
            "cosine_winner": "matryoshka" if matryoshka_metrics.cosine_similarity > google_metrics.cosine_similarity else "google",
            "redundancy_winner": "matryoshka" if matryoshka_metrics.feature_redundancy < google_metrics.feature_redundancy else "google",
        }
    }
    
    output_path = os.path.join(args.output_dir, "comparison_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    print("\n🎉 Comparison complete!")


if __name__ == "__main__":
    main()
