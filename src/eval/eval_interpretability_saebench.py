"""
SAEBench-Style Interpretability Evaluation for Transcoders.

Evaluates and compares transcoders using core interpretability metrics from SAEBench:
1. Absorption Score - measures feature redundancy (lower is better)
2. Reconstruction Quality - FVU, MSE, cosine similarity
3. Feature Sparsity - L0 norm distribution
4. Feature Utilization - dead vs alive features

Reference: SAEBench (https://github.com/adamkarvonen/SAEBench)

Usage:
    python eval_interpretability_saebench.py \
        --ours_checkpoint checkpoints/transcoder/gemma-2-2b/my_model_15000 \
        --google_dir /path/to/gemma_scope_2b \
        --layer 17 \
        --dataset HuggingFaceFW/fineweb-edu \
        --batches 1000 \
        --output_dir analysis_results/saebench_comparison
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.sae import MatryoshkaTranscoder
from src.models.transcoder_activation_store import TranscoderActivationsStore, create_transcoder_config
from src.adapters.gemma_scope_transcoder import GemmaScopeTranscoderAdapter
from src.utils.config import get_default_cfg


@dataclass
class InterpretabilityMetrics:
    """Core interpretability metrics from SAEBench."""
    
    # Absorption Score (SAEBench standard)
    absorption_score: float
    absorption_pairs_above_threshold: int
    absorption_threshold: float
    
    # Reconstruction Quality
    fvu: float  # Fraction of Variance Unexplained (0=perfect, lower is better)
    mse: float
    mae: float
    cosine_similarity: float
    
    # Sparsity (only for models with sparse features)
    mean_l0: Optional[float]
    median_l0: Optional[float]
    l0_std: Optional[float]
    
    # Feature Utilization (only for models with sparse features)
    dead_features_count: Optional[int]
    dead_features_percentage: Optional[float]
    alive_features_count: Optional[int]
    total_features: Optional[int]
    
    # Additional stats
    max_feature_activation: Optional[float]
    mean_feature_activation: Optional[float]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SAEBenchEvaluator:
    """
    Evaluator implementing SAEBench methodology for interpretability metrics.
    
    Core metric: Absorption Score
    - Measures feature redundancy via pairwise cosine similarity
    - Lower score = more diverse, interpretable features
    - Standard threshold: 0.9 (from SAEBench)
    """
    
    def __init__(
        self,
        absorption_threshold: float = 0.9,
        absorption_subset_size: int = 8192,
        dead_threshold: int = 20
    ):
        """
        Initialize SAEBench evaluator.
        
        Args:
            absorption_threshold: Cosine similarity threshold (SAEBench standard: 0.9)
            absorption_subset_size: Max features to sample for absorption (memory efficiency)
            dead_threshold: Consecutive inactive batches to mark feature as dead
        """
        self.absorption_threshold = absorption_threshold
        self.absorption_subset_size = absorption_subset_size
        self.dead_threshold = dead_threshold
    
    @torch.no_grad()
    def compute_absorption_score(
        self, 
        feature_vectors: torch.Tensor
    ) -> Tuple[float, int]:
        """
        Compute Absorption Score following SAEBench methodology.
        
        Measures feature redundancy by computing fraction of feature pairs
        with high cosine similarity (>threshold).
        
        Lower absorption = better (features are diverse and interpretable)
        Higher absorption = worse (features are redundant)
        
        Args:
            feature_vectors: Decoder weight matrix (n_features, d_model)
            
        Returns:
            absorption_score: Fraction of pairs with similarity > threshold
            pairs_above_threshold: Count of redundant pairs
        """
        if feature_vectors is None or feature_vectors.numel() == 0:
            return float('nan'), 0
        
        # L2-normalize feature vectors
        V = F.normalize(feature_vectors, p=2, dim=-1).float()
        
        # Subsample if too large (memory efficiency)
        if V.shape[0] > self.absorption_subset_size:
            idx = torch.randperm(V.shape[0])[:self.absorption_subset_size]
            V = V[idx]
        
        n_features = V.shape[0]
        
        # Compute pairwise cosine similarity matrix
        # S[i,j] = cosine_similarity(V[i], V[j])
        S = torch.abs(V @ V.T)  # Absolute value to catch anti-correlation
        
        # Remove diagonal (self-similarity)
        mask = ~torch.eye(n_features, dtype=torch.bool, device=S.device)
        similarities = S[mask]
        
        # Count pairs above threshold
        above_threshold = (similarities > self.absorption_threshold).sum().item()
        total_pairs = similarities.numel()
        
        # Absorption score: fraction of pairs with high similarity
        absorption_score = above_threshold / max(1, total_pairs)
        
        return absorption_score, above_threshold
    
    @torch.no_grad()
    def compute_reconstruction_metrics(
        self,
        original: torch.Tensor,
        reconstruction: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute reconstruction quality metrics.
        
        Args:
            original: Original target activations
            reconstruction: Reconstructed activations
            
        Returns:
            Dictionary with MSE, MAE, FVU, and cosine similarity
        """
        # Flatten to (N, D) for consistent computation
        orig_flat = original.reshape(-1, original.shape[-1]).float()
        recon_flat = reconstruction.reshape(-1, reconstruction.shape[-1]).float()
        
        # MSE and MAE
        mse = F.mse_loss(recon_flat, orig_flat).item()
        mae = (recon_flat - orig_flat).abs().mean().item()
        
        # FVU (Fraction of Variance Unexplained)
        # FVU = Var(residuals) / Var(original)
        # Lower is better (0 = perfect reconstruction)
        residuals = recon_flat - orig_flat
        var_residuals = residuals.pow(2).mean().item()
        var_original = orig_flat.var().item()
        fvu = var_residuals / (var_original + 1e-8)
        
        # Cosine similarity (averaged over all samples)
        cos_sim = F.cosine_similarity(
            recon_flat, 
            orig_flat, 
            dim=-1
        ).mean().item()
        
        return {
            "mse": mse,
            "mae": mae,
            "fvu": fvu,
            "cosine_similarity": cos_sim,
        }
    
    @torch.no_grad()
    def evaluate_transcoder(
        self,
        transcoder,
        activation_store: TranscoderActivationsStore,
        num_batches: int,
        is_google: bool = False
    ) -> InterpretabilityMetrics:
        """
        Evaluate a transcoder using SAEBench metrics.
        
        Args:
            transcoder: Model to evaluate
            activation_store: Source of activation pairs
            num_batches: Number of batches to evaluate
            is_google: If True, transcoder is Google's (no sparse features)
            
        Returns:
            InterpretabilityMetrics with all computed metrics
        """
        print(f"\n{'='*80}")
        print(f"Evaluating over {num_batches} batches...")
        print(f"{'='*80}")
        
        # Accumulators for reconstruction metrics
        total_mse = 0.0
        total_mae = 0.0
        total_fvu = 0.0
        total_cos = 0.0
        total_samples = 0
        
        # For sparsity (only for ours)
        all_l0_values = []
        all_activations = []
        
        for i in tqdm(range(num_batches), desc="Evaluating"):
            source_batch, target_batch = activation_store.next_batch()
            source_batch = source_batch.to(transcoder.W.device if is_google else transcoder.W_dec.device)
            target_batch = target_batch.to(source_batch.device)
            
            batch_size = source_batch.shape[0]
            
            if is_google:
                # Google transcoder: direct mapping, no sparse features
                reconstruction = transcoder(source_batch)
            else:
                # Our transcoder: has sparse features
                sparse_features = transcoder.encode(source_batch)
                reconstruction = transcoder.decode(sparse_features)
                
                # L0 sparsity
                l0_per_sample = (sparse_features > 0).sum(dim=-1).float()
                all_l0_values.extend(l0_per_sample.cpu().tolist())
                
                # Collect activations
                all_activations.append(sparse_features.cpu())
            
            # Reconstruction metrics
            recon_metrics = self.compute_reconstruction_metrics(target_batch, reconstruction)
            
            # Accumulate
            total_mse += recon_metrics["mse"] * batch_size
            total_mae += recon_metrics["mae"] * batch_size
            total_fvu += recon_metrics["fvu"] * batch_size
            total_cos += recon_metrics["cosine_similarity"] * batch_size
            total_samples += batch_size
        
        # Compute final reconstruction metrics
        avg_mse = total_mse / total_samples
        avg_mae = total_mae / total_samples
        avg_fvu = total_fvu / total_samples
        avg_cos = total_cos / total_samples
        
        # Compute absorption score and feature stats
        if not is_google:
            # Our model has sparse features
            feature_vectors = transcoder.W_dec.detach()
            absorption_score, absorption_pairs = self.compute_absorption_score(feature_vectors)
            
            # Feature utilization
            dead_mask = transcoder.num_batches_not_active >= self.dead_threshold
            dead_count = dead_mask.sum().item()
            total_features = transcoder.num_batches_not_active.numel()
            alive_count = total_features - dead_count
            dead_percentage = 100.0 * dead_count / max(1, total_features)
            
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
            dead_count = None
            dead_percentage = None
            alive_count = None
            total_features = None
            mean_l0 = None
            median_l0 = None
            l0_std = None
            max_activation = None
            mean_activation = None
        
        return InterpretabilityMetrics(
            absorption_score=absorption_score,
            absorption_pairs_above_threshold=absorption_pairs,
            absorption_threshold=self.absorption_threshold,
            fvu=avg_fvu,
            mse=avg_mse,
            mae=avg_mae,
            cosine_similarity=avg_cos,
            mean_l0=mean_l0,
            median_l0=median_l0,
            l0_std=l0_std,
            dead_features_count=dead_count,
            dead_features_percentage=dead_percentage,
            alive_features_count=alive_count,
            total_features=total_features,
            max_feature_activation=max_activation,
            mean_feature_activation=mean_activation,
        )


def load_our_transcoder(
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype
) -> MatryoshkaTranscoder:
    """Load our Matryoshka transcoder from checkpoint."""
    print(f"\n📥 Loading Matryoshka Transcoder from: {checkpoint_path}")
    
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
    
    # Fix prefix_sizes
    if isinstance(cfg.get("prefix_sizes"), str):
        try:
            cfg["prefix_sizes"] = json.loads(cfg["prefix_sizes"])
        except:
            import ast
            cfg["prefix_sizes"] = ast.literal_eval(cfg["prefix_sizes"])
    elif isinstance(cfg.get("group_sizes"), str):
        try:
            cfg["prefix_sizes"] = json.loads(cfg["group_sizes"])
        except:
            import ast
            cfg["prefix_sizes"] = ast.literal_eval(cfg["group_sizes"])
    
    # Ensure numeric fields
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
    if not os.path.exists(state_path):
        state_path = os.path.join(checkpoint_path, "checkpoint.pt")
    transcoder.load_state_dict(torch.load(state_path, map_location=device))
    transcoder.eval()
    
    print(f"✓ Matryoshka Transcoder loaded")
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
    print(f"\n📥 Loading Google's Transcoder from: {google_dir}")
    
    transcoder = GemmaScopeTranscoderAdapter(
        layer=layer,
        repo_dir=google_dir,
        device=device,
        dtype=dtype
    )
    
    print(f"✓ Google Transcoder loaded")
    print(f"  - Layer: {layer}")
    print(f"  - Weight shape: {transcoder.W.shape}")
    
    return transcoder


def print_comparison_report(
    ours_metrics: InterpretabilityMetrics,
    google_metrics: InterpretabilityMetrics
):
    """Print detailed comparison report."""
    print("\n" + "="*80)
    print("📊 SAEBench INTERPRETABILITY COMPARISON")
    print("="*80)
    
    print("\n🎯 RECONSTRUCTION QUALITY")
    print("-"*80)
    print(f"Metric                  Ours          Google        Winner")
    print("-"*80)
    print(f"FVU (lower=better)      {ours_metrics.fvu:.6f}    {google_metrics.fvu:.6f}    {'✨ Ours' if ours_metrics.fvu < google_metrics.fvu else '🏆 Google'}")
    print(f"MSE                     {ours_metrics.mse:.6f}    {google_metrics.mse:.6f}    {'✨ Ours' if ours_metrics.mse < google_metrics.mse else '🏆 Google'}")
    print(f"Cosine Sim (higher=better) {ours_metrics.cosine_similarity:.6f}    {google_metrics.cosine_similarity:.6f}    {'✨ Ours' if ours_metrics.cosine_similarity > google_metrics.cosine_similarity else '🏆 Google'}")
    
    print("\n📊 ABSORPTION SCORE (SAEBench Core Metric)")
    print("-"*80)
    print(f"Absorption measures feature redundancy (lower = better interpretability)")
    print(f"Standard threshold: {ours_metrics.absorption_threshold}")
    print()
    if not np.isnan(ours_metrics.absorption_score):
        print(f"Ours:   {ours_metrics.absorption_score:.6f}")
        print(f"        ({ours_metrics.absorption_pairs_above_threshold:,} pairs above threshold)")
        if ours_metrics.absorption_score < 0.01:
            print(f"        ✅ EXCELLENT - Features are diverse and interpretable")
        elif ours_metrics.absorption_score < 0.05:
            print(f"        ⚠️  MODERATE - Some feature redundancy present")
        else:
            print(f"        ❌ HIGH REDUNDANCY - Features are not diverse")
    else:
        print(f"Ours:   N/A (no sparse features)")
    
    print()
    print(f"Google: N/A (direct mapping, no sparse features)")
    print(f"        Note: Google's transcoder is not a sparse autoencoder,")
    print(f"              so absorption score is not applicable.")
    
    if ours_metrics.mean_l0 is not None:
        print("\n✨ SPARSITY (Matryoshka Transcoder Only)")
        print("-"*80)
        print(f"Mean L0:    {ours_metrics.mean_l0:.1f}")
        print(f"Median L0:  {ours_metrics.median_l0:.1f}")
        print(f"Std L0:     {ours_metrics.l0_std:.1f}")
        print(f"Dead features: {ours_metrics.dead_features_count} / {ours_metrics.total_features} ({ours_metrics.dead_features_percentage:.1f}%)")
        
        if ours_metrics.dead_features_percentage < 10:
            print(f"✅ EXCELLENT feature utilization")
        elif ours_metrics.dead_features_percentage < 30:
            print(f"⚠️  MODERATE feature utilization")
        else:
            print(f"❌ POOR feature utilization")
    
    print("\n" + "="*80)
    print("📝 INTERPRETATION GUIDE")
    print("="*80)
    print("Absorption Score (SAEBench core metric):")
    print("  < 0.01: Excellent - Features are diverse and interpretable")
    print("  < 0.05: Moderate - Some redundancy, but acceptable")
    print("  ≥ 0.05: Poor - High feature redundancy")
    print()
    print("FVU (Reconstruction quality):")
    print("  < 0.1:  Excellent reconstruction")
    print("  < 0.3:  Moderate reconstruction")
    print("  ≥ 0.3:  Poor reconstruction")
    print("="*80)


def save_comparison(
    ours_metrics: InterpretabilityMetrics,
    google_metrics: InterpretabilityMetrics,
    output_dir: str
):
    """Save comparison results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    comparison = {
        "saebench_metrics": {
            "absorption_score": {
                "description": "Measures feature redundancy via pairwise cosine similarity. Lower is better.",
                "threshold": ours_metrics.absorption_threshold,
                "interpretation": "< 0.01 = excellent, < 0.05 = moderate, >= 0.05 = poor"
            },
            "reconstruction_quality": {
                "description": "Measures how well the transcoder reconstructs target activations.",
                "fvu_interpretation": "< 0.1 = excellent, < 0.3 = moderate, >= 0.3 = poor"
            }
        },
        "matryoshka_transcoder": ours_metrics.to_dict(),
        "google_transcoder": google_metrics.to_dict(),
        "comparison": {
            "fvu_difference": ours_metrics.fvu - google_metrics.fvu,
            "fvu_winner": "matryoshka" if ours_metrics.fvu < google_metrics.fvu else "google",
            "cosine_sim_difference": ours_metrics.cosine_similarity - google_metrics.cosine_similarity,
            "cosine_sim_winner": "matryoshka" if ours_metrics.cosine_similarity > google_metrics.cosine_similarity else "google",
            "absorption_score_available": not np.isnan(ours_metrics.absorption_score),
            "note": "Google's transcoder is a direct mapping without sparse features, so absorption score is not applicable."
        }
    }
    
    output_path = os.path.join(output_dir, "saebench_comparison.json")
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SAEBench-style interpretability evaluation for transcoders"
    )
    parser.add_argument("--ours_checkpoint", type=str, required=True,
                       help="Path to Matryoshka transcoder checkpoint")
    parser.add_argument("--google_dir", type=str, required=True,
                       help="Path to Google Gemma Scope transcoder directory")
    parser.add_argument("--layer", type=int, default=17,
                       help="Layer to evaluate (default: 17)")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu",
                       help="Dataset for evaluation")
    parser.add_argument("--batches", type=int, default=1000,
                       help="Number of batches to evaluate (default: 1000)")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size for evaluation")
    parser.add_argument("--seq_len", type=int, default=64,
                       help="Sequence length")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                       help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                       choices=["float16", "bfloat16", "float32"],
                       help="Data type")
    parser.add_argument("--output_dir", type=str, default="analysis_results/saebench_comparison",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    device = torch.device(args.device)
    dtype = dtype_map[args.dtype]
    
    print("="*80)
    print("🔬 SAEBench INTERPRETABILITY EVALUATION")
    print("="*80)
    print(f"Layer: {args.layer}")
    print(f"Device: {device}")
    print(f"Batches: {args.batches}")
    print(f"Dataset: {args.dataset}")
    
    # Load model
    print(f"\n📥 Loading Gemma-2-2B model...")
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=device, dtype=dtype)
    print("✓ Model loaded")
    
    # Create activation store
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
    
    cfg["source_act_size"] = 2304
    cfg["target_act_size"] = 2304
    cfg["act_size"] = 2304
    
    print(f"\n📊 Creating activation store...")
    activation_store = TranscoderActivationsStore(model, cfg)
    print("✓ Activation store created")
    
    # Load transcoders
    our_transcoder = load_our_transcoder(args.ours_checkpoint, device, dtype)
    google_transcoder = load_google_transcoder(args.google_dir, args.layer, device, dtype)
    
    # Create evaluator
    evaluator = SAEBenchEvaluator(
        absorption_threshold=0.9,  # SAEBench standard
        absorption_subset_size=8192,
        dead_threshold=20
    )
    
    # Evaluate Matryoshka transcoder
    print("\n" + "="*80)
    print("📊 EVALUATING MATRYOSHKA TRANSCODER")
    print("="*80)
    ours_metrics = evaluator.evaluate_transcoder(
        our_transcoder,
        activation_store,
        args.batches,
        is_google=False
    )
    
    # Evaluate Google's transcoder
    print("\n" + "="*80)
    print("📊 EVALUATING GOOGLE'S TRANSCODER")
    print("="*80)
    google_metrics = evaluator.evaluate_transcoder(
        google_transcoder,
        activation_store,
        args.batches,
        is_google=True
    )
    
    # Print comparison
    print_comparison_report(ours_metrics, google_metrics)
    
    # Save results
    save_comparison(ours_metrics, google_metrics, args.output_dir)
    
    print(f"\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()

