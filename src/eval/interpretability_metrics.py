"""
Comprehensive Interpretability Metrics for Transcoder Evaluation.

This module implements trusted interpretability metrics based on:
1. SAE Bench framework
2. Anthropic's interpretability research
3. Research consensus on reliable metrics

Key Metrics Implemented:
- Absorption Score: Measures feature redundancy (lower is better)
- Feature Utilization: Tracks dead/alive features
- Sparsity Analysis: L0 norm and activation patterns
- Reconstruction Quality: MSE, MAE, FVU, Cosine Similarity

Reference:
- SAE Bench: https://github.com/adamkarvonen/SAEBench
- Anthropic: https://transformer-circuits.pub/2023/monosemantic-features
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import os


@dataclass
class InterpretabilityMetrics:
    """Container for interpretability evaluation results."""
    
    # Absorption metrics
    absorption_score: float
    absorption_pairs_count: int
    absorption_threshold: float
    
    # Feature utilization
    dead_features_count: int
    dead_features_percentage: float
    alive_features_count: int
    total_features: int
    
    # Sparsity metrics
    mean_l0: float
    median_l0: float
    l0_std: float
    
    # Reconstruction metrics
    mse: float
    mae: float
    fvu: float
    cosine_similarity: float
    
    # Additional statistics
    max_feature_activation: float
    mean_feature_activation: float
    feature_activation_histogram: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "absorption": {
                "score": self.absorption_score,
                "pairs_above_threshold": self.absorption_pairs_count,
                "threshold": self.absorption_threshold,
            },
            "feature_utilization": {
                "dead_count": self.dead_features_count,
                "dead_percentage": self.dead_features_percentage,
                "alive_count": self.alive_features_count,
                "total_features": self.total_features,
            },
            "sparsity": {
                "mean_l0": self.mean_l0,
                "median_l0": self.median_l0,
                "l0_std": self.l0_std,
            },
            "reconstruction": {
                "mse": self.mse,
                "mae": self.mae,
                "fvu": self.fvu,
                "cosine_similarity": self.cosine_similarity,
            },
            "activation_stats": {
                "max_activation": self.max_feature_activation,
                "mean_activation": self.mean_feature_activation,
            }
        }
        
        if self.feature_activation_histogram is not None:
            result["activation_histogram"] = self.feature_activation_histogram
        
        return result


class InterpretabilityEvaluator:
    """
    Comprehensive evaluator for transcoder interpretability.
    
    Implements trusted metrics from research:
    1. Absorption Score (SAE Bench) - measures feature redundancy
    2. Feature Utilization - tracks dead neurons
    3. Sparsity Analysis - L0 distribution
    4. Reconstruction Quality - MSE, MAE, FVU, cosine similarity
    """
    
    def __init__(
        self,
        absorption_threshold: float = 0.9,
        absorption_subset_size: int = 8192,
        dead_threshold: int = 20,
    ):
        """
        Initialize evaluator.
        
        Args:
            absorption_threshold: Cosine similarity threshold for absorption (0.9 is SAE Bench standard)
            absorption_subset_size: Max features to sample for absorption computation (memory efficiency)
            dead_threshold: Number of consecutive inactive batches to consider a feature dead
        """
        self.absorption_threshold = absorption_threshold
        self.absorption_subset_size = absorption_subset_size
        self.dead_threshold = dead_threshold
    
    @torch.no_grad()
    def compute_absorption_score(
        self, 
        feature_vectors: torch.Tensor,
        return_details: bool = False
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        """
        Compute absorption score following SAE Bench methodology.
        
        Absorption measures feature redundancy by computing the fraction of
        feature pairs with high cosine similarity (>0.9).
        
        Lower absorption = better (features are more diverse)
        Higher absorption = worse (features are redundant)
        
        Args:
            feature_vectors: Decoder weight matrix (n_features, d_model)
            return_details: If True, return additional statistics
            
        Returns:
            absorption_score: Fraction of pairs with similarity > threshold
            details: Optional dict with additional statistics
        """
        if feature_vectors is None or feature_vectors.numel() == 0:
            return float('nan'), None
        
        # Normalize feature vectors (L2 norm)
        V = F.normalize(feature_vectors, p=2, dim=-1).float()
        
        # Subsample if too large (memory efficiency)
        if V.shape[0] > self.absorption_subset_size:
            idx = torch.randperm(V.shape[0])[:self.absorption_subset_size]
            V = V[idx]
        
        n_features = V.shape[0]
        
        # Compute pairwise cosine similarity matrix
        # S[i,j] = cosine_similarity(V[i], V[j])
        S = torch.abs(V @ V.T)  # Take absolute value to catch anti-correlation too
        
        # Remove diagonal (self-similarity)
        mask = ~torch.eye(n_features, dtype=torch.bool, device=S.device)
        similarities = S[mask]
        
        # Count pairs above threshold
        above_threshold = (similarities > self.absorption_threshold).sum().item()
        total_pairs = similarities.numel()
        
        # Absorption score: fraction of pairs with high similarity
        absorption_score = above_threshold / max(1, total_pairs)
        
        details = None
        if return_details:
            details = {
                "n_features_evaluated": n_features,
                "total_pairs": total_pairs,
                "pairs_above_threshold": above_threshold,
                "mean_similarity": similarities.mean().item(),
                "max_similarity": similarities.max().item(),
                "median_similarity": similarities.median().item(),
                "similarity_std": similarities.std().item(),
            }
        
        return absorption_score, details
    
    @torch.no_grad()
    def compute_feature_utilization(
        self,
        num_batches_not_active: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Compute feature utilization statistics.
        
        Args:
            num_batches_not_active: Tensor tracking consecutive inactive batches per feature
            
        Returns:
            Dictionary with dead/alive feature counts and percentages
        """
        dead_mask = num_batches_not_active >= self.dead_threshold
        dead_count = dead_mask.sum().item()
        total_features = num_batches_not_active.numel()
        alive_count = total_features - dead_count
        
        return {
            "dead_count": dead_count,
            "dead_percentage": 100.0 * dead_count / max(1, total_features),
            "alive_count": alive_count,
            "alive_percentage": 100.0 * alive_count / max(1, total_features),
            "total_features": total_features,
        }
    
    @torch.no_grad()
    def compute_sparsity_metrics(
        self,
        activations: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute sparsity metrics from feature activations.
        
        Args:
            activations: Feature activation tensor (batch_size, n_features)
            
        Returns:
            Dictionary with L0 statistics
        """
        # L0 norm: count of non-zero activations per sample
        l0_per_sample = (activations > 0).sum(dim=-1).float()
        
        return {
            "mean_l0": l0_per_sample.mean().item(),
            "median_l0": l0_per_sample.median().item(),
            "std_l0": l0_per_sample.std().item(),
            "min_l0": l0_per_sample.min().item(),
            "max_l0": l0_per_sample.max().item(),
        }
    
    @torch.no_grad()
    def compute_reconstruction_metrics(
        self,
        original: torch.Tensor,
        reconstruction: torch.Tensor,
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
        source_activations: torch.Tensor,
        target_activations: torch.Tensor,
    ) -> InterpretabilityMetrics:
        """
        Comprehensive evaluation of a transcoder.
        
        Args:
            transcoder: Transcoder model to evaluate
            source_activations: Source layer activations (batch_size, d_model)
            target_activations: Target layer activations (batch_size, d_model)
            
        Returns:
            InterpretabilityMetrics object with all computed metrics
        """
        transcoder.eval()
        
        # Get feature vectors (decoder weights)
        if hasattr(transcoder, 'W_dec'):
            feature_vectors = transcoder.W_dec.detach()
        elif hasattr(transcoder, 'model') and hasattr(transcoder.model, 'W_dec'):
            feature_vectors = transcoder.model.W_dec.detach()
        else:
            feature_vectors = None
        
        # Compute absorption score
        if feature_vectors is not None:
            absorption_score, absorption_details = self.compute_absorption_score(
                feature_vectors, 
                return_details=True
            )
            absorption_pairs = absorption_details["pairs_above_threshold"]
        else:
            absorption_score = float('nan')
            absorption_pairs = 0
        
        # Get sparse features and reconstruction
        sparse_features = transcoder.encode(source_activations)
        reconstruction = transcoder.decode(sparse_features)
        
        # Feature utilization
        if hasattr(transcoder, 'num_batches_not_active'):
            util_stats = self.compute_feature_utilization(transcoder.num_batches_not_active)
        else:
            util_stats = {
                "dead_count": 0,
                "dead_percentage": 0.0,
                "alive_count": 0,
                "total_features": 0,
            }
        
        # Sparsity metrics
        sparsity_stats = self.compute_sparsity_metrics(sparse_features)
        
        # Reconstruction metrics
        recon_stats = self.compute_reconstruction_metrics(target_activations, reconstruction)
        
        # Activation statistics
        max_activation = sparse_features.max().item()
        mean_activation = sparse_features[sparse_features > 0].mean().item() if (sparse_features > 0).any() else 0.0
        
        return InterpretabilityMetrics(
            absorption_score=absorption_score,
            absorption_pairs_count=absorption_pairs,
            absorption_threshold=self.absorption_threshold,
            dead_features_count=util_stats["dead_count"],
            dead_features_percentage=util_stats["dead_percentage"],
            alive_features_count=util_stats["alive_count"],
            total_features=util_stats["total_features"],
            mean_l0=sparsity_stats["mean_l0"],
            median_l0=sparsity_stats["median_l0"],
            l0_std=sparsity_stats["std_l0"],
            mse=recon_stats["mse"],
            mae=recon_stats["mae"],
            fvu=recon_stats["fvu"],
            cosine_similarity=recon_stats["cosine_similarity"],
            max_feature_activation=max_activation,
            mean_feature_activation=mean_activation,
        )


def save_metrics(metrics: InterpretabilityMetrics, save_path: str):
    """Save metrics to JSON file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)
    
    print(f"Saved metrics to: {save_path}")


def print_metrics_summary(metrics: InterpretabilityMetrics, model_name: str = "Model"):
    """Print formatted summary of metrics."""
    print("\n" + "="*80)
    print(f"INTERPRETABILITY METRICS - {model_name}")
    print("="*80)
    
    print("\n📊 ABSORPTION SCORE (Lower is Better)")
    print(f"  Score: {metrics.absorption_score:.4f}")
    print(f"  Pairs above {metrics.absorption_threshold}: {metrics.absorption_pairs_count:,}")
    print(f"  Interpretation: {'✅ Good (diverse features)' if metrics.absorption_score < 0.01 else '⚠️  High redundancy' if metrics.absorption_score < 0.05 else '❌ Very high redundancy'}")
    
    print("\n🔧 FEATURE UTILIZATION")
    print(f"  Dead features: {metrics.dead_features_count} / {metrics.total_features} ({metrics.dead_features_percentage:.1f}%)")
    print(f"  Alive features: {metrics.alive_features_count} ({100 - metrics.dead_features_percentage:.1f}%)")
    print(f"  Interpretation: {'✅ Good utilization' if metrics.dead_features_percentage < 10 else '⚠️  Some waste' if metrics.dead_features_percentage < 30 else '❌ Poor utilization'}")
    
    print("\n✨ SPARSITY")
    print(f"  Mean L0: {metrics.mean_l0:.1f}")
    print(f"  Median L0: {metrics.median_l0:.1f}")
    print(f"  Std L0: {metrics.l0_std:.1f}")
    
    print("\n🎯 RECONSTRUCTION QUALITY")
    print(f"  FVU: {metrics.fvu:.4f} (lower is better)")
    print(f"  MSE: {metrics.mse:.6f}")
    print(f"  MAE: {metrics.mae:.6f}")
    print(f"  Cosine similarity: {metrics.cosine_similarity:.4f}")
    print(f"  Interpretation: {'✅ Excellent' if metrics.fvu < 0.1 else '⚠️  Moderate' if metrics.fvu < 0.3 else '❌ Poor reconstruction'}")
    
    print("\n🔥 ACTIVATION STATISTICS")
    print(f"  Max activation: {metrics.max_feature_activation:.4f}")
    print(f"  Mean activation (non-zero): {metrics.mean_feature_activation:.4f}")
    
    print("="*80 + "\n")

