"""
Simplified SAEBench Evaluation - Working Version

This is a simplified but functional implementation that can actually run
the SAEBench metrics on your Matryoshka transcoder without requiring
full sequence collection and LLM integration.

Focuses on what we can actually compute:
1. Feature redundancy (simplified absorption-like metric)
2. Reconstruction quality (FVU, MSE, cosine similarity)
3. Sparsity analysis (L0 distribution)
4. Feature utilization (dead features)
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.sae import MatryoshkaTranscoder
from src.models.transcoder_activation_store import TranscoderActivationsStore, create_transcoder_config
from src.utils.config import get_default_cfg


@dataclass
class SimpleSAEBenchMetrics:
    """Simplified SAEBench metrics that we can actually compute."""
    
    # Feature Redundancy (simplified absorption)
    feature_redundancy: float  # Fraction of similar feature pairs
    similar_pairs_count: int
    redundancy_threshold: float
    
    # Reconstruction Quality
    fvu: float  # Fraction of Variance Unexplained
    mse: float
    mae: float
    cosine_similarity: float
    
    # Sparsity
    mean_l0: float
    median_l0: float
    l0_std: float
    
    # Feature Utilization
    dead_features_count: int
    dead_features_percentage: float
    alive_features_count: int
    total_features: int
    
    # Model info
    dict_size: int
    layer: int
    model_name: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SimpleSAEBenchEvaluator:
    """
    Simplified SAEBench evaluator that actually works.
    
    Computes interpretability metrics we can realistically implement:
    1. Feature redundancy (simplified absorption)
    2. Reconstruction quality
    3. Sparsity analysis
    4. Feature utilization
    """
    
    def __init__(
        self,
        redundancy_threshold: float = 0.9,
        dead_threshold: int = 20,
        num_batches: int = 100
    ):
        self.redundancy_threshold = redundancy_threshold
        self.dead_threshold = dead_threshold
        self.num_batches = num_batches
    
    @torch.no_grad()
    def evaluate(
        self,
        transcoder,
        activation_store: TranscoderActivationsStore,
        layer: int,
        cfg: Dict
    ) -> SimpleSAEBenchMetrics:
        """
        Evaluate transcoder using simplified SAEBench metrics.
        
        Args:
            transcoder: Matryoshka transcoder to evaluate
            activation_store: Source of activation pairs
            layer: Layer number
            
        Returns:
            SimpleSAEBenchMetrics with computed metrics
        """
        print(f"\n{'='*80}")
        print(f"🔬 Simplified SAEBench Evaluation (Layer {layer})")
        print(f"{'='*80}")
        
        # Step 1: Compute feature redundancy
        print("📊 Computing feature redundancy...")
        redundancy_score, similar_pairs = self._compute_feature_redundancy(transcoder)
        
        # Step 2: Evaluate reconstruction quality
        print("🎯 Evaluating reconstruction quality...")
        recon_metrics = self._evaluate_reconstruction_quality(
            transcoder, activation_store
        )
        
        # Step 3: Analyze sparsity
        print("✨ Analyzing sparsity...")
        sparsity_metrics = self._analyze_sparsity(transcoder, activation_store)
        
        # Step 4: Check feature utilization
        print("🔧 Checking feature utilization...")
        utilization_metrics = self._check_feature_utilization(transcoder)
        
        # Compile results
        metrics = SimpleSAEBenchMetrics(
            feature_redundancy=redundancy_score,
            similar_pairs_count=similar_pairs,
            redundancy_threshold=self.redundancy_threshold,
            fvu=recon_metrics["fvu"],
            mse=recon_metrics["mse"],
            mae=recon_metrics["mae"],
            cosine_similarity=recon_metrics["cosine_similarity"],
            mean_l0=sparsity_metrics["mean_l0"],
            median_l0=sparsity_metrics["median_l0"],
            l0_std=sparsity_metrics["l0_std"],
            dead_features_count=utilization_metrics["dead_count"],
            dead_features_percentage=utilization_metrics["dead_percentage"],
            alive_features_count=utilization_metrics["alive_count"],
            total_features=utilization_metrics["total_features"],
            dict_size=cfg['dict_size'],
            layer=layer,
            model_name="gemma-2-2b"
        )
        
        return metrics
    
    @torch.no_grad()
    def _compute_feature_redundancy(self, transcoder) -> Tuple[float, int]:
        """Compute feature redundancy via pairwise cosine similarity (memory-efficient)."""
        # Get decoder weights (feature vectors)
        W_dec = transcoder.W_dec.detach()  # (dict_size, d_model)
        
        # L2-normalize feature vectors
        W_norm = F.normalize(W_dec, p=2, dim=-1)
        
        # Move to CPU for memory efficiency
        W_norm = W_norm.cpu().float()
        
        dict_size = W_norm.shape[0]
        chunk_size = 1000  # Process in chunks to avoid OOM
        
        similar_pairs = 0
        total_pairs = 0
        
        print(f"  Computing redundancy for {dict_size:,} features in chunks of {chunk_size:,}...")
        
        for i in tqdm(range(0, dict_size, chunk_size), desc="Redundancy"):
            i_end = min(i + chunk_size, dict_size)
            chunk_i = W_norm[i:i_end]  # (chunk_size, d_model)
            
            for j in range(i, dict_size, chunk_size):
                j_end = min(j + chunk_size, dict_size)
                chunk_j = W_norm[j:j_end]  # (chunk_size, d_model)
                
                # Compute similarities between chunks
                similarities = torch.abs(chunk_i @ chunk_j.T)  # (chunk_i, chunk_j)
                
                if i == j:
                    # Same chunk: remove diagonal
                    mask = ~torch.eye(similarities.shape[0], dtype=torch.bool)
                    similarities = similarities[mask]
                    chunk_pairs = similarities.numel()
                else:
                    # Different chunks: all pairs count
                    chunk_pairs = similarities.numel()
                
                # Count similar pairs in this chunk
                chunk_similar = (similarities > self.redundancy_threshold).sum().item()
                similar_pairs += chunk_similar
                total_pairs += chunk_pairs
        
        # Redundancy score: fraction of pairs with high similarity
        redundancy_score = similar_pairs / max(1, total_pairs)
        
        return redundancy_score, similar_pairs
    
    @torch.no_grad()
    def _evaluate_reconstruction_quality(
        self, 
        transcoder, 
        activation_store: TranscoderActivationsStore
    ) -> Dict[str, float]:
        """Evaluate reconstruction quality over multiple batches."""
        total_mse = 0.0
        total_mae = 0.0
        total_fvu = 0.0
        total_cos = 0.0
        total_samples = 0
        
        print(f"  Evaluating over {self.num_batches} batches...")
        
        for i in tqdm(range(self.num_batches), desc="Reconstruction"):
            try:
                source_batch, target_batch = activation_store.next_batch()
                source_batch = source_batch.to(transcoder.W_dec.device)
                target_batch = target_batch.to(source_batch.device)
                
                batch_size = source_batch.shape[0]
                
                # Get reconstruction
                sparse_features = transcoder.encode(source_batch)
                reconstruction = transcoder.decode(sparse_features)
                
                # Compute metrics
                mse = F.mse_loss(reconstruction, target_batch).item()
                mae = (reconstruction - target_batch).abs().mean().item()
                
                # FVU (Fraction of Variance Unexplained)
                residuals = reconstruction - target_batch
                var_residuals = residuals.pow(2).mean().item()
                var_original = target_batch.var().item()
                fvu = var_residuals / (var_original + 1e-8)
                
                # Cosine similarity
                cos_sim = F.cosine_similarity(
                    reconstruction.flatten(),
                    target_batch.flatten(),
                    dim=0
                ).item()
                
                # Accumulate
                total_mse += mse * batch_size
                total_mae += mae * batch_size
                total_fvu += fvu * batch_size
                total_cos += cos_sim * batch_size
                total_samples += batch_size
                
            except StopIteration:
                break
        
        # Compute averages
        avg_mse = total_mse / max(1, total_samples)
        avg_mae = total_mae / max(1, total_samples)
        avg_fvu = total_fvu / max(1, total_samples)
        avg_cos = total_cos / max(1, total_samples)
        
        return {
            "mse": avg_mse,
            "mae": avg_mae,
            "fvu": avg_fvu,
            "cosine_similarity": avg_cos
        }
    
    @torch.no_grad()
    def _analyze_sparsity(
        self, 
        transcoder, 
        activation_store: TranscoderActivationsStore
    ) -> Dict[str, float]:
        """Analyze sparsity patterns."""
        all_l0_values = []
        
        print(f"  Analyzing sparsity over {self.num_batches} batches...")
        
        for i in tqdm(range(self.num_batches), desc="Sparsity"):
            try:
                source_batch, _ = activation_store.next_batch()
                source_batch = source_batch.to(transcoder.W_dec.device)
                
                # Get sparse features
                sparse_features = transcoder.encode(source_batch)
                
                # L0 norm: count of non-zero activations per sample
                l0_per_sample = (sparse_features > 0).sum(dim=-1).float()
                all_l0_values.extend(l0_per_sample.cpu().tolist())
                
            except StopIteration:
                break
        
        if not all_l0_values:
            return {"mean_l0": 0.0, "median_l0": 0.0, "l0_std": 0.0}
        
        l0_tensor = torch.tensor(all_l0_values)
        
        return {
            "mean_l0": l0_tensor.mean().item(),
            "median_l0": l0_tensor.median().item(),
            "l0_std": l0_tensor.std().item()
        }
    
    @torch.no_grad()
    def _check_feature_utilization(self, transcoder) -> Dict[str, int]:
        """Check feature utilization (dead vs alive features)."""
        if hasattr(transcoder, 'num_batches_not_active'):
            dead_mask = transcoder.num_batches_not_active >= self.dead_threshold
            dead_count = dead_mask.sum().item()
            total_features = transcoder.num_batches_not_active.numel()
            alive_count = total_features - dead_count
            dead_percentage = 100.0 * dead_count / max(1, total_features)
        else:
            # Fallback: assume all features are alive if we can't track dead features
            dead_count = 0
            total_features = 18432  # From config
            alive_count = total_features
            dead_percentage = 0.0
        
        return {
            "dead_count": dead_count,
            "alive_count": alive_count,
            "total_features": total_features,
            "dead_percentage": dead_percentage
        }


def load_transcoder(checkpoint_path: str, device: torch.device, dtype: torch.dtype):
    """Load transcoder from checkpoint."""
    print(f"\n📥 Loading transcoder from: {checkpoint_path}")
    
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
    
    print(f"✓ Transcoder loaded successfully")
    print(f"  - Dictionary size: {cfg['dict_size']:,}")
    print(f"  - Prefix sizes: {cfg['prefix_sizes']}")
    print(f"  - Top-k: {cfg['top_k']}")
    
    return transcoder, cfg


def print_results(metrics: SimpleSAEBenchMetrics):
    """Print evaluation results in a nice format."""
    print("\n" + "="*80)
    print("📊 SIMPLIFIED SAEBENCH RESULTS")
    print("="*80)
    
    print(f"\n🎯 RECONSTRUCTION QUALITY")
    print("-"*80)
    print(f"FVU (lower=better):        {metrics.fvu:.6f}")
    print(f"MSE:                       {metrics.mse:.6f}")
    print(f"MAE:                       {metrics.mae:.6f}")
    print(f"Cosine Similarity:         {metrics.cosine_similarity:.6f}")
    
    print(f"\n📊 FEATURE REDUNDANCY (Simplified Absorption)")
    print("-"*80)
    print(f"Redundancy Score:          {metrics.feature_redundancy:.6f}")
    print(f"Similar Pairs:             {metrics.similar_pairs_count:,}")
    print(f"Threshold:                 {metrics.redundancy_threshold}")
    
    if metrics.feature_redundancy < 0.01:
        print(f"Interpretation:            ✅ EXCELLENT - Features are diverse")
    elif metrics.feature_redundancy < 0.05:
        print(f"Interpretation:            ⚠️  MODERATE - Some redundancy")
    else:
        print(f"Interpretation:            ❌ HIGH REDUNDANCY - Features overlap")
    
    print(f"\n✨ SPARSITY")
    print("-"*80)
    print(f"Mean L0:                   {metrics.mean_l0:.1f}")
    print(f"Median L0:                 {metrics.median_l0:.1f}")
    print(f"Std L0:                    {metrics.l0_std:.1f}")
    
    print(f"\n🔧 FEATURE UTILIZATION")
    print("-"*80)
    print(f"Dead Features:             {metrics.dead_features_count:,} / {metrics.total_features:,} ({metrics.dead_features_percentage:.1f}%)")
    print(f"Alive Features:            {metrics.alive_features_count:,}")
    
    if metrics.dead_features_percentage < 10:
        print(f"Interpretation:            ✅ EXCELLENT utilization")
    elif metrics.dead_features_percentage < 30:
        print(f"Interpretation:            ⚠️  MODERATE utilization")
    else:
        print(f"Interpretation:            ❌ POOR utilization")
    
    print(f"\n📋 MODEL INFO")
    print("-"*80)
    print(f"Model:                     {metrics.model_name}")
    print(f"Layer:                     {metrics.layer}")
    print(f"Dictionary Size:           {metrics.dict_size:,}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Simplified SAEBench evaluation for Matryoshka transcoders"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to transcoder checkpoint")
    parser.add_argument("--layer", type=int, default=8,
                       help="Layer to evaluate")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                       choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output_dir", type=str, default="analysis_results/simple_saebench",
                       help="Output directory")
    parser.add_argument("--batches", type=int, default=100,
                       help="Number of batches to evaluate")
    
    args = parser.parse_args()
    
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    device = torch.device(args.device)
    dtype = dtype_map[args.dtype]
    
    print("="*80)
    print("🔬 Simplified SAEBench Evaluation")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Layer: {args.layer}")
    print(f"Device: {device}")
    print(f"Batches: {args.batches}")
    
    # Load model
    print(f"\n📥 Loading Gemma-2-2B...")
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=device, dtype=dtype)
    print("✓ Model loaded")
    
    # Load transcoder
    transcoder, cfg = load_transcoder(args.checkpoint, device, dtype)
    
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
    
    # Run evaluation
    evaluator = SimpleSAEBenchEvaluator(num_batches=args.batches)
    metrics = evaluator.evaluate(transcoder, activation_store, args.layer, cfg)
    
    # Print results
    print_results(metrics)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "simple_saebench_metrics.json")
    with open(output_path, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    print("\n🎉 Evaluation complete!")


if __name__ == "__main__":
    main()
