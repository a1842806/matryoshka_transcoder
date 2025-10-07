"""
Comprehensive Feature Evaluation for Matryoshka Transcoders.

This script measures feature absorption, splitting, and quality metrics including:
1. Feature similarity (absorption detection)
2. Feature co-activation (splitting detection)
3. Hierarchical feature analysis (Matryoshka-specific)
4. Monosemanticity scores
5. Dead neuron analysis
6. Feature frequency distributions

Based on metrics from:
- SAE Bench
- Matryoshka SAE paper
- Anthropic's interpretability work
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
import json
import os


class FeatureEvaluator:
    """
    Evaluates feature quality, absorption, and splitting in Matryoshka Transcoders.
    
    Key Metrics:
    1. Feature Absorption: High cosine similarity between features (bad)
    2. Feature Splitting: High co-activation between features (bad)
    3. Monosemanticity: Low activation overlap across contexts (good)
    4. Hierarchical Separation: Low similarity within groups, higher across (good for Matryoshka)
    """
    
    def __init__(self, transcoder, activation_store, model, cfg):
        self.transcoder = transcoder
        self.activation_store = activation_store
        self.model = model
        self.cfg = cfg
        self.device = cfg["device"]
        
        # Storage for analysis
        self.feature_acts_history = []
        self.feature_weights = transcoder.W_dec.detach()  # (dict_size, act_size)
        self.dict_size = cfg["dict_size"]
        
        # Group indices for Matryoshka analysis
        if hasattr(transcoder, 'group_indices'):
            self.group_indices = transcoder.group_indices
            self.is_matryoshka = True
        else:
            self.is_matryoshka = False
    
    def collect_activations(self, num_batches=100):
        """Collect feature activations across multiple batches."""
        print(f"Collecting activations from {num_batches} batches...")
        
        activation_patterns = []
        self.transcoder.eval()
        
        with torch.no_grad():
            for i in tqdm(range(num_batches)):
                try:
                    source_batch, target_batch = self.activation_store.next_batch()
                    
                    # Get feature activations
                    sparse_features = self.transcoder.encode(source_batch)
                    activation_patterns.append(sparse_features.cpu())
                    
                except StopIteration:
                    break
        
        # Concatenate all activations
        self.feature_acts_history = torch.cat(activation_patterns, dim=0)
        print(f"Collected {self.feature_acts_history.shape[0]} activation samples")
        
        return self.feature_acts_history
    
    def compute_feature_absorption(self, threshold=0.9):
        """
        Measure feature absorption: High cosine similarity between feature vectors.
        
        Absorption occurs when multiple features learn similar representations,
        wasting dictionary capacity.
        
        Returns:
            absorption_score: Fraction of feature pairs with similarity > threshold
            similar_pairs: List of (feature_i, feature_j, similarity) tuples
        """
        print("\n" + "=" * 70)
        print("Computing Feature Absorption (Cosine Similarity)")
        print("=" * 70)
        
        # Normalize feature vectors
        feature_vecs_norm = F.normalize(self.feature_weights, p=2, dim=1)
        
        # Compute pairwise cosine similarities
        similarities = feature_vecs_norm @ feature_vecs_norm.T
        
        # Remove diagonal (self-similarity)
        similarities = similarities - torch.eye(self.dict_size, device=self.device)
        
        # Count pairs above threshold
        absorbed_pairs = (similarities > threshold).sum().item() / 2  # Divide by 2 for symmetry
        total_pairs = (self.dict_size * (self.dict_size - 1)) / 2
        absorption_score = absorbed_pairs / total_pairs
        
        # Find most similar pairs
        triu_indices = torch.triu_indices(self.dict_size, self.dict_size, offset=1)
        upper_tri_sims = similarities[triu_indices[0], triu_indices[1]]
        
        # Get top 10 most similar pairs
        top_k = min(10, len(upper_tri_sims))
        top_similarities, top_indices = torch.topk(upper_tri_sims, top_k)
        
        similar_pairs = []
        for sim, idx in zip(top_similarities, top_indices):
            i = triu_indices[0][idx].item()
            j = triu_indices[1][idx].item()
            similar_pairs.append((i, j, sim.item()))
        
        print(f"Absorption Score: {absorption_score:.4f}")
        print(f"  ({absorbed_pairs:.0f}/{total_pairs:.0f} pairs with similarity > {threshold})")
        print(f"\nTop 10 Most Similar Feature Pairs:")
        for i, j, sim in similar_pairs[:10]:
            group_i = self._get_group_index(i)
            group_j = self._get_group_index(j)
            print(f"  Feature {i:4d} (Group {group_i}) ↔ Feature {j:4d} (Group {group_j}): {sim:.4f}")
        
        return {
            "absorption_score": absorption_score,
            "absorbed_pairs": absorbed_pairs,
            "similar_pairs": similar_pairs,
            "similarity_matrix": similarities.cpu()
        }
    
    def compute_feature_splitting(self, threshold=0.5):
        """
        Measure feature splitting: High co-activation between features.
        
        Splitting occurs when multiple features activate together on the same inputs,
        indicating they could be merged into a single feature.
        
        Returns:
            splitting_score: Fraction of feature pairs that co-activate frequently
            split_pairs: List of (feature_i, feature_j, co-activation_rate) tuples
        """
        print("\n" + "=" * 70)
        print("Computing Feature Splitting (Co-activation)")
        print("=" * 70)
        
        if self.feature_acts_history is None or len(self.feature_acts_history) == 0:
            print("No activations collected. Run collect_activations() first.")
            return None
        
        # Binarize activations
        active_features = (self.feature_acts_history > 0).float()  # (n_samples, dict_size)
        
        # Compute co-activation rates
        # co_activation[i,j] = fraction of samples where both i and j are active
        co_activation = (active_features.T @ active_features) / active_features.shape[0]
        
        # Remove diagonal
        co_activation = co_activation - torch.eye(self.dict_size)
        
        # Count pairs above threshold
        split_pairs_count = (co_activation > threshold).sum().item() / 2
        total_pairs = (self.dict_size * (self.dict_size - 1)) / 2
        splitting_score = split_pairs_count / total_pairs
        
        # Find most co-activating pairs
        triu_indices = torch.triu_indices(self.dict_size, self.dict_size, offset=1)
        upper_tri_coact = co_activation[triu_indices[0], triu_indices[1]]
        
        top_k = min(10, len(upper_tri_coact))
        top_coactivations, top_indices = torch.topk(upper_tri_coact, top_k)
        
        split_pairs = []
        for coact, idx in zip(top_coactivations, top_indices):
            i = triu_indices[0][idx].item()
            j = triu_indices[1][idx].item()
            split_pairs.append((i, j, coact.item()))
        
        print(f"Splitting Score: {splitting_score:.4f}")
        print(f"  ({split_pairs_count:.0f}/{total_pairs:.0f} pairs co-activate > {threshold} of the time)")
        print(f"\nTop 10 Most Co-activating Feature Pairs:")
        for i, j, coact in split_pairs[:10]:
            group_i = self._get_group_index(i)
            group_j = self._get_group_index(j)
            print(f"  Feature {i:4d} (Group {group_i}) ↔ Feature {j:4d} (Group {group_j}): {coact:.4f}")
        
        return {
            "splitting_score": splitting_score,
            "split_pairs_count": split_pairs_count,
            "split_pairs": split_pairs,
            "co_activation_matrix": co_activation.cpu()
        }
    
    def compute_monosemanticity(self):
        """
        Measure monosemanticity: How specific/interpretable each feature is.
        
        Monosemantic features activate sparsely and consistently for specific concepts.
        Polysemantic features activate for many different things.
        
        Returns:
            monosemanticity_scores: Score per feature (higher = more monosemantic)
        """
        print("\n" + "=" * 70)
        print("Computing Monosemanticity Scores")
        print("=" * 70)
        
        if self.feature_acts_history is None or len(self.feature_acts_history) == 0:
            print("No activations collected. Run collect_activations() first.")
            return None
        
        # For each feature, compute:
        # 1. Sparsity (how often it activates)
        # 2. Activation entropy (how consistent its activation magnitude is)
        
        activation_freq = (self.feature_acts_history > 0).float().mean(dim=0)  # (dict_size,)
        
        # Compute activation entropy for each feature
        # Lower entropy = more consistent = more monosemantic
        entropies = []
        for i in range(self.dict_size):
            acts = self.feature_acts_history[:, i]
            active_acts = acts[acts > 0]
            
            if len(active_acts) > 10:
                # Discretize activations into bins
                hist, _ = np.histogram(active_acts.numpy(), bins=50, density=True)
                hist = hist + 1e-10  # Avoid log(0)
                ent = entropy(hist)
                entropies.append(ent)
            else:
                entropies.append(0.0)  # Too few activations
        
        entropies = torch.tensor(entropies)
        
        # Monosemanticity score: balance between sparsity and consistency
        # High sparsity + low entropy = high monosemanticity
        sparsity_score = 1 - activation_freq  # Reward sparsity
        entropy_score = 1 / (1 + entropies)    # Reward low entropy
        
        monosemanticity = (sparsity_score + entropy_score) / 2
        
        # Statistics
        mean_mono = monosemanticity.mean().item()
        median_mono = monosemanticity.median().item()
        
        print(f"Mean Monosemanticity: {mean_mono:.4f}")
        print(f"Median Monosemanticity: {median_mono:.4f}")
        
        # Most and least monosemantic features
        top_k = min(5, self.dict_size)
        most_mono_values, most_mono_idx = torch.topk(monosemanticity, top_k)
        least_mono_values, least_mono_idx = torch.topk(monosemanticity, top_k, largest=False)
        
        print(f"\nMost Monosemantic Features:")
        for val, idx in zip(most_mono_values, most_mono_idx):
            freq = activation_freq[idx].item()
            print(f"  Feature {idx:4d}: {val:.4f} (activates {freq:.2%} of the time)")
        
        print(f"\nLeast Monosemantic Features (potentially polysemantic):")
        for val, idx in zip(least_mono_values, least_mono_idx):
            freq = activation_freq[idx].item()
            print(f"  Feature {idx:4d}: {val:.4f} (activates {freq:.2%} of the time)")
        
        return {
            "monosemanticity_scores": monosemanticity.cpu(),
            "mean": mean_mono,
            "median": median_mono,
            "activation_freq": activation_freq.cpu(),
            "entropies": entropies.cpu()
        }
    
    def analyze_hierarchical_structure(self):
        """
        Analyze hierarchical structure in Matryoshka Transcoders.
        
        Good Matryoshka models should have:
        - Low similarity within groups (features are diverse)
        - Clear separation between groups (early = coarse, late = fine)
        """
        if not self.is_matryoshka:
            print("Not a Matryoshka model. Skipping hierarchical analysis.")
            return None
        
        print("\n" + "=" * 70)
        print("Analyzing Hierarchical Structure (Matryoshka-specific)")
        print("=" * 70)
        
        feature_vecs_norm = F.normalize(self.feature_weights, p=2, dim=1)
        
        # Compute within-group and between-group similarities
        group_stats = []
        
        for g in range(len(self.group_indices) - 1):
            start_idx = self.group_indices[g]
            end_idx = self.group_indices[g + 1]
            
            # Within-group similarity
            group_features = feature_vecs_norm[start_idx:end_idx]
            within_sim = (group_features @ group_features.T)
            within_sim = within_sim - torch.eye(end_idx - start_idx, device=self.device)
            within_sim_mean = within_sim.abs().mean().item()
            
            # Between-group similarity (with next group if exists)
            if g < len(self.group_indices) - 2:
                next_start = self.group_indices[g + 1]
                next_end = self.group_indices[g + 2]
                next_group_features = feature_vecs_norm[next_start:next_end]
                between_sim = (group_features @ next_group_features.T)
                between_sim_mean = between_sim.abs().mean().item()
            else:
                between_sim_mean = None
            
            group_stats.append({
                "group": g,
                "size": end_idx - start_idx,
                "within_similarity": within_sim_mean,
                "between_similarity": between_sim_mean
            })
            
            print(f"\nGroup {g} (features {start_idx}-{end_idx-1}, size={end_idx - start_idx}):")
            print(f"  Within-group similarity: {within_sim_mean:.4f}")
            if between_sim_mean is not None:
                print(f"  Between-group similarity: {between_sim_mean:.4f}")
                print(f"  Separation ratio: {between_sim_mean / (within_sim_mean + 1e-8):.4f}")
        
        return {"group_stats": group_stats}
    
    def analyze_dead_features(self):
        """Analyze dead/inactive features."""
        print("\n" + "=" * 70)
        print("Analyzing Dead Features")
        print("=" * 70)
        
        if self.feature_acts_history is None or len(self.feature_acts_history) == 0:
            print("No activations collected. Run collect_activations() first.")
            return None
        
        # Count features that never activate
        activation_counts = (self.feature_acts_history > 0).sum(dim=0)
        dead_features = (activation_counts == 0).sum().item()
        dead_fraction = dead_features / self.dict_size
        
        print(f"Dead features: {dead_features} / {self.dict_size} ({dead_fraction:.2%})")
        
        # Per-group analysis for Matryoshka
        if self.is_matryoshka:
            print(f"\nDead features per group:")
            for g in range(len(self.group_indices) - 1):
                start_idx = self.group_indices[g]
                end_idx = self.group_indices[g + 1]
                group_dead = (activation_counts[start_idx:end_idx] == 0).sum().item()
                group_size = end_idx - start_idx
                print(f"  Group {g}: {group_dead} / {group_size} ({group_dead/group_size:.2%})")
        
        return {
            "dead_features": dead_features,
            "dead_fraction": dead_fraction,
            "activation_counts": activation_counts.cpu()
        }
    
    def _get_group_index(self, feature_idx):
        """Get which group a feature belongs to."""
        if not self.is_matryoshka:
            return 0
        
        for g in range(len(self.group_indices) - 1):
            if self.group_indices[g] <= feature_idx < self.group_indices[g + 1]:
                return g
        return len(self.group_indices) - 2
    
    def save_results(self, results, save_dir="evaluation_results"):
        """Save evaluation results."""
        os.makedirs(save_dir, exist_ok=True)
        
        def convert_to_serializable(obj):
            """Recursively convert tensors and numpy arrays to serializable types."""
            if isinstance(obj, torch.Tensor):
                # Convert tensors to lists if small, otherwise skip
                if obj.numel() < 10000:
                    return obj.cpu().tolist() if obj.numel() > 1 else obj.item()
                else:
                    return None  # Skip large tensors
            elif isinstance(obj, np.ndarray):
                if obj.size < 10000:
                    return obj.tolist()
                else:
                    return None
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items() if convert_to_serializable(v) is not None}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj if convert_to_serializable(item) is not None]
            elif isinstance(obj, tuple):
                converted = [convert_to_serializable(item) for item in obj]
                return tuple(item for item in converted if item is not None)
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return None
        
        # Convert results to serializable format
        save_dict = convert_to_serializable(results)
        
        with open(f"{save_dir}/results.json", "w") as f:
            json.dump(save_dict, f, indent=2)
        
        print(f"\nResults saved to {save_dir}/results.json")
    
    def run_full_evaluation(self, num_batches=100, save_dir="evaluation_results"):
        """Run complete feature evaluation."""
        print("\n" + "=" * 70)
        print("FULL FEATURE EVALUATION")
        print("=" * 70)
        
        # Collect activations
        self.collect_activations(num_batches)
        
        # Run all evaluations
        results = {}
        results["absorption"] = self.compute_feature_absorption()
        results["splitting"] = self.compute_feature_splitting()
        results["monosemanticity"] = self.compute_monosemanticity()
        results["dead_features"] = self.analyze_dead_features()
        
        if self.is_matryoshka:
            results["hierarchical"] = self.analyze_hierarchical_structure()
        
        # Summary
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print(f"Absorption Score: {results['absorption']['absorption_score']:.4f} (lower is better)")
        print(f"Splitting Score: {results['splitting']['splitting_score']:.4f} (lower is better)")
        print(f"Mean Monosemanticity: {results['monosemanticity']['mean']:.4f} (higher is better)")
        print(f"Dead Features: {results['dead_features']['dead_fraction']:.2%} (lower is better)")
        print("=" * 70)
        
        # Save results
        self.save_results(results, save_dir)
        
        return results


def evaluate_trained_transcoder(checkpoint_path, activation_store, model, cfg, num_batches=100):
    """
    Convenience function to evaluate a trained transcoder.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        activation_store: TranscoderActivationsStore instance
        model: HookedTransformer model
        cfg: Configuration dict
        num_batches: Number of batches to collect for evaluation
    """
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
    from models.sae import MatryoshkaTranscoder
    
    # Load transcoder
    print(f"Loading transcoder from {checkpoint_path}...")
    transcoder = MatryoshkaTranscoder(cfg)
    transcoder.load_state_dict(torch.load(f"{checkpoint_path}/sae.pt", map_location=cfg["device"]))
    transcoder.eval()
    
    # Run evaluation
    evaluator = FeatureEvaluator(transcoder, activation_store, model, cfg)
    results = evaluator.run_full_evaluation(num_batches, save_dir=f"{checkpoint_path}/evaluation")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Feature Evaluation Script")
    print("This script should be imported and used with a trained transcoder.")
    print("\nExample usage:")
    print("""
from evaluate_features import evaluate_trained_transcoder

# Load your model and stores
model = HookedTransformer.from_pretrained("gpt2-small")
activation_store = TranscoderActivationsStore(model, cfg)

# Evaluate
results = evaluate_trained_transcoder(
    checkpoint_path="checkpoints/your_checkpoint",
    activation_store=activation_store,
    model=model,
    cfg=cfg,
    num_batches=100
)
    """)

