"""
Analysis and visualization utilities for activation samples.

Provides tools to analyze and visualize collected activation samples
following Anthropic's approach for feature interpretability.
"""

import json
import os
from typing import List, Dict, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class ActivationSampleAnalyzer:
    """
    Analyzer for collected activation samples.
    
    Provides methods to:
    - Load and analyze saved samples
    - Generate interpretability reports
    - Visualize activation patterns
    - Compare features
    """
    
    def __init__(self, samples_dir: str):
        """
        Initialize analyzer with saved samples directory.
        
        Args:
            samples_dir: Directory containing saved activation samples
        """
        self.samples_dir = samples_dir
        self.features = {}
        self.summary = None
        self.load_samples()
    
    def load_samples(self):
        """Load all samples from directory."""
        if not os.path.exists(self.samples_dir):
            raise FileNotFoundError(f"Samples directory not found: {self.samples_dir}")
        
        # Load summary
        summary_path = os.path.join(self.samples_dir, "collection_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                self.summary = json.load(f)
        
        # Load feature samples
        for filename in os.listdir(self.samples_dir):
            if filename.startswith("feature_") and filename.endswith("_samples.json"):
                filepath = os.path.join(self.samples_dir, filename)
                
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                feature_idx = data['statistics']['feature_idx']
                self.features[feature_idx] = data
        
        print(f"Loaded {len(self.features)} features from {self.samples_dir}")
    
    def print_feature_report(self, feature_idx: int, num_examples: int = 5):
        """
        Print interpretability report for a single feature.
        
        Args:
            feature_idx: Feature index to analyze
            num_examples: Number of top examples to show
        """
        if feature_idx not in self.features:
            print(f"Feature {feature_idx} not found in loaded samples.")
            return
        
        feature_data = self.features[feature_idx]
        stats = feature_data['statistics']
        samples = feature_data['top_samples'][:num_examples]
        
        print("\n" + "="*80)
        print(f"FEATURE {feature_idx} INTERPRETABILITY REPORT")
        print("="*80)
        
        # Statistics
        print("\nStatistics:")
        print(f"  Max activation: {stats['max_activation']:.4f}")
        print(f"  Mean activation: {stats['mean_activation']:.4f}")
        print(f"  Std activation: {stats.get('std_activation', 0):.4f}")
        print(f"  Total activations seen: {stats['activation_count']}")
        print(f"  Samples stored: {stats['num_samples']}")
        
        # Top activating examples
        print(f"\nTop {num_examples} Activating Examples:")
        print("-"*80)
        
        for i, sample in enumerate(samples, 1):
            print(f"\n[Example {i}] Activation: {sample['activation_value']:.4f}")
            print(f"  Activating text: '{sample['text']}'")
            print(f"  Context: ...{sample['context_text']}...")
            print(f"  Position in sequence: {sample['position']}")
        
        print("\n" + "="*80)
    
    def analyze_feature_clustering(self):
        """Analyze which features have similar activation patterns."""
        if len(self.features) < 2:
            print("Not enough features to analyze clustering.")
            return
        
        print("\n" + "="*80)
        print("FEATURE ACTIVATION ANALYSIS")
        print("="*80)
        
        # Group features by activation statistics
        activation_stats = []
        for feature_idx, data in self.features.items():
            stats = data['statistics']
            activation_stats.append({
                'feature_idx': feature_idx,
                'max_activation': stats['max_activation'],
                'mean_activation': stats['mean_activation'],
                'activation_count': stats['activation_count']
            })
        
        # Sort by max activation
        activation_stats.sort(key=lambda x: x['max_activation'], reverse=True)
        
        print("\nTop 20 Most Active Features:")
        print(f"{'Feature':<10} {'Max Act':<12} {'Mean Act':<12} {'Count':<10}")
        print("-"*50)
        
        for stat in activation_stats[:20]:
            print(f"{stat['feature_idx']:<10} "
                  f"{stat['max_activation']:<12.4f} "
                  f"{stat['mean_activation']:<12.4f} "
                  f"{stat['activation_count']:<10}")
    
    def compare_features(self, feature_idx1: int, feature_idx2: int):
        """Compare two features side by side."""
        if feature_idx1 not in self.features or feature_idx2 not in self.features:
            print("One or both features not found.")
            return
        
        print("\n" + "="*80)
        print(f"COMPARING FEATURES {feature_idx1} AND {feature_idx2}")
        print("="*80)
        
        for idx in [feature_idx1, feature_idx2]:
            data = self.features[idx]
            stats = data['statistics']
            samples = data['top_samples'][:3]
            
            print(f"\n--- Feature {idx} ---")
            print(f"Max activation: {stats['max_activation']:.4f}")
            print(f"Mean activation: {stats['mean_activation']:.4f}")
            print(f"\nTop 3 Examples:")
            for i, sample in enumerate(samples, 1):
                print(f"  {i}. [{sample['activation_value']:.3f}] {sample['text']}")
    
    def plot_activation_distribution(
        self, 
        feature_idx: int, 
        save_path: Optional[str] = None
    ):
        """
        Plot activation value distribution for a feature.
        
        Args:
            feature_idx: Feature to plot
            save_path: Optional path to save plot
        """
        if feature_idx not in self.features:
            print(f"Feature {feature_idx} not found.")
            return
        
        data = self.features[feature_idx]
        samples = data['top_samples']
        activations = [s['activation_value'] for s in samples]
        
        plt.figure(figsize=(10, 6))
        plt.hist(activations, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        plt.xlabel('Activation Value', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Feature {feature_idx} - Activation Distribution', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        # Add statistics
        mean_act = np.mean(activations)
        max_act = np.max(activations)
        plt.axvline(mean_act, color='red', linestyle='--', label=f'Mean: {mean_act:.3f}')
        plt.axvline(max_act, color='green', linestyle='--', label=f'Max: {max_act:.3f}')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_feature_comparison(
        self, 
        feature_indices: List[int], 
        save_path: Optional[str] = None
    ):
        """
        Plot comparison of multiple features' activation statistics.
        
        Args:
            feature_indices: List of features to compare
            save_path: Optional path to save plot
        """
        available_features = [f for f in feature_indices if f in self.features]
        
        if not available_features:
            print("None of the specified features are available.")
            return
        
        # Gather statistics
        max_acts = []
        mean_acts = []
        labels = []
        
        for idx in available_features:
            stats = self.features[idx]['statistics']
            max_acts.append(stats['max_activation'])
            mean_acts.append(stats['mean_activation'])
            labels.append(f"F{idx}")
        
        # Create plot
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, max_acts, width, label='Max Activation', color='steelblue', alpha=0.7)
        ax.bar(x + width/2, mean_acts, width, label='Mean Activation', color='coral', alpha=0.7)
        
        ax.set_xlabel('Feature', fontsize=12)
        ax.set_ylabel('Activation Value', fontsize=12)
        ax.set_title('Feature Activation Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_interpretability_report(
        self, 
        output_file: str,
        top_k_features: int = 50,
        examples_per_feature: int = 5
    ):
        """
        Generate comprehensive interpretability report.
        
        Args:
            output_file: Path to save report (markdown format)
            top_k_features: Number of top features to include
            examples_per_feature: Number of examples to show per feature
        """
        # Sort features by max activation
        sorted_features = sorted(
            self.features.items(),
            key=lambda x: x[1]['statistics']['max_activation'],
            reverse=True
        )[:top_k_features]
        
        with open(output_file, 'w') as f:
            f.write("# SAE Feature Interpretability Report\n\n")
            f.write(f"Generated from: `{self.samples_dir}`\n\n")
            
            # Overall summary
            if self.summary:
                f.write("## Collection Summary\n\n")
                f.write(f"- Total samples seen: {self.summary['total_samples_seen']:,}\n")
                f.write(f"- Total samples stored: {self.summary['total_samples_stored']:,}\n")
                f.write(f"- Features tracked: {self.summary['num_features_tracked']}\n")
                f.write(f"- Avg samples per feature: {self.summary['avg_samples_per_feature']:.1f}\n\n")
            
            # Feature details
            f.write(f"## Top {top_k_features} Most Active Features\n\n")
            
            for feature_idx, data in sorted_features:
                stats = data['statistics']
                samples = data['top_samples'][:examples_per_feature]
                
                f.write(f"### Feature {feature_idx}\n\n")
                f.write(f"**Statistics:**\n")
                f.write(f"- Max activation: {stats['max_activation']:.4f}\n")
                f.write(f"- Mean activation: {stats['mean_activation']:.4f}\n")
                f.write(f"- Total activations: {stats['activation_count']}\n")
                f.write(f"- Samples stored: {stats['num_samples']}\n\n")
                
                f.write(f"**Top {examples_per_feature} Examples:**\n\n")
                for i, sample in enumerate(samples, 1):
                    f.write(f"{i}. **[{sample['activation_value']:.3f}]** `{sample['text']}`\n")
                    f.write(f"   - Context: _{sample['context_text']}_\n")
                    f.write(f"   - Position: {sample['position']}\n\n")
                
                f.write("---\n\n")
        
        print(f"\n✅ Interpretability report saved to: {output_file}")
    
    def find_similar_features(self, feature_idx: int, top_k: int = 5):
        """
        Find features with similar activation patterns (based on text similarity).
        
        Args:
            feature_idx: Reference feature
            top_k: Number of similar features to return
        """
        if feature_idx not in self.features:
            print(f"Feature {feature_idx} not found.")
            return
        
        print(f"\n Finding features similar to Feature {feature_idx}...")
        print(f"(Based on activation value distributions)")
        
        ref_stats = self.features[feature_idx]['statistics']
        ref_max = ref_stats['max_activation']
        ref_mean = ref_stats['mean_activation']
        
        similarities = []
        for other_idx, other_data in self.features.items():
            if other_idx == feature_idx:
                continue
            
            other_stats = other_data['statistics']
            other_max = other_stats['max_activation']
            other_mean = other_stats['mean_activation']
            
            # Simple distance metric
            distance = abs(ref_max - other_max) + abs(ref_mean - other_mean)
            similarities.append((other_idx, distance))
        
        # Sort by similarity (smallest distance first)
        similarities.sort(key=lambda x: x[1])
        
        print(f"\nTop {top_k} most similar features:")
        for i, (idx, dist) in enumerate(similarities[:top_k], 1):
            other_stats = self.features[idx]['statistics']
            print(f"{i}. Feature {idx} (distance: {dist:.4f})")
            print(f"   Max: {other_stats['max_activation']:.4f}, Mean: {other_stats['mean_activation']:.4f}")


def main():
    """Example usage of analyzer."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_activation_samples.py <samples_directory>")
        print("\nExample:")
        print("  python analyze_activation_samples.py checkpoints/sae/gemma-2-2b/my_model_activation_samples")
        sys.exit(1)
    
    samples_dir = sys.argv[1]
    
    # Create analyzer
    analyzer = ActivationSampleAnalyzer(samples_dir)
    
    # Generate report
    report_path = os.path.join(samples_dir, "interpretability_report.md")
    analyzer.generate_interpretability_report(
        report_path,
        top_k_features=50,
        examples_per_feature=10
    )
    
    # Print summary
    analyzer.analyze_feature_clustering()
    
    # Show example feature
    if analyzer.features:
        first_feature = list(analyzer.features.keys())[0]
        analyzer.print_feature_report(first_feature, num_examples=5)


if __name__ == "__main__":
    main()

