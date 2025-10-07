"""
Example script demonstrating activation sample collection for SAE interpretability.

This script shows how to:
1. Enable activation sample collection during training
2. Collect samples for top-activating features
3. Analyze and visualize the collected samples

Based on Anthropic's approach: https://transformer-circuits.pub/2023/monosemantic-features
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformer_lens import HookedTransformer
from models.sae import BatchTopKSAE
from models.activation_store import ActivationsStore
from training.training import train_sae
from utils.config import get_default_cfg, post_init_cfg
from utils.analyze_activation_samples import ActivationSampleAnalyzer


def train_sae_with_sample_collection():
    """Example: Train an SAE with activation sample collection enabled."""
    
    print("="*80)
    print("SAE Training with Activation Sample Collection")
    print("="*80)
    
    # Get default config
    cfg = get_default_cfg()
    
    # Modify for quick demo (reduce training time)
    cfg["model_name"] = "gpt2"  # Smaller model for demo
    cfg["layer"] = 6
    cfg["dict_size"] = 3072  # 4x expansion
    cfg["top_k"] = 32
    cfg["num_tokens"] = int(1e7)  # 10M tokens for demo
    cfg["batch_size"] = 2048
    cfg["checkpoint_freq"] = 5000
    cfg["perf_log_freq"] = 500
    
    # *** ENABLE ACTIVATION SAMPLE COLLECTION ***
    cfg["save_activation_samples"] = True
    cfg["sample_collection_freq"] = 500  # Collect every 500 steps
    cfg["max_samples_per_feature"] = 50  # Store top 50 samples per feature
    cfg["sample_context_size"] = 30      # 30 tokens of context
    cfg["sample_activation_threshold"] = 0.3  # Only store activations > 0.3
    cfg["top_features_to_save"] = 50     # Save top 50 most active features
    cfg["samples_per_feature_to_save"] = 10  # Save 10 examples per feature
    
    cfg = post_init_cfg(cfg)
    
    print(f"\nConfiguration:")
    print(f"  Model: {cfg['model_name']}")
    print(f"  Layer: {cfg['layer']}")
    print(f"  Dictionary size: {cfg['dict_size']}")
    print(f"  Top-K: {cfg['top_k']}")
    print(f"  Training tokens: {cfg['num_tokens']:,.0f}")
    print(f"\nSample Collection Settings:")
    print(f"  Enabled: {cfg['save_activation_samples']}")
    print(f"  Collection frequency: every {cfg['sample_collection_freq']} steps")
    print(f"  Samples per feature: {cfg['max_samples_per_feature']}")
    print(f"  Context size: {cfg['sample_context_size']} tokens")
    print(f"  Activation threshold: {cfg['sample_activation_threshold']}")
    
    # Load model
    print(f"\nLoading model...")
    model = HookedTransformer.from_pretrained(
        cfg["model_name"],
        device=cfg["device"]
    )
    
    # Create activation store
    print(f"Creating activation store...")
    activation_store = ActivationsStore(model, cfg)
    
    # Create SAE
    print(f"Creating SAE...")
    sae = BatchTopKSAE(cfg)
    
    # Train SAE (sample collection happens automatically)
    print(f"\n{'='*80}")
    print("Starting training with sample collection...")
    print(f"{'='*80}\n")
    
    train_sae(sae, activation_store, model, cfg)
    
    print(f"\n{'='*80}")
    print("Training completed!")
    print(f"{'='*80}")
    
    # Get sample directory path
    sample_dir = f"checkpoints/sae/{cfg['model_name']}/{cfg['name']}_activation_samples"
    
    return sample_dir


def analyze_collected_samples(sample_dir):
    """Example: Analyze collected activation samples."""
    
    print(f"\n{'='*80}")
    print("Analyzing Collected Activation Samples")
    print(f"{'='*80}\n")
    
    if not os.path.exists(sample_dir):
        print(f"Sample directory not found: {sample_dir}")
        print("Please run training with sample collection first.")
        return
    
    # Create analyzer
    analyzer = ActivationSampleAnalyzer(sample_dir)
    
    # Print overall statistics
    print("\n1. Overall Feature Statistics")
    print("-"*80)
    analyzer.analyze_feature_clustering()
    
    # Print detailed report for top features
    print("\n2. Detailed Feature Reports")
    print("-"*80)
    
    # Show top 3 most active features
    if analyzer.summary and 'top_features_by_activation' in analyzer.summary:
        top_features = analyzer.summary['top_features_by_activation'][:3]
        
        for i, feature_info in enumerate(top_features, 1):
            feature_idx = feature_info['feature_idx']
            print(f"\n--- Feature {feature_idx} (Rank #{i}) ---")
            analyzer.print_feature_report(feature_idx, num_examples=5)
    
    # Generate markdown report
    print("\n3. Generating Interpretability Report")
    print("-"*80)
    report_path = os.path.join(sample_dir, "interpretability_report.md")
    analyzer.generate_interpretability_report(
        report_path,
        top_k_features=50,
        examples_per_feature=10
    )
    
    # Compare similar features
    if len(analyzer.features) >= 2:
        print("\n4. Finding Similar Features")
        print("-"*80)
        first_feature = list(analyzer.features.keys())[0]
        analyzer.find_similar_features(first_feature, top_k=5)
    
    print(f"\n{'='*80}")
    print("Analysis Complete!")
    print(f"Full report saved to: {report_path}")
    print(f"{'='*80}\n")


def quick_demo():
    """Quick demonstration using pre-collected samples (if available)."""
    
    print("="*80)
    print("Activation Sample Collection - Quick Demo")
    print("="*80)
    print("\nThis demo shows how to collect and analyze activation samples")
    print("for SAE feature interpretability, following Anthropic's approach.")
    print("="*80)
    
    # Check if we have pre-existing samples
    checkpoint_dirs = [
        "checkpoints/sae",
        "checkpoints/transcoder"
    ]
    
    sample_dirs = []
    for base_dir in checkpoint_dirs:
        if os.path.exists(base_dir):
            for root, dirs, files in os.walk(base_dir):
                if "collection_summary.json" in files:
                    sample_dirs.append(root)
    
    if sample_dirs:
        print(f"\nFound {len(sample_dirs)} existing sample collection(s):")
        for i, d in enumerate(sample_dirs, 1):
            print(f"  {i}. {d}")
        
        print("\nAnalyzing first collection...")
        analyze_collected_samples(sample_dirs[0])
    else:
        print("\nNo pre-existing samples found.")
        print("\nTo collect samples, run training with 'save_activation_samples=True'")
        print("in your configuration.")
        print("\nExample:")
        print("  cfg['save_activation_samples'] = True")
        print("  cfg['sample_collection_freq'] = 1000")
        print("  cfg['max_samples_per_feature'] = 100")
        print("\nThen run: python src/scripts/example_activation_sample_collection.py train")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Demonstration of activation sample collection for SAE interpretability"
    )
    parser.add_argument(
        "mode",
        choices=["train", "analyze", "demo"],
        help="Mode: 'train' to train with collection, 'analyze' to analyze existing samples, 'demo' for quick demo"
    )
    parser.add_argument(
        "--sample-dir",
        type=str,
        help="Directory containing activation samples (for 'analyze' mode)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "train":
        sample_dir = train_sae_with_sample_collection()
        print(f"\n✅ Training complete! Samples saved to: {sample_dir}")
        print(f"\nTo analyze, run:")
        print(f"  python {__file__} analyze --sample-dir {sample_dir}")
        
    elif args.mode == "analyze":
        if not args.sample_dir:
            print("Error: --sample-dir required for 'analyze' mode")
            sys.exit(1)
        analyze_collected_samples(args.sample_dir)
        
    elif args.mode == "demo":
        quick_demo()


if __name__ == "__main__":
    main()

