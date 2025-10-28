"""View activation samples from training results."""

import json
import os
import sys
from typing import List, Dict, Optional
import argparse
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.utils.clean_results import CleanResultsManager
    CLEAN_RESULTS_AVAILABLE = True
except ImportError:
    CLEAN_RESULTS_AVAILABLE = False

def load_samples(samples_dir: str) -> Dict:
    """Load activation samples from directory."""
    summary_path = os.path.join(samples_dir, "collection_summary.json")
    if not os.path.exists(summary_path):
        print(f"❌ No collection_summary.json found in {samples_dir}")
        return {}
    
    with open(summary_path, 'r') as f:
        return json.load(f)

def print_feature_samples(samples_dir: str, feature_idx: int, num_samples: int = 5):
    """Print top activation samples for a specific feature."""
    feature_file = os.path.join(samples_dir, f"feature_{feature_idx:05d}_samples.json")
    
    if not os.path.exists(feature_file):
        print(f"❌ No samples found for feature {feature_idx}")
        return
    
    with open(feature_file, 'r') as f:
        samples = json.load(f)
    
    print(f"\n🔥 Feature {feature_idx} - Top {min(num_samples, len(samples))} Activation Samples:")
    print("=" * 80)
    
    for i, sample in enumerate(samples[:num_samples]):
        print(f"\n{i+1}. Activation: {sample['activation_value']:.4f}")
        print(f"   Text: {sample['text']}")
        print(f"   Context: {sample['context_text']}")

def list_top_features(samples_dir: str, top_k: int = 10):
    """List top features by activation count."""
    summary = load_samples(samples_dir)
    if not summary:
        return
    
    features = summary.get("feature_stats", {})
    if not features:
        print("❌ No feature statistics found")
        return
    
    sorted_features = sorted(features.items(), key=lambda x: x[1]["sample_count"], reverse=True)
    
    print(f"\n📊 Top {top_k} Features by Sample Count:")
    print("=" * 50)
    
    for i, (feature_idx, stats) in enumerate(sorted_features[:top_k]):
        print(f"{i+1:2d}. Feature {feature_idx:4d}: {stats['sample_count']:3d} samples, "
              f"max activation: {stats['max_activation']:.4f}")

def browse_experiments():
    """Browse all experiments."""
    if not CLEAN_RESULTS_AVAILABLE:
        print("❌ Clean results manager not available.")
        return
    
    manager = CleanResultsManager()
    experiments = manager.list_experiments()
    
    if not experiments:
        print("No experiments found.")
        return
    
    print("=" * 80)
    print("🧪 AVAILABLE EXPERIMENTS")
    print("=" * 80)
    
    for model_name, layers in experiments.items():
        print(f"\n📊 {model_name.upper().replace('_', '-')}")
        print("-" * 40)
        
        for layer_num, exp_list in layers.items():
            print(f"\n  Layer {layer_num}:")
            for i, exp in enumerate(exp_list, 1):
                status = "✅ Model" if exp["has_checkpoint"] else "❌ No Model"
                if exp["has_samples"]:
                    status += " + 📊 Samples"
                print(f"    {i}. {exp['name']:<30} {status}")
    
    print("\n" + "=" * 80)
    print("Select an experiment to view activation samples:")
    print("Format: model_layer_experiment_index")
    print("Example: gemma_2_2b_17_1")
    print("Or 'q' to quit")
    
    try:
        choice = input("Choice: ").strip().lower()
        if choice == 'q':
            return
        
        parts = choice.split('_')
        if len(parts) != 4:
            print("Invalid format. Use: model_layer_experiment_index")
            return
        
        model_name = f"{parts[0]}_{parts[1]}_{parts[2]}"
        layer_num = int(parts[3])
        exp_idx = int(parts[4]) - 1
        
        if model_name in experiments and layer_num in experiments[model_name]:
            exp_list = experiments[model_name][layer_num]
            if 0 <= exp_idx < len(exp_list):
                exp = exp_list[exp_idx]
                if exp["has_samples"]:
                    view_samples(os.path.join(exp["path"], "activation_samples"))
                else:
                    print("❌ No activation samples found.")
            else:
                print("❌ Invalid experiment index.")
        else:
            print("❌ Invalid model or layer.")
    
    except (ValueError, IndexError, KeyboardInterrupt):
        print("Invalid input or cancelled.")

def view_samples(samples_dir: str):
    """Interactive sample viewing."""
    if not os.path.exists(samples_dir):
        print(f"❌ Directory not found: {samples_dir}")
        return
    
    summary = load_samples(samples_dir)
    if not summary:
        return
    
    print(f"\n📊 Activation Samples: {os.path.basename(samples_dir)}")
    print("=" * 80)
    
    list_top_features(samples_dir, 10)
    
    while True:
        print("\n" + "=" * 50)
        print("Options:")
        print("  <feature_number> - View samples for feature")
        print("  list - List top features")
        print("  q - Quit")
        
        try:
            choice = input("\nChoice: ").strip().lower()
            
            if choice == 'q':
                break
            elif choice == 'list':
                list_top_features(samples_dir, 10)
            else:
                try:
                    feature_idx = int(choice)
                    print_feature_samples(samples_dir, feature_idx, 5)
                except ValueError:
                    print("Invalid choice. Enter a feature number, 'list', or 'q'.")
        
        except KeyboardInterrupt:
            break

def main():
    parser = argparse.ArgumentParser(description="View activation samples from training results")
    
    parser.add_argument("--model", type=str, help="Model name (e.g., gemma-2-2b)")
    parser.add_argument("--layer", type=int, help="Layer number (e.g., 17)")
    parser.add_argument("--experiment", type=str, help="Experiment name (e.g., 2024-10-27_15k-steps)")
    parser.add_argument("--browse", action="store_true", help="Browse all available experiments")
    parser.add_argument("--sample_dir", type=str, help="Path to activation samples directory")
    
    args = parser.parse_args()
    
    if args.browse:
        browse_experiments()
        return
    
    elif args.model and args.layer and args.experiment:
        if not CLEAN_RESULTS_AVAILABLE:
            print("❌ Clean results manager not available.")
            return
        
        manager = CleanResultsManager()
        experiments = manager.list_experiments(model_name=args.model, layer=args.layer)
        
        model_key = args.model.replace("-", "_").replace(".", "")
        if model_key not in experiments or args.layer not in experiments[model_key]:
            print(f"❌ No experiments found for {args.model} layer {args.layer}")
            return
        
        exp_list = experiments[model_key][args.layer]
        matching_exps = [exp for exp in exp_list if exp["name"] == args.experiment]
        
        if not matching_exps:
            print(f"❌ No experiment found: {args.experiment}")
            return
        
        exp = matching_exps[0]
        if not exp["has_samples"]:
            print("❌ No activation samples found.")
            return
        
        samples_path = os.path.join(exp["path"], "activation_samples")
        view_samples(samples_path)
    
    elif args.sample_dir:
        view_samples(args.sample_dir)
    
    else:
        print("Usage:")
        print("  python view_activation_samples.py --browse")
        print("  python view_activation_samples.py --model gemma-2-2b --layer 17 --experiment 2024-10-27_15k-steps")

if __name__ == "__main__":
    main()