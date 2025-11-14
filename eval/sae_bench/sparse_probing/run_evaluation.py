"""
Command-line script to run sparse probing evaluation.

Usage:
    python run_evaluation.py --layer 3 --model gemma-2-2b
    python run_evaluation.py --layers 0 1 2 3 --model gemma-2-2b
    python run_evaluation.py --all-layers --model gemma-2-2b
"""

import argparse
import sys
import os
import torch
from pathlib import Path

# Add paths for imports (similar to absorption module)
_current_file = os.path.abspath(__file__)
_sparse_probing_dir = os.path.dirname(_current_file)
_absorption_dir = os.path.join(os.path.dirname(_sparse_probing_dir), "absorption")
_project_root = os.path.abspath(os.path.join(_sparse_probing_dir, "..", "..", ".."))
_src_path = os.path.join(_project_root, "src")

# Add to path - sparse_probing first so its modules are found
if _sparse_probing_dir not in sys.path:
    sys.path.insert(0, _sparse_probing_dir)
if _absorption_dir not in sys.path:
    sys.path.insert(0, _absorption_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

# Now import - use direct imports when run as script
# Import from sparse_probing directory explicitly
import importlib.util
spec = importlib.util.spec_from_file_location("sparse_probing_config", os.path.join(_sparse_probing_dir, "config.py"))
sparse_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sparse_config)
SparseProbingConfig = sparse_config.SparseProbingConfig

spec = importlib.util.spec_from_file_location("sparse_probing_evaluation", os.path.join(_sparse_probing_dir, "evaluation.py"))
sparse_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sparse_eval)
evaluate_feature_probing = sparse_eval.evaluate_feature_probing
evaluate_multiple_layers = sparse_eval.evaluate_multiple_layers

from matryoshka_wrapper import load_matryoshka_transcoder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run sparse probing evaluation on Matryoshka transcoders"
    )
    
    # Model and layer selection
    parser.add_argument(
        "--model",
        type=str,
        default="gemma-2-2b",
        help="Model name (default: gemma-2-2b)"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Single layer to evaluate"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Multiple layers to evaluate"
    )
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Evaluate all available layers (0-17)"
    )
    
    # Transcoder configuration
    parser.add_argument(
        "--prefix-level",
        type=int,
        default=None,
        help="Matryoshka prefix level to use (0-4, None=full)"
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="final.pt",
        help="Checkpoint filename (default: final.pt)"
    )
    
    # Evaluation configuration
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of vocabulary samples (default: 10000)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (default: cuda:0)"
    )
    
    # Feature selection
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Maximum number of features to evaluate (default: all)"
    )
    parser.add_argument(
        "--min-activation-rate",
        type=float,
        default=0.01,
        help="Minimum activation rate to include feature (default: 0.01)"
    )
    parser.add_argument(
        "--top-k-features",
        type=int,
        default=10,
        help="Number of top features to report per concept (default: 10)"
    )
    
    # Probe training
    parser.add_argument(
        "--probe-l1-penalty",
        type=float,
        default=0.01,
        help="L1 penalty for probe training (default: 0.01)"
    )
    parser.add_argument(
        "--train-test-split",
        type=float,
        default=0.8,
        help="Train/test split ratio (default: 0.8)"
    )
    
    # Output settings
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/sparse_probing",
        help="Output directory (default: results/sparse_probing)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to disk"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine which layers to evaluate
    if args.all_layers:
        layers_to_eval = list(range(18))  # Layers 0-17
    elif args.layers is not None:
        layers_to_eval = args.layers
    elif args.layer is not None:
        layers_to_eval = [args.layer]
    else:
        print("Error: Must specify --layer, --layers, or --all-layers")
        return
    
    # Check CUDA availability
    if "cuda" in args.device and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Create configuration
    config = SparseProbingConfig(
        model_name=args.model,
        layers_to_eval=layers_to_eval,
        device=args.device,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_features=args.max_features,
        min_feature_activation_rate=args.min_activation_rate,
        top_k_features=args.top_k_features,
        probe_l1_penalty=args.probe_l1_penalty,
        train_test_split=args.train_test_split,
        results_dir=args.results_dir,
        verbose=True,
    )
    
    print("\n" + "="*80)
    print("Sparse Probing Evaluation")
    print("="*80)
    print(f"Model: {config.model_name}")
    print(f"Layers: {layers_to_eval}")
    print(f"Prefix level: {args.prefix_level if args.prefix_level is not None else 'Full'}")
    print(f"Device: {config.device}")
    print(f"Samples: {config.num_samples}")
    print(f"Max features: {config.max_features if config.max_features else 'All'}")
    print(f"Top K features per concept: {config.top_k_features}")
    print("="*80 + "\n")
    
    # Run evaluation
    if len(layers_to_eval) == 1:
        # Single layer evaluation
        layer = layers_to_eval[0]
        
        try:
            # Load transcoder
            print(f"Loading transcoder for layer {layer}...")
            transcoder_wrapper = load_matryoshka_transcoder(
                model_name=args.model,
                layer=layer,
                prefix_level=args.prefix_level,
                device=args.device,
            )
            
            # Run evaluation
            results = evaluate_feature_probing(
                config=config,
                transcoder_wrapper=transcoder_wrapper,
                save_results=not args.no_save,
            )
            
            print("\n✓ Evaluation completed successfully!")
            
        except Exception as e:
            print(f"\n✗ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return
    
    else:
        # Multi-layer evaluation
        print(f"Evaluating {len(layers_to_eval)} layers...")
        
        try:
            results = evaluate_multiple_layers(
                config=config,
                layers=layers_to_eval,
                prefix_level=args.prefix_level,
            )
            
            print(f"\n✓ Evaluated {len(results)}/{len(layers_to_eval)} layers successfully!")
            
            # Print summary
            print("\n" + "="*80)
            print("Summary Across Layers")
            print("="*80)
            
            for layer in sorted(results.keys()):
                summary = results[layer]['summary']
                print(f"\nLayer {layer}:")
                print(f"  Mean max F1: {summary['mean_max_f1']:.3f}")
                print(f"  Mean avg F1: {summary['mean_avg_f1']:.3f}")
                print(f"  Features evaluated: {summary['n_features_evaluated']}/{summary['n_features_total']}")
            
        except Exception as e:
            print(f"\n✗ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return


if __name__ == "__main__":
    main()

