"""
Command-line script to run absorption evaluation.

Usage:
    python run_evaluation.py --layer 3 --model gemma-2-2b
    python run_evaluation.py --layers 0 1 2 3 --model gemma-2-2b
    python run_evaluation.py --all-layers --model gemma-2-2b
"""

import argparse
import torch
from pathlib import Path

from .config import AbsorptionConfig
from .evaluation import run_absorption_evaluation, evaluate_multiple_layers
from .matryoshka_wrapper import load_matryoshka_transcoder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run absorption score evaluation on Matryoshka transcoders (SAE Bench style)"
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
    
    # Evaluation configuration
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of vocabulary samples for probe training (default: 10000)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for absorption calculation (default: 10)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (default: cuda:0)"
    )
    
    # Output settings
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/absorption",
        help="Output directory (default: results/absorption)"
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
    config = AbsorptionConfig(
        model_name=args.model,
        layers_to_eval=layers_to_eval,
        device=args.device,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        results_dir=args.results_dir,
        verbose=True,
    )
    
    print("\n" + "="*80)
    print("Absorption Score Evaluation (SAE Bench Style)")
    print("="*80)
    print(f"Model: {config.model_name}")
    print(f"Layers: {layers_to_eval}")
    print(f"Prefix level: {args.prefix_level if args.prefix_level is not None else 'Full'}")
    print(f"Device: {config.device}")
    print(f"Samples: {config.num_samples}")
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
            results = run_absorption_evaluation(
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
                print(f"  Mean absorption fraction: {summary['mean_absorption_fraction']:.3f}")
                print(f"  Mean full absorption rate: {summary['mean_full_absorption_rate']:.3f}")
            
        except Exception as e:
            print(f"\n✗ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return


if __name__ == "__main__":
    main()
