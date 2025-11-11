"""
Simple example of running absorption evaluation.

This script demonstrates the basic usage of the absorption evaluation module.
"""

import torch
from config import AbsorptionConfig
from evaluation import run_absorption_evaluation
from matryoshka_wrapper import load_matryoshka_transcoder


def main():
    """Run a simple absorption evaluation on a single layer."""
    
    # Configuration
    MODEL_NAME = "gemma-2-2b"
    LAYER = 3
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"Running absorption evaluation on {MODEL_NAME} layer {LAYER}")
    print(f"Using device: {DEVICE}\n")
    
    # Create evaluation config
    config = AbsorptionConfig(
        model_name=MODEL_NAME,
        layers_to_eval=[LAYER],
        device=DEVICE,
        num_samples=5000,  # Reduced for quick test
        batch_size=32,
        
        # Absorption detection settings
        absorption_threshold=0.3,
        ablation_threshold=0.01,
        use_integrated_gradients=True,  # Faster approximation
        
        # Metrics to calculate
        calculate_precision_recall=True,
        calculate_f1=True,
        calculate_absorption_rate=True,
        
        # Output
        results_dir="results/absorption",
        save_detailed_results=True,
        verbose=True,
    )
    
    try:
        # Load trained Matryoshka transcoder
        print("Loading transcoder...")
        transcoder_wrapper = load_matryoshka_transcoder(
            model_name=MODEL_NAME,
            layer=LAYER,
            prefix_level=None,  # Use full size, or set to 0-4 for specific prefix
            device=DEVICE,
        )
        
        print(f"Transcoder info: {transcoder_wrapper.get_feature_stats()}\n")
        
        # Run evaluation
        results = run_absorption_evaluation(
            config=config,
            transcoder_wrapper=transcoder_wrapper,
            save_results=True,
        )
        
        # Print summary
        print("\n" + "="*80)
        print("Results Summary")
        print("="*80)
        
        summary = results['summary']
        print(f"\nClassification Performance:")
        print(f"  Mean F1: {summary['mean_f1']:.3f} ± {summary['std_f1']:.3f}")
        print(f"  Mean Precision: {summary['mean_precision']:.3f} ± {summary['std_precision']:.3f}")
        print(f"  Mean Recall: {summary['mean_recall']:.3f} ± {summary['std_recall']:.3f}")
        
        if 'mean_absorption_rate' in summary:
            print(f"\nAbsorption Analysis:")
            print(f"  Mean absorption rate: {summary['mean_absorption_rate']:.3f} ± {summary['std_absorption_rate']:.3f}")
            print(f"  Max absorption rate: {summary['max_absorption_rate']:.3f}")
            print(f"  Min absorption rate: {summary['min_absorption_rate']:.3f}")
        
        # Show some per-letter details
        print(f"\nPer-Letter Examples:")
        for letter in ['a', 's', 'z']:
            if letter in results['results_by_letter']:
                letter_result = results['results_by_letter'][letter]
                print(f"\nLetter '{letter}':")
                print(f"  Feature: {letter_result['feature_idx']}")
                print(f"  F1: {letter_result['classification']['f1']:.3f}")
                if 'absorption' in letter_result:
                    print(f"  Absorption rate: {letter_result['absorption']['absorption_rate']:.3f}")
                    print(f"  Absorbing features: {len(letter_result['absorption']['absorbing_features'])}")
        
        print("\n✓ Evaluation complete!")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure you have trained a transcoder for this layer first:")
        print(f"  python src/scripts/train_all_layers_10k.py")
        
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

