"""
Simple example of running absorption evaluation.

This script demonstrates the basic usage of the absorption evaluation module.
"""

import torch
from .config import AbsorptionConfig
from .evaluation import run_absorption_evaluation
from .matryoshka_wrapper import load_matryoshka_transcoder


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
        batch_size=10,
        results_dir="results/absorption",
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
        print(f"\nAbsorption Analysis:")
        print(f"  Mean absorption fraction: {summary['mean_absorption_fraction']:.3f} ± {summary['std_absorption_fraction']:.3f}")
        print(f"  Mean full absorption rate: {summary['mean_full_absorption_rate']:.3f} ± {summary['std_full_absorption_rate']:.3f}")
        print(f"  Mean cosine similarity: {summary['mean_cosine_similarity']:.3f} ± {summary['std_cosine_similarity']:.3f}")
        
        # Show some per-letter details
        print(f"\nPer-Letter Examples:")
        for letter in ['a', 's', 'z']:
            if letter in results['results_by_letter']:
                letter_result = results['results_by_letter'][letter]
                print(f"\nLetter '{letter}':")
                print(f"  Main feature: {letter_result['main_feature_idx']}")
                print(f"  Cosine similarity: {letter_result['cosine_similarity']:.3f}")
                print(f"  Mean absorption fraction: {letter_result['mean_absorption_fraction']:.3f}")
                print(f"  Full absorption rate: {letter_result['full_absorption_rate']:.3f}")
        
        print("\n✓ Evaluation complete!")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure you have trained a transcoder for this layer first.")
        
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
