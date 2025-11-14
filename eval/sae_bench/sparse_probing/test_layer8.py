#!/usr/bin/env python3
"""Quick test of sparse probing evaluation on layer 8."""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

print("Starting sparse probing evaluation test for layer 8...")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

try:
    import torch
    print(f"✓ PyTorch imported: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"✗ Error importing PyTorch: {e}")
    sys.exit(1)

try:
    from transformer_lens import HookedTransformer
    print("✓ TransformerLens imported")
except Exception as e:
    print(f"✗ Error importing TransformerLens: {e}")
    sys.exit(1)

try:
    from eval.sae_bench.sparse_probing.config import SparseProbingConfig
    print("✓ Config imported")
except Exception as e:
    print(f"✗ Error importing config: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from eval.sae_bench.absorption.matryoshka_wrapper import load_matryoshka_transcoder
    print("✓ Wrapper imported")
except Exception as e:
    print(f"✗ Error importing wrapper: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from eval.sae_bench.sparse_probing.evaluation import evaluate_feature_probing
    print("✓ Evaluation imported")
except Exception as e:
    print(f"✗ Error importing evaluation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create config
print("\nCreating configuration...")
config = SparseProbingConfig(
    model_name="gemma-2-2b",
    layers_to_eval=[8],
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    num_samples=10000,  # Full evaluation
    batch_size=32,
    max_features=500,  # Evaluate 500 features
    top_k_features=10,
    verbose=True,
)
print(f"✓ Config created: device={config.device}, samples={config.num_samples}")

# Load transcoder
print("\nLoading transcoder for layer 8...")
try:
    transcoder_wrapper = load_matryoshka_transcoder(
        model_name="gemma-2-2b",
        layer=8,
        prefix_level=None,
        device=config.device,
    )
    print(f"✓ Transcoder loaded")
    print(f"  Features: {transcoder_wrapper.n_features}")
    print(f"  Layer: {transcoder_wrapper.layer}")
except Exception as e:
    print(f"✗ Error loading transcoder: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Run evaluation
print("\n" + "="*80)
print("Running sparse probing evaluation...")
print("="*80)

try:
    results = evaluate_feature_probing(
        config=config,
        transcoder_wrapper=transcoder_wrapper,
        save_results=True,
    )
    
    print("\n" + "="*80)
    print("✓ EVALUATION COMPLETE!")
    print("="*80)
    
    # Print summary
    summary = results['summary']
    print(f"\nResults Summary:")
    print(f"  Mean max F1: {summary['mean_max_f1']:.3f} ± {summary['std_max_f1']:.3f}")
    print(f"  Mean avg F1: {summary['mean_avg_f1']:.3f} ± {summary['std_avg_f1']:.3f}")
    print(f"  Features evaluated: {summary['n_features_evaluated']}/{summary['n_features_total']}")
    
    # Print top features for a few letters
    print("\nTop features for sample letters:")
    for letter in ['a', 'e', 's']:
        if letter in results['results_by_letter']:
            top_features = results['results_by_letter'][letter]['top_features']
            print(f"\n  Letter '{letter}':")
            for i, feat in enumerate(top_features[:3]):
                print(f"    {i+1}. Feature {feat['feature_idx']}: F1={feat['test_metrics']['f1']:.3f}")
    
    print("\n✓ Test completed successfully!")
    
except Exception as e:
    print(f"\n✗ Error during evaluation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

