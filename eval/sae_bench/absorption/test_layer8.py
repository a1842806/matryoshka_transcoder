#!/usr/bin/env python3
"""Quick test of absorption evaluation on layer 8."""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

print("Starting absorption evaluation test for layer 8...")
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
    from .config import AbsorptionConfig
    print("✓ Config imported")
except Exception as e:
    print(f"✗ Error importing config: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from .matryoshka_wrapper import load_matryoshka_transcoder
    print("✓ Wrapper imported")
except Exception as e:
    print(f"✗ Error importing wrapper: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from .evaluation import run_absorption_evaluation
    print("✓ Evaluation imported")
except Exception as e:
    print(f"✗ Error importing evaluation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create config
print("\nCreating configuration...")
config = AbsorptionConfig(
    model_name="gemma-2-2b",
    layers_to_eval=[8],
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    num_samples=2000,  # Small for quick test
    batch_size=10,
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
print("Running absorption evaluation...")
print("="*80)

try:
    results = run_absorption_evaluation(
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
    print(f"  Mean absorption fraction: {summary['mean_absorption_fraction']:.3f} ± {summary['std_absorption_fraction']:.3f}")
    print(f"  Mean full absorption rate: {summary['mean_full_absorption_rate']:.3f} ± {summary['std_full_absorption_rate']:.3f}")
    print(f"  Mean cosine similarity: {summary['mean_cosine_similarity']:.3f} ± {summary['std_cosine_similarity']:.3f}")
    
    print("\n✓ Test completed successfully!")
    
except Exception as e:
    print(f"\n✗ Error during evaluation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
