#!/usr/bin/env python3
"""Minimal test of absorption evaluation on layer 8."""

import sys
import os

# Add project root to path
project_root = "/home/datngo/User/matryoshka_sae"
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "eval/sae_bench/absorption"))

# Set up logging to file
log_file = os.path.join(project_root, "absorption_test.log")
log = open(log_file, 'w')

def log_print(msg):
    print(msg)
    log.write(msg + '\n')
    log.flush()

log_print("="*80)
log_print("Absorption Evaluation Test - Layer 8")
log_print("="*80)

# Test imports
log_print("\n[1/6] Testing imports...")
try:
    import torch
    log_print(f"  ✓ PyTorch {torch.__version__}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log_print(f"  ✓ Device: {device}")
except Exception as e:
    log_print(f"  ✗ PyTorch error: {e}")
    log.close()
    sys.exit(1)

try:
    from transformer_lens import HookedTransformer
    log_print("  ✓ TransformerLens")
except Exception as e:
    log_print(f"  ✗ TransformerLens error: {e}")
    log.close()
    sys.exit(1)

try:
    from eval.sae_bench.absorption.config import AbsorptionConfig
    from eval.sae_bench.absorption.matryoshka_wrapper import load_matryoshka_transcoder
    from eval.sae_bench.absorption.evaluation import run_absorption_evaluation
    log_print("  ✓ Absorption modules")
except Exception as e:
    log_print(f"  ✗ Absorption modules error: {e}")
    import traceback
    log.write(traceback.format_exc())
    log.close()
    sys.exit(1)

# Create config
log_print("\n[2/6] Creating configuration...")
config = AbsorptionConfig(
    model_name="gemma-2-2b",
    layers_to_eval=[8],
    device=device,
    num_samples=1000,  # Very small for quick test
    batch_size=8,
    min_tokens_per_letter=30,  # Reduced
    calculate_absorption_rate=False,  # Skip for speed
    verbose=False,
)
log_print(f"  ✓ Config: {config.num_samples} samples, device={device}")

# Load transcoder
log_print("\n[3/6] Loading transcoder...")
try:
    transcoder_wrapper = load_matryoshka_transcoder(
        model_name="gemma-2-2b",
        layer=8,
        prefix_level=None,
        device=device,
    )
    log_print(f"  ✓ Transcoder loaded: {transcoder_wrapper.n_features} features")
except Exception as e:
    log_print(f"  ✗ Loading error: {e}")
    import traceback
    log.write(traceback.format_exc())
    log.close()
    sys.exit(1)

# Run evaluation
log_print("\n[4/6] Running evaluation...")
log_print("  (This may take a few minutes...)")

try:
    results = run_absorption_evaluation(
        config=config,
        transcoder_wrapper=transcoder_wrapper,
        save_results=True,
    )
    log_print("  ✓ Evaluation complete!")
except Exception as e:
    log_print(f"  ✗ Evaluation error: {e}")
    import traceback
    log.write(traceback.format_exc())
    log.close()
    sys.exit(1)

# Print results
log_print("\n[5/6] Results:")
summary = results['summary']
log_print(f"  Mean F1: {summary['mean_f1']:.3f} ± {summary['std_f1']:.3f}")
log_print(f"  Mean Precision: {summary['mean_precision']:.3f} ± {summary['std_precision']:.3f}")
log_print(f"  Mean Recall: {summary['mean_recall']:.3f} ± {summary['std_recall']:.3f}")

log_print("\n[6/6] Example results for letters:")
for letter in ['a', 'm', 's', 'z']:
    if letter in results['results_by_letter']:
        lr = results['results_by_letter'][letter]
        f1 = lr['classification']['f1']
        log_print(f"  Letter '{letter}': F1={f1:.3f}")

log_print("\n" + "="*80)
log_print("✓ TEST COMPLETE!")
log_print("="*80)
log_print(f"\nResults saved to: results/absorption/gemma-2-2b/layer8/")

log.close()
print(f"\n✓ Test completed! Check {log_file} for full output.")

