"""
Example: Evaluating Transcoder Interpretability

This example demonstrates how to use the interpretability evaluation tools
to assess your transcoder's quality and compare with baselines.

Run this as a script or use interactively in a notebook.
"""

import torch
from transformer_lens import HookedTransformer
import sys
import os

# Add parent to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.sae import MatryoshkaTranscoder
from src.models.transcoder_activation_store import TranscoderActivationsStore, create_transcoder_config
from src.utils.config import get_default_cfg
from src.eval.interpretability_metrics import (
    InterpretabilityEvaluator,
    print_metrics_summary
)


def example_evaluate_checkpoint():
    """
    Example 1: Evaluate a saved checkpoint.
    """
    print("="*80)
    print("EXAMPLE 1: Evaluate Saved Checkpoint")
    print("="*80)
    
    # Configuration
    checkpoint_path = "checkpoints/transcoder/gemma-2-2b/your_model_15000"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    # Load model
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=device, dtype=dtype)
    
    # Load transcoder (simplified - use actual loading logic from evaluate_interpretability_standalone.py)
    import json
    with open(os.path.join(checkpoint_path, "config.json"), 'r') as f:
        cfg = json.load(f)
    
    transcoder = MatryoshkaTranscoder(cfg).to(device)
    transcoder.load_state_dict(torch.load(os.path.join(checkpoint_path, "sae.pt"), map_location=device))
    transcoder.eval()
    
    # Create activation store for evaluation data
    eval_cfg = get_default_cfg()
    eval_cfg["model_name"] = "gemma-2-2b"
    eval_cfg["device"] = device
    eval_cfg["dtype"] = dtype
    eval_cfg = create_transcoder_config(eval_cfg, source_layer=17, target_layer=17, 
                                       source_site="mlp_in", target_site="mlp_out")
    
    activation_store = TranscoderActivationsStore(model, eval_cfg)
    
    # Create evaluator
    evaluator = InterpretabilityEvaluator(
        absorption_threshold=0.9,  # SAE Bench standard
        absorption_subset_size=8192,
        dead_threshold=20
    )
    
    # Get a batch and evaluate
    source_batch, target_batch = activation_store.next_batch()
    metrics = evaluator.evaluate_transcoder(transcoder, source_batch, target_batch)
    
    # Print results
    print_metrics_summary(metrics, "Your Transcoder")
    
    return metrics


def example_compare_training_runs():
    """
    Example 2: Compare metrics across different training checkpoints.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Compare Multiple Checkpoints")
    print("="*80)
    
    checkpoints = [
        "checkpoints/transcoder/gemma-2-2b/model_5000",
        "checkpoints/transcoder/gemma-2-2b/model_10000",
        "checkpoints/transcoder/gemma-2-2b/model_15000",
    ]
    
    # This would evaluate each checkpoint and compare
    # (Simplified for example - implement full logic as needed)
    
    print("\nCheckpoint Comparison:")
    print(f"{'Step':<10} {'FVU':<10} {'Absorption':<12} {'Dead %':<10}")
    print("-"*45)
    
    # Example results (replace with actual evaluation)
    results = [
        (5000, 0.45, 0.023, 25.3),
        (10000, 0.38, 0.018, 15.2),
        (15000, 0.34, 0.012, 10.1),
    ]
    
    for step, fvu, absorption, dead_pct in results:
        print(f"{step:<10} {fvu:<10.4f} {absorption:<12.4f} {dead_pct:<10.1f}")
    
    print("\n💡 Improvement over training:")
    print(f"  - FVU improved by: {(results[0][1] - results[-1][1]):.4f}")
    print(f"  - Absorption improved by: {(results[0][2] - results[-1][2]):.4f}")
    print(f"  - Dead % reduced by: {(results[0][3] - results[-1][3]):.1f}%")


def example_analyze_absorption_details():
    """
    Example 3: Detailed absorption analysis.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Detailed Absorption Analysis")
    print("="*80)
    
    # Assume we have a transcoder loaded
    # feature_vectors = transcoder.W_dec  # (dict_size, d_model)
    
    # For demonstration, create synthetic feature vectors
    dict_size = 18432
    d_model = 2304
    feature_vectors = torch.randn(dict_size, d_model)
    
    evaluator = InterpretabilityEvaluator(absorption_threshold=0.9)
    
    # Compute with details
    absorption_score, details = evaluator.compute_absorption_score(
        feature_vectors,
        return_details=True
    )
    
    print(f"\nAbsorption Analysis:")
    print(f"  Score: {absorption_score:.4f}")
    print(f"  Features evaluated: {details['n_features_evaluated']:,}")
    print(f"  Total pairs: {details['total_pairs']:,}")
    print(f"  Pairs above threshold: {details['pairs_above_threshold']:,}")
    print(f"  Mean similarity: {details['mean_similarity']:.4f}")
    print(f"  Max similarity: {details['max_similarity']:.4f}")
    print(f"  Median similarity: {details['median_similarity']:.4f}")
    
    print("\n💡 Interpretation:")
    if absorption_score < 0.01:
        print("  ✅ Excellent: Features are diverse and interpretable")
    elif absorption_score < 0.05:
        print("  ⚠️  Moderate: Some feature redundancy present")
    else:
        print("  ❌ High: Significant feature absorption, reducing interpretability")


def example_batch_evaluation():
    """
    Example 4: Evaluate over multiple batches for stable metrics.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Multi-Batch Evaluation")
    print("="*80)
    
    # This demonstrates the pattern used in compare_interpretability.py
    # for stable metric computation over many batches
    
    print("\nEvaluating over 1000 batches...")
    print("This provides more stable and reliable metrics.")
    
    # Pseudo-code pattern:
    """
    accumulators = {
        'mse': 0.0,
        'fvu': 0.0,
        'l0_values': [],
        'total_samples': 0
    }
    
    for i in range(1000):
        source, target = activation_store.next_batch()
        sparse_features = transcoder.encode(source)
        reconstruction = transcoder.decode(sparse_features)
        
        # Accumulate metrics
        mse = F.mse_loss(reconstruction, target).item()
        accumulators['mse'] += mse * source.shape[0]
        accumulators['total_samples'] += source.shape[0]
        # ... etc
    
    # Compute averages
    final_mse = accumulators['mse'] / accumulators['total_samples']
    """
    
    print("\n✓ Multi-batch evaluation provides:")
    print("  - Reduced variance in metrics")
    print("  - More representative absorption scores")
    print("  - Stable L0 distribution estimates")


def example_full_pipeline():
    """
    Example 5: Complete evaluation pipeline.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Complete Evaluation Pipeline")
    print("="*80)
    
    print("\nStep-by-step evaluation workflow:")
    print("\n1. Load checkpoint")
    print("   checkpoint = load_transcoder('checkpoints/...')")
    
    print("\n2. Create evaluation dataset")
    print("   activation_store = TranscoderActivationsStore(model, cfg)")
    
    print("\n3. Initialize evaluator")
    print("   evaluator = InterpretabilityEvaluator()")
    
    print("\n4. Evaluate over batches")
    print("   metrics = evaluate_model_on_batches(...)")
    
    print("\n5. Analyze results")
    print("   print_metrics_summary(metrics)")
    
    print("\n6. Save results")
    print("   save_metrics(metrics, 'results/metrics.json')")
    
    print("\n7. Compare with baseline")
    print("   compare_with_google_transcoder(...)")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("INTERPRETABILITY EVALUATION EXAMPLES")
    print("="*80)
    print("\nThis file contains examples of how to use the interpretability")
    print("evaluation tools. Each function demonstrates a different use case.")
    print("\nTo run a specific example, uncomment the corresponding line below:")
    print("="*80 + "\n")
    
    # Uncomment to run specific examples:
    
    # example_evaluate_checkpoint()
    # example_compare_training_runs()
    example_analyze_absorption_details()
    # example_batch_evaluation()
    # example_full_pipeline()
    
    print("\n" + "="*80)
    print("For full evaluation, use:")
    print("  python src/eval/evaluate_interpretability_standalone.py --checkpoint <path>")
    print("\nFor comparison with Google:")
    print("  python src/eval/compare_interpretability.py --ours_checkpoint <path> --google_dir <path>")
    print("="*80 + "\n")

