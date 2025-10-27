"""
Standalone Interpretability Evaluation Script.

Evaluate a single transcoder's interpretability metrics without comparison.

Usage:
    python evaluate_interpretability_standalone.py \
        --checkpoint checkpoints/transcoder/gemma-2-2b/my_model_15000 \
        --layer 17 \
        --batches 1000 \
        --device cuda:0
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.sae import MatryoshkaTranscoder
from src.models.transcoder_activation_store import TranscoderActivationsStore, create_transcoder_config
from src.utils.config import get_default_cfg
from src.eval.interpretability_metrics import (
    InterpretabilityEvaluator,
    save_metrics,
    print_metrics_summary
)


def load_transcoder(checkpoint_path: str, device: torch.device, dtype: torch.dtype):
    """Load transcoder from checkpoint."""
    print(f"\n📥 Loading transcoder from: {checkpoint_path}")
    
    # Load config
    config_path = os.path.join(checkpoint_path, "config.json")
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    
    # Fix data types
    if isinstance(cfg.get("dtype"), str):
        try:
            cfg["dtype"] = getattr(torch, cfg["dtype"].split(".")[-1])
        except:
            cfg["dtype"] = dtype
    
    if isinstance(cfg.get("model_dtype"), str):
        try:
            cfg["model_dtype"] = getattr(torch, cfg["model_dtype"].split(".")[-1])
        except:
            cfg["model_dtype"] = dtype
    
    # Fix group_sizes
    if isinstance(cfg.get("group_sizes"), str):
        try:
            cfg["group_sizes"] = json.loads(cfg["group_sizes"])
        except:
            import ast
            cfg["group_sizes"] = ast.literal_eval(cfg["group_sizes"])
    
    # Ensure numeric fields
    for key in ["dict_size", "top_k", "top_k_aux", "n_batches_to_dead"]:
        if key in cfg:
            cfg[key] = int(cfg[key])
    
    for key in ["lr", "aux_penalty", "min_lr"]:
        if key in cfg and isinstance(cfg[key], str) and "/" in cfg[key]:
            num, den = cfg[key].split("/")
            cfg[key] = float(num) / float(den)
        elif key in cfg:
            cfg[key] = float(cfg[key])
    
    cfg["device"] = device
    cfg["dtype"] = dtype
    
    # Create and load transcoder
    transcoder = MatryoshkaTranscoder(cfg).to(device=device, dtype=dtype)
    state_path = os.path.join(checkpoint_path, "sae.pt")
    transcoder.load_state_dict(torch.load(state_path, map_location=device))
    transcoder.eval()
    
    print(f"✓ Transcoder loaded successfully")
    print(f"  - Dictionary size: {cfg['dict_size']:,}")
    print(f"  - Group sizes: {cfg.get('group_sizes', 'N/A')}")
    print(f"  - Top-k: {cfg.get('top_k', 'N/A')}")
    
    return transcoder, cfg


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate transcoder interpretability metrics"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to transcoder checkpoint directory")
    parser.add_argument("--layer", type=int, default=17,
                       help="Layer to evaluate (default: 17)")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu",
                       help="Dataset for evaluation")
    parser.add_argument("--batches", type=int, default=1000,
                       help="Number of batches to evaluate")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size for evaluation")
    parser.add_argument("--seq_len", type=int, default=64,
                       help="Sequence length")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                       help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                       choices=["float16", "bfloat16", "float32"],
                       help="Data type")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: checkpoint_dir/interpretability_eval)")
    
    args = parser.parse_args()
    
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    device = torch.device(args.device)
    dtype = dtype_map[args.dtype]
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.checkpoint, "interpretability_eval")
    
    print("="*80)
    print("🔬 INTERPRETABILITY EVALUATION")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Layer: {args.layer}")
    print(f"Device: {device}")
    print(f"Batches: {args.batches}")
    
    # Load model
    print(f"\n📥 Loading Gemma-2-2B model...")
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=device, dtype=dtype)
    print("✓ Model loaded")
    
    # Load transcoder
    transcoder, transcoder_cfg = load_transcoder(args.checkpoint, device, dtype)
    
    # Create activation store
    cfg = get_default_cfg()
    cfg["model_name"] = "gemma-2-2b"
    cfg["device"] = device
    cfg["dtype"] = dtype
    cfg["dataset_path"] = args.dataset
    cfg["batch_size"] = args.batch_size
    cfg["seq_len"] = args.seq_len
    cfg["model_batch_size"] = 2
    cfg["num_batches_in_buffer"] = 1
    
    cfg = create_transcoder_config(
        cfg,
        source_layer=args.layer,
        target_layer=args.layer,
        source_site="mlp_in",
        target_site="mlp_out"
    )
    
    cfg["source_act_size"] = 2304
    cfg["target_act_size"] = 2304
    cfg["act_size"] = 2304
    
    print(f"\n📊 Creating activation store...")
    activation_store = TranscoderActivationsStore(model, cfg)
    print("✓ Activation store created")
    
    # Create evaluator
    evaluator = InterpretabilityEvaluator(
        absorption_threshold=0.9,
        absorption_subset_size=8192,
        dead_threshold=transcoder_cfg.get("n_batches_to_dead", 20)
    )
    
    # Evaluate over batches
    print(f"\n🔍 Evaluating over {args.batches} batches...")
    
    # Import the evaluation function from compare script
    from src.eval.compare_interpretability import evaluate_model_on_batches
    
    metrics = evaluate_model_on_batches(
        transcoder,
        activation_store,
        evaluator,
        args.batches,
        is_google=False
    )
    
    # Print results
    print_metrics_summary(metrics, "Your Transcoder")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    save_metrics(metrics, os.path.join(args.output_dir, "metrics.json"))
    
    # Also save a summary text file
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("INTERPRETABILITY EVALUATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Layer: {args.layer}\n")
        f.write(f"Evaluation batches: {args.batches}\n\n")
        
        f.write("KEY METRICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"FVU (lower is better):              {metrics.fvu:.4f}\n")
        f.write(f"Absorption Score (lower is better): {metrics.absorption_score:.4f}\n")
        f.write(f"Dead Features:                      {metrics.dead_features_percentage:.1f}%\n")
        f.write(f"Mean L0 Sparsity:                   {metrics.mean_l0:.1f}\n")
        f.write(f"Cosine Similarity:                  {metrics.cosine_similarity:.4f}\n\n")
        
        f.write("INTERPRETATION:\n")
        f.write("-"*80 + "\n")
        
        if metrics.fvu < 0.1:
            f.write("✅ Reconstruction: EXCELLENT (FVU < 0.1)\n")
        elif metrics.fvu < 0.3:
            f.write("⚠️  Reconstruction: MODERATE (0.1 ≤ FVU < 0.3)\n")
        else:
            f.write("❌ Reconstruction: NEEDS IMPROVEMENT (FVU ≥ 0.3)\n")
        
        if metrics.absorption_score < 0.01:
            f.write("✅ Interpretability: EXCELLENT (absorption < 0.01)\n")
        elif metrics.absorption_score < 0.05:
            f.write("⚠️  Interpretability: MODERATE (0.01 ≤ absorption < 0.05)\n")
        else:
            f.write("❌ Interpretability: HIGH REDUNDANCY (absorption ≥ 0.05)\n")
        
        if metrics.dead_features_percentage < 10:
            f.write("✅ Feature Utilization: EXCELLENT (< 10% dead)\n")
        elif metrics.dead_features_percentage < 30:
            f.write("⚠️  Feature Utilization: MODERATE (10-30% dead)\n")
        else:
            f.write("❌ Feature Utilization: POOR (> 30% dead)\n")
    
    print(f"\n✅ Results saved to: {args.output_dir}")
    print(f"   - metrics.json: Full metrics in JSON format")
    print(f"   - summary.txt: Human-readable summary")
    
    # Print improvement suggestions
    print("\n" + "="*80)
    print("💡 IMPROVEMENT SUGGESTIONS")
    print("="*80)
    
    if metrics.fvu > 0.3:
        print("\n🎯 High FVU (> 0.3) - To improve reconstruction:")
        print("  1. Increase dictionary size")
        print("  2. Adjust learning rate (try 4e-4 to 6e-4)")
        print("  3. Use learning rate warmup + cosine decay")
        print("  4. Increase top-k for more active features")
    
    if metrics.absorption_score > 0.05:
        print("\n📊 High Absorption (> 0.05) - To improve interpretability:")
        print("  1. Add diversity regularization during training")
        print("  2. Try different initialization schemes")
        print("  3. Adjust Matryoshka group sizes")
    
    if metrics.dead_features_percentage > 30:
        print("\n🔧 High Dead Features (> 30%) - To improve utilization:")
        print("  1. Lower aux_penalty (try 1/64 instead of 1/32)")
        print("  2. Reduce top_k_aux (256 instead of 512)")
        print("  3. Increase learning rate during warmup")
        print("  4. Adjust n_batches_to_dead threshold")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

