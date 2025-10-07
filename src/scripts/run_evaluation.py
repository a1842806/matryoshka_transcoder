"""
Quick script to evaluate a trained Matryoshka Transcoder.

Usage:
    python run_evaluation.py checkpoints/your_checkpoint_path
"""

import sys
import torch
from transformer_lens import HookedTransformer
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.transcoder_activation_store import TranscoderActivationsStore, create_transcoder_config
from scripts.evaluate_features import evaluate_trained_transcoder
from utils.config import get_default_cfg
import json
import os


def main(checkpoint_path=None):
    """Run evaluation on a trained transcoder."""
    
    if checkpoint_path is None:
        # Find most recent checkpoint in organized structure
        checkpoint_base = "checkpoints"
        if not os.path.exists(checkpoint_base):
            print("❌ No checkpoints directory found!")
            print("Train a model first using: python src/scripts/train_gpt2_transcoder.py")
            return
        
        # Search in organized structure: checkpoints/{model_type}/{base_model}/
        checkpoints = []
        for model_type in ["sae", "transcoder"]:
            type_dir = os.path.join(checkpoint_base, model_type)
            if os.path.exists(type_dir):
                for base_model in os.listdir(type_dir):
                    model_dir = os.path.join(type_dir, base_model)
                    if os.path.isdir(model_dir):
                        for ckpt in os.listdir(model_dir):
                            ckpt_path = os.path.join(model_dir, ckpt)
                            if os.path.isdir(ckpt_path) and os.path.exists(os.path.join(ckpt_path, "config.json")):
                                checkpoints.append(ckpt_path)
        
        if not checkpoints:
            print("❌ No checkpoints found!")
            print("Train a model first using: python src/scripts/train_gpt2_transcoder.py")
            return
        
        # Use most recent
        checkpoint_path = max(checkpoints, key=os.path.getmtime)
        print(f"Using most recent checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Load config from checkpoint
    config_path = os.path.join(checkpoint_path, "config.json")
    if not os.path.exists(config_path):
        print(f"❌ Config not found in checkpoint: {config_path}")
        return
    
    print("=" * 70)
    print("MATRYOSHKA TRANSCODER EVALUATION")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    
    # Load config
    with open(config_path, "r") as f:
        cfg = json.load(f)
    
    # Convert string types back to torch types
    if isinstance(cfg["dtype"], str):
        cfg["dtype"] = getattr(torch, cfg["dtype"].split(".")[-1])
    if isinstance(cfg["model_dtype"], str):
        cfg["model_dtype"] = getattr(torch, cfg["model_dtype"].split(".")[-1])
    if isinstance(cfg["group_sizes"], str):
        cfg["group_sizes"] = json.loads(cfg["group_sizes"])
    
    # Ensure device is set
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nModel: {cfg['model_name']}")
    print(f"Dictionary size: {cfg['dict_size']}")
    print(f"Group sizes: {cfg['group_sizes']}")
    print(f"Device: {cfg['device']}")
    
    # Load model
    print(f"\n📥 Loading {cfg['model_name']}...")
    model = HookedTransformer.from_pretrained_no_processing(
        cfg["model_name"]
    ).to(cfg["model_dtype"]).to(cfg["device"])
    print("✓ Model loaded")
    
    # Create activation store
    print(f"\n📊 Creating activation store...")
    print(f"  Source: {cfg['source_hook_point']}")
    print(f"  Target: {cfg['target_hook_point']}")
    activation_store = TranscoderActivationsStore(model, cfg)
    print("✓ Activation store created")
    
    # Run evaluation
    print(f"\n🔍 Running evaluation...")
    print("This will:")
    print("  1. Collect activations from 100 batches")
    print("  2. Measure feature absorption (cosine similarity)")
    print("  3. Measure feature splitting (co-activation)")
    print("  4. Compute monosemanticity scores")
    print("  5. Analyze dead features")
    print("  6. Analyze hierarchical structure (Matryoshka-specific)")
    print("\nThis may take 10-15 minutes...")
    print("=" * 70)
    
    try:
        results = evaluate_trained_transcoder(
            checkpoint_path=checkpoint_path,
            activation_store=activation_store,
            model=model,
            cfg=cfg,
            num_batches=100
        )
        
        print("\n" + "=" * 70)
        print("✅ EVALUATION COMPLETE")
        print("=" * 70)
        
        # Try to save results
        try:
            eval_dir = f"{checkpoint_path}/evaluation"
            print(f"\nResults saved to: {eval_dir}/results.json")
        except Exception as e:
            print(f"\n⚠️ Note: Could not save full results to JSON: {e}")
            print("Summary results are shown below.")
        
        # Print recommendations
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)
        
        absorption = results["absorption"]["absorption_score"]
        splitting = results["splitting"]["splitting_score"]
        monosemanticity = results["monosemanticity"]["mean"]
        dead_fraction = results["dead_features"]["dead_fraction"]
        
        issues = []
        
        if absorption > 0.15:
            issues.append("❌ High feature absorption detected")
            print("High Absorption:")
            print("  → Many features are learning similar representations")
            print("  → Fix: Increase dict_size or top_k")
        elif absorption > 0.05:
            issues.append("⚠️  Moderate feature absorption")
            print("Moderate Absorption:")
            print("  → Some feature redundancy")
            print("  → Consider increasing dict_size")
        else:
            print("✅ Low absorption - features are diverse")
        
        if splitting > 0.15:
            issues.append("❌ High feature splitting detected")
            print("\nHigh Splitting:")
            print("  → Many features co-activate frequently")
            print("  → Fix: Decrease top_k")
        elif splitting > 0.05:
            issues.append("⚠️  Moderate feature splitting")
            print("\nModerate Splitting:")
            print("  → Some features co-activate")
            print("  → Consider decreasing top_k")
        else:
            print("✅ Low splitting - features are independent")
        
        if monosemanticity < 0.4:
            issues.append("❌ Low monosemanticity")
            print("\nLow Monosemanticity:")
            print("  → Features may be polysemantic (represent multiple concepts)")
            print("  → Fix: Train longer or adjust architecture")
        elif monosemanticity < 0.7:
            issues.append("⚠️  Moderate monosemanticity")
            print("\nModerate Monosemanticity:")
            print("  → Features are somewhat specific")
            print("  → Could be improved with more training")
        else:
            print("✅ High monosemanticity - features are specific")
        
        if dead_fraction > 0.15:
            issues.append("❌ Many dead features")
            print("\nMany Dead Features:")
            print("  → {:.1%} of features never activate".format(dead_fraction))
            print("  → Fix: Increase aux_penalty")
        elif dead_fraction > 0.05:
            issues.append("⚠️  Some dead features")
            print("\nSome Dead Features:")
            print("  → {:.1%} of features never activate".format(dead_fraction))
            print("  → Consider increasing aux_penalty")
        else:
            print("✅ Few dead features - efficient dictionary use")
        
        if not issues:
            print("\n🎉 Model looks great! All metrics are in good range.")
        else:
            print(f"\n⚠️  {len(issues)} issue(s) detected - see recommendations above")
        
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        main(checkpoint_path)
    else:
        print("Usage: python run_evaluation.py [checkpoint_path]")
        print("\nOr run without arguments to evaluate most recent checkpoint:")
        print("python run_evaluation.py")
        print()
        main()

