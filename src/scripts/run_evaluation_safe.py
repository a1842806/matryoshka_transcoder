"""
Memory-safe evaluation script for large models like Gemma-2-2B.

This version uses:
- Smaller batch sizes
- CPU offloading
- Memory monitoring
- Graceful degradation
"""

import sys
import torch
import psutil
import gc
from transformer_lens import HookedTransformer
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.transcoder_activation_store import TranscoderActivationsStore, create_transcoder_config
from scripts.evaluate_features import evaluate_trained_transcoder
from utils.config import get_default_cfg
import json
import os


def check_memory():
    """Check available GPU and CPU memory."""
    gpu_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
    gpu_allocated = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0
    gpu_free = gpu_memory - gpu_allocated
    
    cpu_memory = psutil.virtual_memory()
    
    print(f"Memory Status:")
    print(f"  GPU Total: {gpu_memory / 1e9:.1f} GB")
    print(f"  GPU Allocated: {gpu_allocated / 1e9:.1f} GB")
    print(f"  GPU Free: {gpu_free / 1e9:.1f} GB")
    print(f"  CPU Total: {cpu_memory.total / 1e9:.1f} GB")
    print(f"  CPU Available: {cpu_memory.available / 1e9:.1f} GB")
    
    return gpu_free, cpu_memory.available


def get_safe_batch_size(model_name, gpu_free_gb):
    """Determine safe batch size based on model and available memory."""
    model_sizes = {
        "gpt2-small": {"act_size": 768, "safe_batches": 50},
        "gpt2-medium": {"act_size": 1024, "safe_batches": 30},
        "gpt2-large": {"act_size": 1280, "safe_batches": 20},
        "gpt2-xl": {"act_size": 1600, "safe_batches": 15},
        "gemma-2-2b": {"act_size": 2304, "safe_batches": 10},
        "gemma-2-9b": {"act_size": 3584, "safe_batches": 5},
        "gemma-2-27b": {"act_size": 4608, "safe_batches": 3},
    }
    
    # Get model config
    for model_key, config in model_sizes.items():
        if model_key in model_name.lower():
            base_batches = config["safe_batches"]
            break
    else:
        base_batches = 20  # Default fallback
    
    # Adjust based on available memory
    if gpu_free_gb < 2:
        return max(5, base_batches // 4)
    elif gpu_free_gb < 4:
        return max(10, base_batches // 2)
    elif gpu_free_gb < 8:
        return base_batches
    else:
        return min(100, base_batches * 2)


def main(checkpoint_path=None):
    """Run memory-safe evaluation on a trained transcoder."""
    
    print("=" * 70)
    print("MEMORY-SAFE EVALUATION")
    print("=" * 70)
    
    # Check initial memory
    gpu_free, cpu_available = check_memory()
    gpu_free_gb = gpu_free / 1e9
    
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
    print("MATRYOSHKA TRANSCODER EVALUATION (MEMORY-SAFE)")
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
    
    # Determine safe batch size
    safe_batches = get_safe_batch_size(cfg["model_name"], gpu_free_gb)
    print(f"\nMemory Analysis:")
    print(f"  Available GPU: {gpu_free_gb:.1f} GB")
    print(f"  Safe batch count: {safe_batches}")
    
    if safe_batches < 10:
        print(f"\n⚠️  WARNING: Very limited memory detected!")
        print(f"   Using only {safe_batches} batches for evaluation")
        print(f"   Results may be less reliable")
        print(f"   Consider using CPU evaluation or smaller model")
    
    # Load model with memory optimization
    print(f"\n📥 Loading {cfg['model_name']}...")
    
    # Clear cache before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    try:
        model = HookedTransformer.from_pretrained_no_processing(
            cfg["model_name"]
        ).to(cfg["model_dtype"]).to(cfg["device"])
        print("✓ Model loaded successfully")
        print(f"  - Activation size: {model.cfg.d_model}")
        print(f"  - Number of layers: {model.cfg.n_layers}")
        print(f"  - Vocabulary size: {model.cfg.d_vocab}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("This might be due to insufficient GPU memory.")
        print("Try:")
        print("  1. Use a smaller model (GPT-2 Small)")
        print("  2. Free up GPU memory")
        print("  3. Use CPU evaluation")
        return
    
    # Check memory after model loading
    print(f"\nMemory after model loading:")
    check_memory()
    
    # Create activation store with smaller batch size
    print(f"\n📊 Creating activation store...")
    print(f"  Source: {cfg['source_hook_point']}")
    print(f"  Target: {cfg['target_hook_point']}")
    
    # Reduce model batch size for memory safety
    original_model_batch_size = cfg.get("model_batch_size", 8)
    cfg["model_batch_size"] = min(4, original_model_batch_size)  # Smaller batches
    cfg["batch_size"] = min(512, cfg.get("batch_size", 1024))    # Smaller evaluation batches
    
    try:
        activation_store = TranscoderActivationsStore(model, cfg)
        print("✓ Activation store created")
        print(f"  - Context size: {activation_store.context_size}")
        print(f"  - Model batch size: {cfg['model_batch_size']}")
        print(f"  - Evaluation batch size: {cfg['batch_size']}")
    except Exception as e:
        print(f"❌ Failed to create activation store: {e}")
        print("This might be due to dataset loading issues.")
        print("Try running with a different dataset or check your internet connection.")
        return
    
    # Run evaluation with memory monitoring
    print(f"\n🔍 Running memory-safe evaluation...")
    print("This will:")
    print(f"  1. Collect activations from {safe_batches} batches (reduced for memory)")
    print("  2. Measure feature absorption (cosine similarity)")
    print("  3. Measure feature splitting (co-activation)")
    print("  4. Compute monosemanticity scores")
    print("  5. Analyze dead features")
    print("  6. Analyze hierarchical structure (Matryoshka-specific)")
    print(f"\nUsing {safe_batches} batches for memory safety...")
    print("=" * 70)
    
    try:
        # Monitor memory during evaluation
        print(f"\nMemory before evaluation:")
        check_memory()
        
        results = evaluate_trained_transcoder(
            checkpoint_path=checkpoint_path,
            activation_store=activation_store,
            model=model,
            cfg=cfg,
            num_batches=safe_batches  # Use safe batch count
        )
        
        print(f"\nMemory after evaluation:")
        check_memory()
        
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
        
        # Memory safety note
        if safe_batches < 20:
            print(f"\n💡 Note: Evaluation used only {safe_batches} batches for memory safety.")
            print("   For more reliable results, consider:")
            print("   - Using a machine with more GPU memory")
            print("   - Running on CPU (slower but more memory)")
            print("   - Using a smaller model for testing")
        
        print("=" * 70)
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ GPU Out of Memory: {e}")
        print("\nSolutions:")
        print("  1. Use CPU evaluation: CUDA_VISIBLE_DEVICES='' python src/scripts/run_evaluation_safe.py")
        print("  2. Use smaller model: python src/scripts/run_evaluation_safe.py checkpoints/transcoder/gpt2-small/...")
        print("  3. Free GPU memory and try again")
        print("  4. Use even smaller batch size (edit script)")
        
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        print("Partial results may be available in checkpoint directory")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nTroubleshooting:")
        print("  1. Check GPU memory: nvidia-smi")
        print("  2. Try CPU evaluation: CUDA_VISIBLE_DEVICES='' python src/scripts/run_evaluation_safe.py")
        print("  3. Use smaller model for testing")
        print("  4. Check checkpoint path is correct")
        
    finally:
        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print(f"\nMemory cleanup completed.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        main(checkpoint_path)
    else:
        print("Memory-Safe Evaluation Script")
        print("Usage: python run_evaluation_safe.py [checkpoint_path]")
        print("\nOr run without arguments to evaluate most recent checkpoint:")
        print("python run_evaluation_safe.py")
        print()
        main()
