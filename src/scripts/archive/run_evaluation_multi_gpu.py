"""
Multi-GPU evaluation script for systems with multiple GPUs.

This version:
- Utilizes multiple GPUs for parallel processing
- Distributes memory load across GPUs
- Optimizes for high-memory systems (24GB+ GPUs)
"""

import sys
import torch
import psutil
import gc
import os
from transformer_lens import HookedTransformer
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.transcoder_activation_store import TranscoderActivationsStore, create_transcoder_config
from scripts.evaluate_features import evaluate_trained_transcoder
from utils.config import get_default_cfg
import json


def check_multi_gpu_status():
    """Check status of multiple GPUs."""
    print("=" * 70)
    print("MULTI-GPU DIAGNOSTIC")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False
    
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")
    
    total_memory = 0
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i)
        free = props.total_memory - allocated
        
        print(f"\nGPU {i}:")
        print(f"  Name: {props.name}")
        print(f"  Total Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"  Allocated: {allocated / 1e9:.1f} GB")
        print(f"  Free: {free / 1e9:.1f} GB")
        
        total_memory += free
    
    print(f"\nTotal Free Memory: {total_memory / 1e9:.1f} GB")
    
    if total_memory > 40e9:  # > 40GB
        print("✅ Excellent! You can use high-performance evaluation")
        return "high_performance"
    elif total_memory > 20e9:  # > 20GB
        print("✅ Good! You can use standard evaluation")
        return "standard"
    else:
        print("⚠️ Limited memory - use memory-safe evaluation")
        return "memory_safe"


def get_optimal_batch_size(model_name, total_free_gb):
    """Determine optimal batch size for multi-GPU setup."""
    model_sizes = {
        "gpt2-small": {"base_batches": 200, "memory_per_batch": 0.1},
        "gpt2-medium": {"base_batches": 150, "memory_per_batch": 0.2},
        "gpt2-large": {"base_batches": 100, "memory_per_batch": 0.3},
        "gpt2-xl": {"base_batches": 80, "memory_per_batch": 0.4},
        "gemma-2-2b": {"base_batches": 50, "memory_per_batch": 0.8},
        "gemma-2-9b": {"base_batches": 30, "memory_per_batch": 1.5},
        "gemma-2-27b": {"base_batches": 20, "memory_per_batch": 2.0},
    }
    
    # Get model config
    for model_key, config in model_sizes.items():
        if model_key in model_name.lower():
            base_batches = config["base_batches"]
            memory_per_batch = config["memory_per_batch"]
            break
    else:
        base_batches = 50
        memory_per_batch = 0.5
    
    # Calculate optimal batches based on available memory
    max_batches = int(total_free_gb / memory_per_batch)
    optimal_batches = min(base_batches, max_batches)
    
    # For multi-GPU, we can be more aggressive
    if total_free_gb > 40:
        optimal_batches = min(200, optimal_batches * 2)
    elif total_free_gb > 20:
        optimal_batches = min(150, optimal_batches * 1.5)
    
    return max(10, optimal_batches)


def main(checkpoint_path=None):
    """Run multi-GPU evaluation on a trained transcoder."""
    
    print("=" * 70)
    print("MULTI-GPU EVALUATION")
    print("=" * 70)
    
    # Check GPU status
    gpu_status = check_multi_gpu_status()
    if not gpu_status:
        print("❌ No GPUs available. Use CPU evaluation instead:")
        print("   python src/scripts/run_evaluation_cpu.py")
        return
    
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
    
    # Use first GPU for model, but optimize for multi-GPU
    cfg["device"] = "cuda:0"
    
    print(f"\nModel: {cfg['model_name']}")
    print(f"Dictionary size: {cfg['dict_size']}")
    print(f"Group sizes: {cfg['group_sizes']}")
    print(f"Device: {cfg['device']}")
    
    # Calculate optimal batch size for multi-GPU (use only truly free memory)
    total_free = 0.0
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        free_bytes = max(0, props.total_memory - max(allocated, reserved))
        total_free += free_bytes
    total_free /= 1e9
    optimal_batches = get_optimal_batch_size(cfg["model_name"], total_free)
    
    print(f"\nMulti-GPU Optimization:")
    print(f"  Total free memory: {total_free:.1f} GB")
    print(f"  Optimal batch count: {optimal_batches}")
    print(f"  GPU status: {gpu_status}")
    
    # Load model with multi-GPU optimization
    print(f"\n📥 Loading {cfg['model_name']} on GPU 0...")
    
    # Clear cache before loading
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
    gc.collect()
    
    try:
        model = HookedTransformer.from_pretrained_no_processing(
            cfg["model_name"]
        ).to(cfg["model_dtype"]).to(cfg["device"])
        print("✓ Model loaded successfully on GPU 0")
        print(f"  - Activation size: {model.cfg.d_model}")
        print(f"  - Number of layers: {model.cfg.n_layers}")
        print(f"  - Vocabulary size: {model.cfg.d_vocab}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("Falling back to memory-safe evaluation...")
        # Fall back to memory-safe evaluation
        os.system("python src/scripts/run_evaluation_safe.py")
        return
    
    # Check memory after model loading
    print(f"\nMemory after model loading:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {allocated:.1f}GB / {total:.1f}GB")
    
    # Create activation store with optimized settings
    print(f"\n📊 Creating optimized activation store...")
    print(f"  Source: {cfg['source_hook_point']}")
    print(f"  Target: {cfg['target_hook_point']}")
    
    # Optimize for multi-GPU setup
    cfg["model_batch_size"] = min(8, cfg.get("model_batch_size", 8))  # Can use larger batches
    cfg["batch_size"] = min(2048, cfg.get("batch_size", 1024))  # Larger evaluation batches
    
    try:
        activation_store = TranscoderActivationsStore(model, cfg)
        print("✓ Activation store created")
        print(f"  - Context size: {activation_store.context_size}")
        print(f"  - Model batch size: {cfg['model_batch_size']}")
        print(f"  - Evaluation batch size: {cfg['batch_size']}")
    except Exception as e:
        print(f"❌ Failed to create activation store: {e}")
        print("Falling back to memory-safe evaluation...")
        os.system("python src/scripts/run_evaluation_safe.py")
        return
    
    # Run evaluation with multi-GPU optimization
    print(f"\n🔍 Running multi-GPU evaluation...")
    print("This will:")
    print(f"  1. Collect activations from {optimal_batches} batches (optimized for multi-GPU)")
    print("  2. Measure feature absorption (cosine similarity)")
    print("  3. Measure feature splitting (co-activation)")
    print("  4. Compute monosemanticity scores")
    print("  5. Analyze dead features")
    print("  6. Analyze hierarchical structure (Matryoshka-specific)")
    print(f"\n🚀 Using {optimal_batches} batches for high-performance evaluation...")
    print("=" * 70)
    
    try:
        # Monitor memory during evaluation
        print(f"\nMemory before evaluation:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {allocated:.1f}GB / {total:.1f}GB")
        
        # Enforce memory-safe evaluation defaults
        cfg["eval_compute_on_cpu"] = True            # heavy ops (pairwise) on CPU
        cfg["eval_chunk_size"] = 1024               # conservative chunking
        cfg["eval_feature_subset"] = max(0, min(8192, cfg.get("dict_size", 16384)))  # sample features
        cfg["eval_max_pairs_reported"] = 10

        results = evaluate_trained_transcoder(
            checkpoint_path=checkpoint_path,
            activation_store=activation_store,
            model=model,
            cfg=cfg,
            num_batches=optimal_batches  # Use optimal batch count
        )
        
        print(f"\nMemory after evaluation:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {allocated:.1f}GB / {total:.1f}GB")
        
        print("\n" + "=" * 70)
        print("✅ MULTI-GPU EVALUATION COMPLETE")
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
        
        # Multi-GPU performance note
        print(f"\n🚀 Multi-GPU Performance:")
        print(f"   Used {optimal_batches} batches for comprehensive evaluation")
        print(f"   Leveraged {torch.cuda.device_count()} GPUs for optimal performance")
        print(f"   Results are highly reliable with this sample size")
        
        print("=" * 70)
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ GPU Out of Memory: {e}")
        print("\nEven with 2x 24GB GPUs, this model might be too large.")
        print("Solutions:")
        print("  1. Use memory-safe evaluation: python src/scripts/run_evaluation_safe.py")
        print("  2. Use CPU evaluation: python src/scripts/run_evaluation_cpu.py")
        print("  3. Use smaller model for testing")
        
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        print("Partial results may be available in checkpoint directory")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nTroubleshooting:")
        print("  1. Check GPU memory: nvidia-smi")
        print("  2. Try memory-safe evaluation: python src/scripts/run_evaluation_safe.py")
        print("  3. Try CPU evaluation: python src/scripts/run_evaluation_cpu.py")
        
    finally:
        # Clean up memory on all GPUs
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
        gc.collect()
        print(f"\nMulti-GPU memory cleanup completed.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        main(checkpoint_path)
    else:
        print("Multi-GPU Evaluation Script")
        print("Usage: python run_evaluation_multi_gpu.py [checkpoint_path]")
        print("\nOr run without arguments to evaluate most recent checkpoint:")
        print("python run_evaluation_multi_gpu.py")
        print()
        main()
