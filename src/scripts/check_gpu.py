"""
GPU diagnostic script to check memory and status.

Run this to diagnose GPU issues before evaluation.
"""

import torch
import subprocess
import psutil
import os


def check_gpu_status():
    """Check GPU status and memory."""
    print("=" * 70)
    print("GPU DIAGNOSTIC")
    print("=" * 70)
    
    # Check CUDA availability
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            props = torch.cuda.get_device_properties(i)
            print(f"  Name: {props.name}")
            print(f"  Total Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1e9:.1f} GB")
            print(f"  Cached: {torch.cuda.memory_reserved(i) / 1e9:.1f} GB")
            print(f"  Free: {(props.total_memory - torch.cuda.memory_allocated(i)) / 1e9:.1f} GB")
    else:
        print("❌ CUDA not available!")
        print("   - Check CUDA installation")
        print("   - Check PyTorch CUDA version")
        print("   - Try: pip install torch --index-url https://download.pytorch.org/whl/cu118")
    
    # Check nvidia-smi
    print(f"\nNVIDIA-SMI Status:")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ nvidia-smi working")
            # Extract memory info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'MiB' in line and ('Memory' in line or 'GPU' in line):
                    print(f"  {line.strip()}")
        else:
            print("❌ nvidia-smi failed")
    except Exception as e:
        print(f"❌ nvidia-smi error: {e}")
    
    # Check system memory
    print(f"\nSystem Memory:")
    memory = psutil.virtual_memory()
    print(f"  Total: {memory.total / 1e9:.1f} GB")
    print(f"  Available: {memory.available / 1e9:.1f} GB")
    print(f"  Used: {memory.used / 1e9:.1f} GB")
    print(f"  Percent: {memory.percent}%")
    
    # Check for running processes
    print(f"\nGPU Processes:")
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                for line in lines[1:]:  # Skip header
                    if line.strip():
                        print(f"  {line}")
            else:
                print("  No GPU processes running")
        else:
            print("  Could not query GPU processes")
    except Exception as e:
        print(f"  Error querying processes: {e}")
    
    # Recommendations
    print(f"\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - use CPU evaluation:")
        print("   python src/scripts/run_evaluation_cpu.py")
        return
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    gpu_allocated = torch.cuda.memory_allocated(0) / 1e9
    gpu_free = gpu_memory - gpu_allocated
    
    if gpu_free < 2:
        print("❌ Very low GPU memory (< 2GB free)")
        print("   Solutions:")
        print("   1. Use CPU evaluation: python src/scripts/run_evaluation_cpu.py")
        print("   2. Kill other GPU processes")
        print("   3. Restart to clear GPU memory")
        print("   4. Use smaller model (GPT-2 Small)")
    elif gpu_free < 4:
        print("⚠️  Limited GPU memory (2-4GB free)")
        print("   Solutions:")
        print("   1. Use memory-safe evaluation: python src/scripts/run_evaluation_safe.py")
        print("   2. Use CPU evaluation: python src/scripts/run_evaluation_cpu.py")
        print("   3. Kill other GPU processes")
    elif gpu_free < 8:
        print("⚠️  Moderate GPU memory (4-8GB free)")
        print("   Solutions:")
        print("   1. Use memory-safe evaluation: python src/scripts/run_evaluation_safe.py")
        print("   2. Monitor memory usage during evaluation")
    else:
        print("✅ Good GPU memory (> 8GB free)")
        print("   You can use:")
        print("   1. Standard evaluation: python src/scripts/run_evaluation.py")
        print("   2. Memory-safe evaluation: python src/scripts/run_evaluation_safe.py")
    
    # Model-specific recommendations
    print(f"\nModel-Specific Recommendations:")
    print(f"  GPT-2 Small: Needs ~2GB GPU memory")
    print(f"  GPT-2 Medium: Needs ~4GB GPU memory") 
    print(f"  Gemma-2-2B: Needs ~8GB GPU memory")
    print(f"  Gemma-2-9B: Needs ~16GB GPU memory")
    
    print("=" * 70)


if __name__ == "__main__":
    check_gpu_status()
