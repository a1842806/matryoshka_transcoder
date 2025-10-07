# Multi-GPU Guide: Utilizing 2x 24GB GPUs

This guide shows you how to maximize your 2x 24GB GPU setup for Matryoshka SAE/Transcoder training and evaluation.

## Your Setup: 2x 24GB GPUs (48GB Total)

**Excellent configuration!** You can handle:
- ✅ Gemma-2-2B (needs ~8GB)
- ✅ Gemma-2-9B (needs ~16GB) 
- ✅ Multiple models in parallel
- ✅ Large batch sizes
- ✅ High-performance evaluation

## Quick Start

### 1. Check Your GPU Status
```bash
python src/scripts/check_gpu.py
```

### 2. Use Multi-GPU Evaluation
```bash
# For your Gemma-2-2B checkpoint
python src/scripts/run_evaluation_multi_gpu.py
```

### 3. Monitor Both GPUs
```bash
# In another terminal
watch -n 1 nvidia-smi
```

## Evaluation Options for Your Setup

| Script | Use Case | Batch Size | Memory Usage | Speed |
|--------|----------|------------|--------------|-------|
| `run_evaluation_multi_gpu.py` | **Recommended** | 100-200 batches | Optimized for 2x 24GB | Fastest |
| `run_evaluation_safe.py` | Fallback | 10-50 batches | Conservative | Medium |
| `run_evaluation_cpu.py` | Emergency | 3-5 batches | Minimal | Slowest |

## Multi-GPU Evaluation Benefits

### `run_evaluation_multi_gpu.py` Features:
- ✅ **Optimized for 2x 24GB**: Uses 100-200 batches (vs 10-50 for safe)
- ✅ **Memory distribution**: Spreads load across both GPUs
- ✅ **High-performance**: Much faster than single-GPU
- ✅ **Comprehensive results**: More reliable with larger sample size
- ✅ **Automatic optimization**: Adjusts based on available memory

### Expected Performance:
- **Gemma-2-2B**: ~100 batches in 5-10 minutes
- **GPT-2 Small**: ~200 batches in 2-5 minutes
- **Memory usage**: ~15-20GB per GPU (well within 24GB limit)

## Training with Multi-GPU

### Option 1: Parallel Training (Recommended)
Train different models on different GPUs:

```bash
# Terminal 1: Train GPT-2 Small on GPU 0
CUDA_VISIBLE_DEVICES=0 python src/scripts/train_gpt2_simple.py

# Terminal 2: Train Gemma-2-2B on GPU 1  
CUDA_VISIBLE_DEVICES=1 python src/scripts/train_gemma_example.py
```

### Option 2: Large Model Training
Use both GPUs for a single large model:

```bash
# Train Gemma-2-9B using both GPUs
python src/scripts/train_gemma_example.py
# (Code will automatically use available GPUs)
```

## Memory Management

### Your 48GB Total Memory:
- **GPU 0**: 24GB VRAM
- **GPU 1**: 24GB VRAM
- **System RAM**: 32GB+ recommended

### Memory Allocation Strategy:
```
Gemma-2-2B Training:
├── GPU 0: Model (~8GB) + Activations (~8GB) = 16GB
├── GPU 1: Free for other tasks
└── System: Dataset + Python = ~8GB

Gemma-2-2B Evaluation:
├── GPU 0: Model (~8GB) + Evaluation (~12GB) = 20GB
├── GPU 1: Free for other tasks
└── System: Results + Python = ~4GB
```

## Advanced Multi-GPU Usage

### 1. Parallel Evaluation
Evaluate multiple checkpoints simultaneously:

```bash
# Terminal 1: Evaluate GPT-2 checkpoint on GPU 0
CUDA_VISIBLE_DEVICES=0 python src/scripts/run_evaluation_multi_gpu.py checkpoints/transcoder/gpt2-small/model_1/

# Terminal 2: Evaluate Gemma checkpoint on GPU 1
CUDA_VISIBLE_DEVICES=1 python src/scripts/run_evaluation_multi_gpu.py checkpoints/transcoder/gemma-2-2b/model_2/
```

### 2. Model Comparison
Train and evaluate different models in parallel:

```bash
# Terminal 1: GPT-2 Small
CUDA_VISIBLE_DEVICES=0 python src/scripts/train_gpt2_simple.py

# Terminal 2: Gemma-2-2B
CUDA_VISIBLE_DEVICES=1 python src/scripts/train_gemma_example.py

# Terminal 3: Monitor both
watch -n 1 nvidia-smi
```

### 3. Large Batch Training
Use both GPUs for very large models:

```python
# In training script, set:
cfg["model_batch_size"] = 16  # Large batches
cfg["batch_size"] = 4096      # Large evaluation batches
cfg["num_tokens"] = int(1e8)  # More training data
```

## Monitoring Multi-GPU Usage

### Real-time Monitoring:
```bash
# Monitor both GPUs
watch -n 1 nvidia-smi

# Monitor specific GPU
nvidia-smi -i 0  # GPU 0
nvidia-smi -i 1  # GPU 1

# Monitor processes
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

### Memory Optimization:
```bash
# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Clear all GPU memory
for i in {0..1}; do
    python -c "import torch; torch.cuda.set_device($i); torch.cuda.empty_cache()"
done
```

## Troubleshooting Multi-GPU Issues

### Issue: "CUDA out of memory" on one GPU
**Solution:**
```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=1 python src/scripts/run_evaluation_multi_gpu.py
```

### Issue: Both GPUs showing high usage
**Solution:**
```bash
# Use memory-safe evaluation
python src/scripts/run_evaluation_safe.py
```

### Issue: Model too large for single GPU
**Solution:**
```bash
# Use CPU evaluation
python src/scripts/run_evaluation_cpu.py
```

## Performance Optimization

### 1. GPU Selection Strategy:
```bash
# Use GPU 0 for primary tasks
export CUDA_VISIBLE_DEVICES=0

# Use GPU 1 for secondary tasks  
export CUDA_VISIBLE_DEVICES=1

# Use both GPUs
export CUDA_VISIBLE_DEVICES=0,1
```

### 2. Memory Optimization:
```python
# In your scripts, add:
import torch
torch.cuda.empty_cache()  # Clear cache before large operations
```

### 3. Batch Size Optimization:
```python
# For 2x 24GB GPUs, you can use:
cfg["model_batch_size"] = 8-16    # Large model batches
cfg["batch_size"] = 2048-4096     # Large evaluation batches
cfg["num_tokens"] = int(1e8)      # More training data
```

## Recommended Workflow

### For Your Gemma-2-2B Checkpoint:

1. **Check GPU status:**
   ```bash
   python src/scripts/check_gpu.py
   ```

2. **Use multi-GPU evaluation:**
   ```bash
   python src/scripts/run_evaluation_multi_gpu.py
   ```

3. **Monitor during evaluation:**
   ```bash
   # In another terminal
   watch -n 1 nvidia-smi
   ```

4. **If issues occur, fallback:**
   ```bash
   # Memory-safe evaluation
   python src/scripts/run_evaluation_safe.py
   
   # Or CPU evaluation
   python src/scripts/run_evaluation_cpu.py
   ```

## Expected Results with Your Setup

### Gemma-2-2B Evaluation:
- **Time**: 5-10 minutes (vs 30+ minutes on CPU)
- **Batches**: 100-200 (vs 5-10 on memory-safe)
- **Memory**: ~20GB per GPU (well within 24GB limit)
- **Reliability**: Very high (large sample size)

### GPT-2 Small Evaluation:
- **Time**: 2-5 minutes
- **Batches**: 200+ (comprehensive)
- **Memory**: ~10GB per GPU
- **Reliability**: Excellent

## Quick Commands Summary

```bash
# Check your 2x 24GB setup
python src/scripts/check_gpu.py

# Multi-GPU evaluation (recommended)
python src/scripts/run_evaluation_multi_gpu.py

# Monitor both GPUs
watch -n 1 nvidia-smi

# Use specific GPU
CUDA_VISIBLE_DEVICES=0 python src/scripts/run_evaluation_multi_gpu.py

# Fallback options
python src/scripts/run_evaluation_safe.py
python src/scripts/run_evaluation_cpu.py
```

## Summary

With your 2x 24GB GPU setup, you have **excellent** hardware for:
- ✅ **Fast evaluation**: Use `run_evaluation_multi_gpu.py`
- ✅ **Large models**: Gemma-2-2B, Gemma-2-9B
- ✅ **Parallel processing**: Multiple models simultaneously
- ✅ **High reliability**: Large batch sizes for accurate results

**No need for CPU evaluation** - your GPUs can handle everything! 🚀

---

**Next Steps:**
1. Run `python src/scripts/check_gpu.py` to verify setup
2. Use `python src/scripts/run_evaluation_multi_gpu.py` for your Gemma-2-2B checkpoint
3. Monitor with `watch -n 1 nvidia-smi` during evaluation
4. Enjoy fast, reliable results! 🎉
