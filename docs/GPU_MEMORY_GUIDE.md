# GPU Memory Guide: Handling Out-of-Memory Issues

This guide helps you diagnose and fix GPU memory issues during evaluation, especially with large models like Gemma-2-2B.

## Quick Diagnosis

### 1. Check GPU Status

```bash
# Run diagnostic script
python src/scripts/check_gpu.py

# Or manually check
nvidia-smi
```

### 2. Check Available Memory

```bash
# Check GPU memory
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv

# Check system memory
free -h
```

## Memory Requirements by Model

| Model | GPU Memory Needed | CPU Memory Needed | Safe Batch Size |
|-------|------------------|-------------------|-----------------|
| GPT-2 Small | ~2GB | ~4GB | 50 batches |
| GPT-2 Medium | ~4GB | ~8GB | 30 batches |
| Gemma-2-2B | ~8GB | ~16GB | 10 batches |
| Gemma-2-9B | ~16GB | ~32GB | 5 batches |
| Gemma-2-27B | ~32GB | ~64GB | 3 batches |

## Solutions by Memory Level

### ❌ Very Low Memory (< 2GB GPU)

**Symptoms:**
- `torch.cuda.OutOfMemoryError`
- GPU memory exhausted
- System becomes unresponsive

**Solutions:**
1. **Use CPU evaluation** (recommended):
   ```bash
   python src/scripts/run_evaluation_cpu.py
   ```

2. **Kill GPU processes**:
   ```bash
   # Find GPU processes
   nvidia-smi
   
   # Kill specific process
   kill -9 <PID>
   
   # Or kill all Python processes using GPU
   pkill -f python
   ```

3. **Restart to clear memory**:
   ```bash
   sudo reboot
   ```

4. **Use smaller model**:
   ```bash
   # Train/evaluate GPT-2 Small instead
   python src/scripts/train_gpt2_simple.py
   ```

### ⚠️ Limited Memory (2-4GB GPU)

**Symptoms:**
- Evaluation starts but crashes during processing
- Memory warnings
- Slow performance

**Solutions:**
1. **Use memory-safe evaluation**:
   ```bash
   python src/scripts/run_evaluation_safe.py
   ```

2. **Use CPU evaluation**:
   ```bash
   python src/scripts/run_evaluation_cpu.py
   ```

3. **Reduce batch sizes** (edit script):
   ```python
   # In run_evaluation_safe.py, reduce:
   safe_batches = 5  # Instead of 10
   cfg["model_batch_size"] = 2  # Instead of 4
   cfg["batch_size"] = 128  # Instead of 256
   ```

### ⚠️ Moderate Memory (4-8GB GPU)

**Symptoms:**
- Works but may be slow
- Occasional memory warnings
- May crash with large models

**Solutions:**
1. **Use memory-safe evaluation**:
   ```bash
   python src/scripts/run_evaluation_safe.py
   ```

2. **Monitor memory usage**:
   ```bash
   # In another terminal
   watch -n 1 nvidia-smi
   ```

3. **Use smaller evaluation batches**:
   ```bash
   # Edit run_evaluation_safe.py
   safe_batches = 20  # Instead of 50
   ```

### ✅ Good Memory (> 8GB GPU)

**Solutions:**
1. **Standard evaluation**:
   ```bash
   python src/scripts/run_evaluation.py
   ```

2. **Memory-safe evaluation** (if you want to be cautious):
   ```bash
   python src/scripts/run_evaluation_safe.py
   ```

## Evaluation Scripts Comparison

| Script | Memory Usage | Speed | Reliability | Use When |
|--------|-------------|-------|-------------|----------|
| `run_evaluation.py` | High | Fast | Standard | > 8GB GPU |
| `run_evaluation_safe.py` | Medium | Medium | High | 4-8GB GPU |
| `run_evaluation_cpu.py` | Low | Slow | Very High | < 4GB GPU |

## Step-by-Step Troubleshooting

### Step 1: Diagnose the Problem

```bash
# Check GPU status
python src/scripts/check_gpu.py

# Check what's using GPU memory
nvidia-smi

# Check system memory
free -h
```

### Step 2: Choose Appropriate Solution

**If GPU memory < 2GB:**
```bash
# Use CPU evaluation
python src/scripts/run_evaluation_cpu.py
```

**If GPU memory 2-8GB:**
```bash
# Use memory-safe evaluation
python src/scripts/run_evaluation_safe.py
```

**If GPU memory > 8GB:**
```bash
# Use standard evaluation
python src/scripts/run_evaluation.py
```

### Step 3: Monitor During Evaluation

```bash
# In another terminal, monitor GPU
watch -n 1 nvidia-smi

# Or monitor system memory
watch -n 1 free -h
```

### Step 4: If Still Failing

1. **Reduce batch sizes further**:
   ```python
   # Edit the evaluation script
   safe_batches = 3  # Very small
   cfg["model_batch_size"] = 1  # Minimal
   cfg["batch_size"] = 64  # Small
   ```

2. **Use even smaller model**:
   ```bash
   # Train GPT-2 Small instead of Gemma-2-2B
   python src/scripts/train_gpt2_simple.py
   ```

3. **Use CPU with minimal batches**:
   ```bash
   python src/scripts/run_evaluation_cpu.py
   ```

## Prevention Strategies

### 1. Monitor Memory During Training

```bash
# Monitor GPU during training
watch -n 1 nvidia-smi
```

### 2. Use Appropriate Model Sizes

| Your GPU Memory | Recommended Models |
|-----------------|-------------------|
| < 4GB | GPT-2 Small only |
| 4-8GB | GPT-2 Small, GPT-2 Medium |
| 8-16GB | GPT-2 Small/Medium, Gemma-2-2B |
| > 16GB | Any model |

### 3. Configure Training for Memory

```python
# In training scripts, use smaller batches
cfg["model_batch_size"] = 4  # Instead of 8
cfg["batch_size"] = 512  # Instead of 1024
cfg["seq_len"] = 64  # Instead of 128
```

### 4. Use Memory-Efficient Settings

```python
# For large models
cfg["model_dtype"] = torch.float16  # Half precision
cfg["dtype"] = torch.float16

# For very large models
cfg["model_dtype"] = torch.bfloat16  # Even more memory efficient
```

## Emergency Recovery

### If GPU is Completely Unresponsive

1. **Kill all GPU processes**:
   ```bash
   sudo fuser -v /dev/nvidia*
   sudo kill -9 <PIDs>
   ```

2. **Reset GPU**:
   ```bash
   sudo nvidia-smi --gpu-reset
   ```

3. **Restart system**:
   ```bash
   sudo reboot
   ```

### If System is Frozen

1. **Hard reset** (hold power button)
2. **Check for hardware issues**
3. **Update GPU drivers**

## Best Practices

### 1. Always Check Memory First

```bash
# Before running evaluation
python src/scripts/check_gpu.py
```

### 2. Start with Safe Options

```bash
# Start with memory-safe evaluation
python src/scripts/run_evaluation_safe.py

# If that works, try standard
python src/scripts/run_evaluation.py
```

### 3. Monitor During Execution

```bash
# Keep nvidia-smi open during evaluation
watch -n 1 nvidia-smi
```

### 4. Use Appropriate Models

- **For learning/testing**: GPT-2 Small
- **For research**: Gemma-2-2B (if you have 8GB+ GPU)
- **For production**: Match model size to your hardware

## Common Error Messages

### `torch.cuda.OutOfMemoryError`

**Cause:** GPU memory exhausted
**Solution:** Use CPU evaluation or smaller model

### `CUDA out of memory`

**Cause:** Not enough GPU memory
**Solution:** Use `run_evaluation_safe.py` or `run_evaluation_cpu.py`

### `RuntimeError: CUDA error: out of memory`

**Cause:** GPU memory allocation failed
**Solution:** Restart and use memory-safe evaluation

### `Killed` (process killed)

**Cause:** System out of memory (OOM killer)
**Solution:** Use CPU evaluation or reduce batch sizes

## Hardware Recommendations

### Minimum Requirements

- **GPU**: 4GB VRAM (for GPT-2 Small)
- **RAM**: 8GB system memory
- **CPU**: 4 cores

### Recommended for Gemma-2-2B

- **GPU**: 8GB+ VRAM (RTX 3070, RTX 4060 Ti, etc.)
- **RAM**: 16GB+ system memory
- **CPU**: 8+ cores

### Recommended for Gemma-2-9B

- **GPU**: 16GB+ VRAM (RTX 4080, RTX 4090, A100, etc.)
- **RAM**: 32GB+ system memory
- **CPU**: 16+ cores

## Summary

| Problem | Solution |
|---------|----------|
| GPU memory < 2GB | Use `run_evaluation_cpu.py` |
| GPU memory 2-8GB | Use `run_evaluation_safe.py` |
| GPU memory > 8GB | Use `run_evaluation.py` |
| Still failing | Reduce batch sizes or use smaller model |
| System frozen | Restart and use CPU evaluation |

## Quick Commands

```bash
# Check GPU status
python src/scripts/check_gpu.py

# Memory-safe evaluation
python src/scripts/run_evaluation_safe.py

# CPU evaluation (always works)
python src/scripts/run_evaluation_cpu.py

# Monitor GPU memory
watch -n 1 nvidia-smi
```

Remember: **CPU evaluation is always safe** - it's slower but won't burn out your GPU! 🔥➡️❄️
