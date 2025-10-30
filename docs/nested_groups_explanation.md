# Nested Groups in Matryoshka Transcoder

## Overview

The Matryoshka Transcoder implements **nested groups** (like Russian nesting dolls), where each larger group contains all features from smaller groups plus additional new features. This enables adaptive computation at multiple granularities from a single trained model.

## Implementation Details

### Group Structure

With `group_sizes = [1152, 2304, 4608, 10368]`:

```
Group 0: features [0:1152]     → 1,152 features total
Group 1: features [0:3456]     → 3,456 features total (includes Group 0 + 2,304 new)
Group 2: features [0:8064]     → 8,064 features total (includes Groups 0-1 + 4,608 new)
Group 3: features [0:18432]    → 18,432 features total (all features)
```

### Code Implementation

```python
# In MatryoshkaTranscoder.__init__()
total_dict_size = sum(cfg["group_sizes"])  # 18432
self.group_sizes = cfg["group_sizes"]  # [1152, 2304, 4608, 10368]
self.group_indices = [0] + list(torch.cumsum(torch.tensor(cfg["group_sizes"]), dim=0))
# group_indices = [0, 1152, 3456, 8064, 18432]
```

### Nested Reconstruction

During forward pass:

```python
for i in range(self.active_groups):
    # NESTED: Always start from 0, end at cumulative size
    start_idx = 0  # Always 0 for nested groups!
    end_idx = self.group_indices[i+1]  # Cumulative boundary
    
    # Use features [0:end_idx]
    W_dec_slice = self.W_dec[start_idx:end_idx, :]
    acts_topk_slice = all_acts_topk[:, start_idx:end_idx]
    x_reconstruct = acts_topk_slice @ W_dec_slice + self.b_dec
    intermediate_reconstructs.append(x_reconstruct)
```

## Nested vs Sequential

### ✅ NESTED (Current Implementation - CORRECT)

Each group uses all features from index 0 to its boundary:

```
Group 0: [0:1152]      (1,152 features)
Group 1: [0:3456]      (3,456 features) ← includes Group 0
Group 2: [0:8064]      (8,064 features) ← includes Groups 0-1
Group 3: [0:18432]     (18,432 features) ← includes all previous groups
```

**Key property**: Group i ⊃ Group i-1 (containment)

### ❌ SEQUENTIAL (INCORRECT for Matryoshka)

Each group would use only its own range:

```
Group 0: [0:1152]      (1,152 features)
Group 1: [1152:3456]   (2,304 features) ← WRONG! Doesn't include Group 0
Group 2: [3456:8064]   (4,608 features) ← WRONG! Separate features
Group 3: [8064:18432]  (10,368 features) ← WRONG! No nesting
```

**Key problem**: Groups are disjoint, no nesting property

## Why Nested Groups?

### 1. Adaptive Computation
- **Low compute**: Use Group 0 (1,152 features)
- **Medium compute**: Use Group 1 (3,456 features)
- **High compute**: Use Group 3 (18,432 features)
- **Same model, different accuracy/speed tradeoffs**

### 2. Hierarchical Features
- **Group 0**: Coarse, high-level features
- **Group 1**: Adds medium-level features
- **Group 2**: Adds fine-grained features
- **Group 3**: Maximum detail

### 3. One Model, Multiple Capacities
Instead of training 4 separate models at different sizes, train one model that can operate at 4 different capacity levels.

## Mathematical Formulation

For nested group $i$:

$$\text{Reconstruction}_i = \sum_{j=0}^{\text{group\_indices}[i+1]} a_j \mathbf{w}_j + \mathbf{b}$$

Where:
- $a_j$ are the sparse activations
- $\mathbf{w}_j$ are the decoder columns
- Sum always starts at $j=0$ (nested property)

## Verification

Run the verification script to confirm nested structure:

```bash
python src/utils/verify_nested_groups.py
```

Expected output:
```
✅ Group 1 correctly contains all features from Group 0
✅ Group 2 correctly contains all features from Group 1
✅ Group 3 correctly contains all features from Group 2
✅ NESTED GROUPS VERIFIED CORRECTLY!
```

## Training Configurations

### Layer 17 (from `train_gemma_layer17_with_warmup_decay_samples.py`)
```python
cfg["dict_size"] = 18432
cfg["group_sizes"] = [1152, 2304, 4608, 10368]
```

### Layer 12 (from `train_gemma_layer12_with_warmup_decay_samples.py`)
```python
cfg["dict_size"] = 36864
cfg["group_sizes"] = [2304, 4608, 9216, 20736]
```

### Layer 8 (from `train_gemma_layer8_with_warmup_decay_samples.py`)
```python
cfg["dict_size"] = 18432
cfg["group_sizes"] = [2304, 4608, 9216, 2304]
```

## Benefits of Nested Implementation

1. **Efficiency**: Single forward pass computes all group reconstructions
2. **Flexibility**: Can choose reconstruction quality at inference time
3. **Memory efficient**: Only one set of weights, reused across groups
4. **Interpretability**: Hierarchical feature organization
5. **Production friendly**: Deploy one model for multiple use cases

## Common Misconceptions

### ❌ Misconception 1: "Groups are separate feature banks"
**Reality**: Groups are cumulative. Each group builds on previous ones.

### ❌ Misconception 2: "Group i only uses new features added in that group"
**Reality**: Group i uses ALL features from 0 to its boundary.

### ❌ Misconception 3: "Need to train separate models for each group"
**Reality**: One model provides all group capacities simultaneously.

## References

- **Matryoshka SAE Paper**: [Arxiv 2503.17547](https://arxiv.org/abs/2503.17547)
- **Matryoshka Representation Learning**: [Arxiv 2205.13147](https://arxiv.org/abs/2205.13147)
- **Original Matryoshka SAE Code**: Implements nested groups via cumulative sum

## Code References

Key files demonstrating nested implementation:

1. `src/models/sae.py` (lines 91-104): Group initialization
2. `src/models/sae.py` (lines 207-223): Nested reconstruction loop
3. `src/models/sae.py` (lines 380-397): Decode method with nested groups
4. `src/utils/verify_nested_groups.py`: Verification utilities

## Summary

✅ **Your implementation is CORRECT** - it uses nested groups properly!

The key insight: `torch.cumsum()` creates cumulative boundaries, and always starting reconstruction from index 0 ensures each group contains all previous features, implementing the true Matryoshka nesting property.

