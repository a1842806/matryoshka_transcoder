# Feature Evaluation Guide: Measuring Absorption & Splitting

## Overview

This guide explains how to evaluate feature quality in your Matryoshka Transcoder, focusing on measuring **feature absorption** and **feature splitting** - two key indicators of model quality.

## Key Metrics

### 1. Feature Absorption (Cosine Similarity)

**What it is:** Multiple features learning similar representations

**How to measure:** Cosine similarity between feature weight vectors

**Interpretation:**
- **Low absorption** (< 0.05): Features are diverse ✅
- **Moderate absorption** (0.05-0.15): Some redundancy ⚠️
- **High absorption** (> 0.15): Many redundant features ❌

**Why it's bad:** Wastes dictionary capacity on duplicate features

### 2. Feature Splitting (Co-activation)

**What it is:** Multiple features activating together frequently

**How to measure:** Co-activation frequency across inputs

**Interpretation:**
- **Low splitting** (< 0.05): Features are independent ✅
- **Moderate splitting** (0.05-0.15): Some co-activation ⚠️
- **High splitting** (> 0.15): Features should be merged ❌

**Why it's bad:** Indicates features could be combined into one

### 3. Monosemanticity

**What it is:** How specific/interpretable each feature is

**How to measure:** Sparsity + activation consistency

**Interpretation:**
- **High monosemanticity** (> 0.7): Specific, interpretable ✅
- **Moderate monosemanticity** (0.4-0.7): Mixed ⚠️
- **Low monosemanticity** (< 0.4): Polysemantic, noisy ❌

**Why it matters:** Monosemantic features are more interpretable

### 4. Hierarchical Structure (Matryoshka-specific)

**What it is:** Feature organization across groups

**How to measure:** Within-group vs between-group similarity

**Good Matryoshka model:**
- Low within-group similarity (diverse features in each group)
- Clear between-group separation (groups learn different abstraction levels)

### 5. Dead Features

**What it is:** Features that never activate

**How to measure:** Count features with zero activation

**Interpretation:**
- **Low dead** (< 5%): Efficient dictionary use ✅
- **Moderate dead** (5-15%): Some waste ⚠️
- **High dead** (> 15%): Poor dictionary utilization ❌

---

## When to Evaluate

### Option 1: After Training (Recommended)

**Advantages:**
- ✅ Complete picture of learned features
- ✅ Stable representations
- ✅ More reliable metrics
- ✅ Comprehensive analysis

**When to use:**
- Final model evaluation
- Model comparison
- Detailed interpretability analysis

**How:**
```python
from evaluate_features import evaluate_trained_transcoder

results = evaluate_trained_transcoder(
    checkpoint_path="checkpoints/your_model_4878",
    activation_store=activation_store,
    model=model,
    cfg=cfg,
    num_batches=100  # More batches = more reliable
)
```

### Option 2: During Training

**Advantages:**
- ✅ Real-time monitoring
- ✅ Early problem detection
- ✅ Can adjust hyperparameters mid-training

**Disadvantages:**
- ⚠️ Slower training
- ⚠️ Features still evolving
- ⚠️ Less reliable metrics

**When to use:**
- Monitoring training health
- Debugging issues
- Experimental runs

**How:**
```python
# In training loop
if step % 5000 == 0:  # Every 5000 steps
    evaluator = FeatureEvaluator(transcoder, activation_store, model, cfg)
    evaluator.collect_activations(num_batches=20)  # Fewer batches for speed
    absorption = evaluator.compute_feature_absorption()
    splitting = evaluator.compute_feature_splitting()
    
    # Log to W&B
    wandb.log({
        "eval/absorption_score": absorption["absorption_score"],
        "eval/splitting_score": splitting["splitting_score"]
    })
```

---

## Quick Start

### Basic Evaluation (After Training)

```python
from evaluate_features import FeatureEvaluator, evaluate_trained_transcoder
from transformer_lens import HookedTransformer
from transcoder_activation_store import TranscoderActivationsStore

# Load model
model = HookedTransformer.from_pretrained("gpt2-small").to("cuda")

# Create activation store
activation_store = TranscoderActivationsStore(model, cfg)

# Evaluate
results = evaluate_trained_transcoder(
    checkpoint_path="checkpoints/gpt2-small_blocks.8.hook_resid_mid_to_blocks.8.hook_mlp_out_12288_matryoshka-transcoder_64_0.0003_4878",
    activation_store=activation_store,
    model=model,
    cfg=cfg,
    num_batches=100
)

# Results saved to: checkpoints/.../evaluation/results.json
```

### Individual Metrics

```python
from evaluate_features import FeatureEvaluator

evaluator = FeatureEvaluator(transcoder, activation_store, model, cfg)

# Collect activations first
evaluator.collect_activations(num_batches=100)

# Run individual analyses
absorption = evaluator.compute_feature_absorption()
splitting = evaluator.compute_feature_splitting()
monosemanticity = evaluator.compute_monosemanticity()
dead_features = evaluator.analyze_dead_features()

# Matryoshka-specific
if hasattr(transcoder, 'group_indices'):
    hierarchical = evaluator.analyze_hierarchical_structure()
```

---

## Interpreting Results

### Example: Good Matryoshka Transcoder

```
EVALUATION SUMMARY
====================================================================
Absorption Score: 0.0234 (lower is better)    ✅ Low absorption
Splitting Score: 0.0456 (lower is better)     ✅ Low splitting
Mean Monosemanticity: 0.7123 (higher is better)  ✅ High monosemanticity
Dead Features: 3.2% (lower is better)          ✅ Few dead features
====================================================================

Hierarchical Analysis:
Group 0: within=0.15, between=0.25  ✅ Clear separation
Group 1: within=0.18, between=0.31  ✅ Clear separation
Group 2: within=0.22, between=0.38  ✅ Clear separation
```

### Example: Problems to Fix

```
EVALUATION SUMMARY
====================================================================
Absorption Score: 0.1823 (lower is better)    ❌ HIGH absorption
Splitting Score: 0.1256 (lower is better)     ❌ HIGH splitting
Mean Monosemanticity: 0.3421 (higher is better)  ❌ LOW monosemanticity
Dead Features: 18.5% (lower is better)         ❌ MANY dead features
====================================================================

Top Similar Pairs (Absorption):
  Feature  123 (Group 1) ↔ Feature  456 (Group 1): 0.94  ❌ Nearly identical
  Feature  789 (Group 2) ↔ Feature  1011 (Group 2): 0.91  ❌ Nearly identical

Top Co-activating Pairs (Splitting):
  Feature  567 (Group 1) ↔ Feature  890 (Group 1): 0.78  ❌ Always activate together
```

**Fixes:**
1. **High absorption → Increase dictionary size or top_k**
2. **High splitting → Decrease top_k or increase sparsity penalty**
3. **Low monosemanticity → Train longer or adjust architecture**
4. **Many dead features → Increase auxiliary loss weight**

---

## Comparison to SAE Bench

### SAE Bench Metrics

SAE Bench provides:
- Automated interpretability scores
- Downstream task performance
- Sparsity-performance tradeoffs

### Our Metrics

Our evaluation focuses on:
- **Feature quality** (absorption, splitting)
- **Hierarchical structure** (Matryoshka-specific)
- **Efficient dictionary use** (dead features, monosemanticity)

### When to Use Each

**Use SAE Bench when:**
- You want automated interpretability scores
- You care about downstream task performance
- You want to compare to baseline SAEs

**Use our evaluation when:**
- You want to understand feature quality
- You're training Matryoshka models (hierarchical analysis)
- You want to diagnose training issues
- You want quick, interpretable metrics

**Best practice:** Use both! They complement each other.

---

## Best Practices

### 1. Evaluation Frequency

**During training:**
- Every 5000-10000 steps
- Use 20-50 batches for speed

**After training:**
- Once at the end
- Use 100-200 batches for reliability

### 2. Number of Batches

- **20 batches**: Quick check (~2-3 minutes)
- **100 batches**: Standard evaluation (~10-15 minutes)
- **200+ batches**: Thorough analysis (~20-30 minutes)

### 3. What to Monitor During Training

**Critical metrics:**
1. Dead features (should stay < 10%)
2. Absorption score (should stay < 0.1)
3. Splitting score (should stay < 0.1)

**If problems arise:**
- High dead features → Increase `aux_penalty`
- High absorption → Increase `dict_size` or `top_k`
- High splitting → Decrease `top_k`

### 4. Matryoshka-Specific Tips

**Good hierarchical structure:**
- Within-group similarity increases with group index
- Between-group similarity shows clear jumps
- Early groups (coarse) have lower similarity
- Later groups (fine) have higher similarity

**If hierarchy is poor:**
- Groups learning similar things → Adjust `group_sizes`
- No clear separation → Train longer or adjust loss weights

---

## Advanced Usage

### Compare Multiple Checkpoints

```python
checkpoints = [
    "checkpoints/model_1000",
    "checkpoints/model_2000",
    "checkpoints/model_3000",
    "checkpoints/model_4000"
]

for ckpt in checkpoints:
    print(f"\nEvaluating {ckpt}...")
    results = evaluate_trained_transcoder(ckpt, activation_store, model, cfg)
    print(f"Absorption: {results['absorption']['absorption_score']:.4f}")
    print(f"Splitting: {results['splitting']['splitting_score']:.4f}")
```

### Visualize Feature Similarity

```python
import matplotlib.pyplot as plt
import seaborn as sns

absorption = evaluator.compute_feature_absorption()
similarity_matrix = absorption["similarity_matrix"]

plt.figure(figsize=(12, 10))
sns.heatmap(similarity_matrix, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title("Feature Similarity Matrix")
plt.savefig("similarity_heatmap.png")
```

### Export for SAE Bench

```python
# Export feature activations for SAE Bench
activations = evaluator.feature_acts_history
torch.save({
    "activations": activations,
    "feature_weights": evaluator.feature_weights,
    "config": cfg
}, "sae_bench_export.pt")
```

---

## Troubleshooting

### "No activations collected"
```python
# Run this first
evaluator.collect_activations(num_batches=100)
```

### "Out of memory during evaluation"
```python
# Use fewer batches
evaluator.collect_activations(num_batches=20)
```

### "Evaluation takes too long"
```python
# Reduce batch count or use GPU
cfg["device"] = "cuda"
evaluator = FeatureEvaluator(transcoder, activation_store, model, cfg)
```

---

## Summary

### Quick Reference

| Metric | Good | Bad | Fix |
|--------|------|-----|-----|
| Absorption | < 0.05 | > 0.15 | ↑ dict_size, ↑ top_k |
| Splitting | < 0.05 | > 0.15 | ↓ top_k |
| Monosemanticity | > 0.7 | < 0.4 | Train longer |
| Dead Features | < 5% | > 15% | ↑ aux_penalty |

### Recommended Workflow

1. **During training**: Monitor critical metrics every 5000 steps
2. **After training**: Run full evaluation with 100+ batches
3. **Compare**: Evaluate multiple checkpoints to track improvement
4. **Interpret**: Use results to understand feature quality
5. **Iterate**: Adjust hyperparameters based on findings

### Next Steps

- Run evaluation on your trained transcoder
- Compare to baseline SAEs
- Experiment with hyperparameters
- Use SAE Bench for additional metrics
- Analyze specific features for interpretability

Happy evaluating! 🔍



