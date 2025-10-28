# SAEBench Interpretability Evaluation Guide

This guide explains how to evaluate and compare transcoders using SAEBench-standard interpretability metrics.

## Overview

The evaluation script (`eval_interpretability_saebench.py`) implements core interpretability metrics from SAEBench to evaluate and compare sparse autoencoders and transcoders.

## Core Metrics

### 1. Absorption Score (Primary SAEBench Metric)

**What it measures:** Feature redundancy via pairwise cosine similarity between feature vectors (decoder weights).

**How it works:**
- Computes cosine similarity between all pairs of feature vectors
- Counts pairs with similarity > 0.9 (SAEBench standard threshold)
- Returns fraction of pairs that are "absorbed" (highly similar)

**Interpretation:**
- `< 0.01`: ✅ **EXCELLENT** - Features are diverse and interpretable
- `0.01 - 0.05`: ⚠️ **MODERATE** - Some redundancy, but acceptable
- `≥ 0.05`: ❌ **POOR** - High feature redundancy

**Lower is better** - Lower absorption means features are more diverse and interpretable.

**Why it matters:** 
- Redundant features waste dictionary capacity
- Diverse features = better interpretability
- Core metric for evaluating SAE quality

### 2. Reconstruction Quality (FVU)

**What it measures:** How well the transcoder reconstructs target activations.

**FVU (Fraction of Variance Unexplained):**
```
FVU = Var(residuals) / Var(original)
    = MSE(original, reconstruction) / Var(original)
```

**Interpretation:**
- `< 0.1`: ✅ **EXCELLENT** - Nearly perfect reconstruction
- `0.1 - 0.3`: ⚠️ **MODERATE** - Acceptable reconstruction
- `≥ 0.3`: ❌ **POOR** - Significant information loss

**Lower is better** - 0 = perfect reconstruction, 1 = no better than mean.

### 3. Sparsity Metrics

**What it measures:** Distribution of active features per sample (L0 norm).

**Metrics reported:**
- Mean L0: Average number of active features
- Median L0: Median number of active features  
- Std L0: Standard deviation (consistency of sparsity)

**Why it matters:**
- Sparse representations are more interpretable
- Consistency (low std) indicates stable feature activation

### 4. Feature Utilization

**What it measures:** Proportion of features that are actively used.

**Dead features:** Features that haven't activated in N consecutive batches (default: 20).

**Interpretation:**
- `< 10% dead`: ✅ **EXCELLENT** utilization
- `10-30% dead`: ⚠️ **MODERATE** utilization
- `> 30% dead`: ❌ **POOR** utilization

## Usage

### Basic Usage

```bash
python src/eval/eval_interpretability_saebench.py \
    --ours_checkpoint checkpoints/transcoder/gemma-2-2b/my_model_15000 \
    --google_dir /path/to/gemma_scope_2b \
    --layer 17 \
    --batches 1000 \
    --output_dir analysis_results/saebench_comparison
```

### Required Arguments

- `--ours_checkpoint`: Path to your Matryoshka transcoder checkpoint directory
- `--google_dir`: Path to Google's Gemma Scope transcoder directory
- `--layer`: Layer number to evaluate (e.g., 17)

### Optional Arguments

- `--dataset`: Dataset for evaluation (default: `HuggingFaceFW/fineweb-edu`)
- `--batches`: Number of batches to evaluate (default: 1000)
  - More batches = more accurate but slower
  - 1000 batches ≈ 5-10 minutes on GPU
- `--batch_size`: Batch size for evaluation (default: 256)
- `--seq_len`: Sequence length (default: 64)
- `--device`: Device to use (default: `cuda:0`)
- `--dtype`: Data type (default: `bfloat16`)
- `--output_dir`: Output directory (default: `analysis_results/saebench_comparison`)

## Example Output

```
================================================================================
📊 SAEBench INTERPRETABILITY COMPARISON
================================================================================

🎯 RECONSTRUCTION QUALITY
--------------------------------------------------------------------------------
Metric                  Ours          Google        Winner
--------------------------------------------------------------------------------
FVU (lower=better)      0.045123    0.052341    ✨ Ours
MSE                     0.003421    0.003987    ✨ Ours
Cosine Sim (higher=better) 0.987654    0.983210    ✨ Ours

📊 ABSORPTION SCORE (SAEBench Core Metric)
--------------------------------------------------------------------------------
Absorption measures feature redundancy (lower = better interpretability)
Standard threshold: 0.9

Ours:   0.008234
        (234 pairs above threshold)
        ✅ EXCELLENT - Features are diverse and interpretable

Google: N/A (direct mapping, no sparse features)
        Note: Google's transcoder is not a sparse autoencoder,
              so absorption score is not applicable.

✨ SPARSITY (Matryoshka Transcoder Only)
--------------------------------------------------------------------------------
Mean L0:    96.4
Median L0:  95.0
Std L0:     12.3
Dead features: 142 / 18432 (0.8%)
✅ EXCELLENT feature utilization

================================================================================
📝 INTERPRETATION GUIDE
================================================================================
Absorption Score (SAEBench core metric):
  < 0.01: Excellent - Features are diverse and interpretable
  < 0.05: Moderate - Some redundancy, but acceptable
  ≥ 0.05: Poor - High feature redundancy

FVU (Reconstruction quality):
  < 0.1:  Excellent reconstruction
  < 0.3:  Moderate reconstruction
  ≥ 0.3:  Poor reconstruction
================================================================================
```

## Output Files

The script saves results to `{output_dir}/saebench_comparison.json`:

```json
{
  "saebench_metrics": {
    "absorption_score": {
      "description": "Measures feature redundancy via pairwise cosine similarity. Lower is better.",
      "threshold": 0.9,
      "interpretation": "< 0.01 = excellent, < 0.05 = moderate, >= 0.05 = poor"
    },
    "reconstruction_quality": {
      "description": "Measures how well the transcoder reconstructs target activations.",
      "fvu_interpretation": "< 0.1 = excellent, < 0.3 = moderate, >= 0.3 = poor"
    }
  },
  "matryoshka_transcoder": {
    "absorption_score": 0.008234,
    "absorption_pairs_above_threshold": 234,
    "fvu": 0.045123,
    "mse": 0.003421,
    "cosine_similarity": 0.987654,
    "mean_l0": 96.4,
    "median_l0": 95.0,
    "dead_features_percentage": 0.8
  },
  "google_transcoder": {
    "absorption_score": null,
    "fvu": 0.052341,
    "mse": 0.003987,
    "cosine_similarity": 0.983210
  },
  "comparison": {
    "fvu_difference": -0.007218,
    "fvu_winner": "matryoshka",
    "note": "Google's transcoder is a direct mapping without sparse features, so absorption score is not applicable."
  }
}
```

## Comparison: Matryoshka vs Google

### What's Different?

**Matryoshka Transcoder (Ours):**
- **Sparse autoencoder** with learned features
- Has interpretable sparse features
- Absorption score applies (measures feature diversity)
- Supports nested evaluation (Matryoshka structure)
- More parameters but sparse activation

**Google Gemma Scope Transcoder:**
- **Direct linear mapping** (no sparse features)
- No feature learning, just MLP approximation
- Absorption score N/A (no features to compare)
- Simpler, fewer parameters
- Dense activation

### Which Metrics Apply?

| Metric | Matryoshka | Google | Notes |
|--------|------------|--------|-------|
| Absorption Score | ✅ | ❌ | Only applies to sparse autoencoders |
| FVU | ✅ | ✅ | Both reconstruct activations |
| MSE/MAE | ✅ | ✅ | Both reconstruct activations |
| Cosine Similarity | ✅ | ✅ | Both reconstruct activations |
| Sparsity (L0) | ✅ | ❌ | Only sparse models have sparsity |
| Feature Utilization | ✅ | ❌ | Only sparse models have features |

### Key Comparison Points

1. **Interpretability:** Matryoshka has interpretable features (measured by absorption), Google doesn't
2. **Reconstruction:** Both can be compared on FVU, MSE, cosine similarity
3. **Efficiency:** Google may be faster (dense), Matryoshka more sparse

## Troubleshooting

### "No such file or directory"

Make sure your checkpoint path is correct:
```bash
ls checkpoints/transcoder/gemma-2-2b/my_model_15000/
# Should show: sae.pt (or checkpoint.pt) and config.json
```

### "CUDA out of memory"

Reduce batch size:
```bash
--batch_size 128  # Instead of default 256
```

Or use CPU:
```bash
--device cpu
```

### "Model dtype mismatch"

Ensure dtype matches your training:
```bash
--dtype bfloat16  # If you trained with bfloat16
```

## References

- **SAEBench Paper:** https://arxiv.org/abs/2503.09532
- **SAEBench GitHub:** https://github.com/adamkarvonen/SAEBench  
- **Absorption Score:** Measures feature redundancy (SAEBench standard metric)
- **Matryoshka SAE Paper:** Nested representations with flexible capacity

## Tips for Good Results

1. **Use enough batches:** 1000+ batches for stable estimates
2. **Match training conditions:** Use same dtype, similar batch size
3. **Check absorption score:** < 0.01 indicates excellent feature diversity
4. **Compare FVU:** Lower is better, < 0.1 is excellent
5. **Monitor dead features:** < 10% dead is ideal

