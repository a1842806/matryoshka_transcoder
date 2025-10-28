# SAEBench Metrics Quick Reference

## Absorption Score (Primary Metric)

**Formula:** `Fraction of feature pairs with cosine_similarity > 0.9`

| Score | Quality | Interpretation |
|-------|---------|----------------|
| < 0.01 | ✅ Excellent | Features are diverse and interpretable |
| 0.01 - 0.05 | ⚠️ Moderate | Some redundancy, acceptable |
| ≥ 0.05 | ❌ Poor | High feature redundancy |

**Direction:** Lower is better  
**Applies to:** Sparse autoencoders only (not direct mappings)

---

## FVU (Fraction of Variance Unexplained)

**Formula:** `Var(residuals) / Var(original) = MSE / Var(original)`

| FVU | Quality | Interpretation |
|-----|---------|----------------|
| < 0.1 | ✅ Excellent | Nearly perfect reconstruction |
| 0.1 - 0.3 | ⚠️ Moderate | Acceptable reconstruction |
| ≥ 0.3 | ❌ Poor | Significant information loss |

**Direction:** Lower is better (0 = perfect)  
**Applies to:** All transcoders

---

## Cosine Similarity

**Formula:** `cosine(reconstruction, original)`

| Score | Quality |
|-------|---------|
| > 0.95 | ✅ Excellent |
| 0.85 - 0.95 | ⚠️ Moderate |
| < 0.85 | ❌ Poor |

**Direction:** Higher is better (1 = perfect)  
**Applies to:** All transcoders

---

## Sparsity (L0 Norm)

**Measures:** Number of active features per sample

**Good indicators:**
- **Low mean L0:** Sparse representation
- **Low std L0:** Consistent sparsity
- **Mean ≈ Median:** Stable distribution

**Applies to:** Sparse autoencoders only

---

## Dead Features

**Definition:** Features inactive for N consecutive batches (default: 20)

| Percentage | Quality | Interpretation |
|------------|---------|----------------|
| < 10% | ✅ Excellent | High feature utilization |
| 10-30% | ⚠️ Moderate | Some waste |
| > 30% | ❌ Poor | Low feature utilization |

**Direction:** Lower is better  
**Applies to:** Sparse autoencoders only

---

## Matryoshka vs Google: Which Metrics Apply?

| Metric | Matryoshka | Google | Why? |
|--------|------------|--------|------|
| **Absorption Score** | ✅ | ❌ | Google has no sparse features |
| **FVU** | ✅ | ✅ | Both reconstruct activations |
| **Cosine Similarity** | ✅ | ✅ | Both reconstruct activations |
| **Sparsity (L0)** | ✅ | ❌ | Google has no sparsity |
| **Dead Features** | ✅ | ❌ | Google has no features |

**Key Insight:** Google's transcoder is a direct linear mapping, not a sparse autoencoder.  
Only reconstruction metrics (FVU, cosine similarity) can be compared directly.

---

## Command Line Usage

```bash
# Basic evaluation
python src/eval/eval_interpretability_saebench.py \
    --ours_checkpoint checkpoints/transcoder/gemma-2-2b/my_model_15000 \
    --google_dir /path/to/gemma_scope_2b \
    --layer 17 \
    --batches 1000

# Quick evaluation (fewer batches)
--batches 500  # ~2-5 minutes

# Thorough evaluation (more batches)
--batches 2000  # ~10-20 minutes

# Use CPU
--device cpu

# Different dtype
--dtype float32
```

---

## Interpreting Your Results

### 🎯 Goal: Low Absorption + Low FVU

**Best case:**
- Absorption < 0.01 (diverse features)
- FVU < 0.1 (excellent reconstruction)
- Dead features < 10%

**Good case:**
- Absorption < 0.05 (acceptable diversity)
- FVU < 0.3 (moderate reconstruction)
- Dead features < 30%

### 🚩 Red Flags

- **High absorption (≥ 0.05):** Features are redundant → reduce dictionary size or adjust training
- **High FVU (≥ 0.3):** Poor reconstruction → increase capacity or adjust learning rate
- **Many dead features (> 30%):** Wasted capacity → adjust aux penalty or learning rate

---

## SAEBench Paper Reference

**Citation:**
```bibtex
@article{saebench2025,
  title={SAEBench: A Comprehensive Benchmark for Sparse Autoencoders},
  journal={arXiv preprint arXiv:2503.09532},
  year={2025}
}
```

**Key contributions:**
- Standardized absorption score (threshold: 0.9)
- Benchmark datasets and evaluation protocol
- Comparison methodology for sparse autoencoders

**GitHub:** https://github.com/adamkarvonen/SAEBench

