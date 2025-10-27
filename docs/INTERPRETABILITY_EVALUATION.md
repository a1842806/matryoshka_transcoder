# Interpretability Evaluation Guide

This guide explains how to evaluate your Matryoshka Transcoder using trusted interpretability metrics and compare it against Google's Gemma Scope transcoder.

## 📊 Overview

We implement **research-backed, trusted interpretability metrics** based on:

1. **SAE Bench** - Standard benchmark for sparse autoencoders
2. **Anthropic's Research** - Interpretability best practices
3. **Research Consensus** - Metrics validated across multiple papers

## 🎯 Key Metrics Implemented

### 1. **Absorption Score** (Primary Interpretability Metric)
- **What it measures**: Feature redundancy through pairwise cosine similarity
- **Formula**: `Fraction of feature pairs with similarity > 0.9`
- **Interpretation**:
  - `< 0.01`: ✅ Excellent (features are diverse and interpretable)
  - `0.01 - 0.05`: ⚠️ Moderate (some redundancy)
  - `> 0.05`: ❌ High redundancy (features absorb each other)
- **Why it matters**: Lower absorption = more interpretable, diverse features
- **Reference**: SAE Bench standard metric

### 2. **Feature Utilization** (Optimization Quality)
- **What it measures**: Percentage of dead (inactive) features
- **Formula**: `Dead features / Total features × 100%`
- **Interpretation**:
  - `< 10%`: ✅ Excellent utilization
  - `10% - 30%`: ⚠️ Some waste
  - `> 30%`: ❌ Poor utilization
- **Why it matters**: High dead percentage = wasted capacity

### 3. **Reconstruction Quality** (Accuracy Metrics)
- **FVU (Fraction of Variance Unexplained)**:
  - `< 0.1`: ✅ Excellent
  - `0.1 - 0.3`: ⚠️ Moderate  
  - `> 0.3`: ❌ Poor
- **Your current FVU: 0.34** - This is in the moderate range, room for improvement

### 4. **Sparsity Analysis**
- **L0 norm**: Average number of active features per sample
- **Distribution**: Mean, median, std of L0 across samples

## 🚀 Quick Start

### Option 1: Compare with Google's Transcoder

```bash
python src/eval/compare_interpretability.py \
    --ours_checkpoint checkpoints/transcoder/gemma-2-2b/your_model_15000 \
    --google_dir /path/to/gemma_scope_2b_transcoders \
    --layer 17 \
    --batches 1000 \
    --device cuda:0 \
    --output_dir analysis_results/interpretability_comparison
```

**Parameters:**
- `--ours_checkpoint`: Path to your trained transcoder checkpoint
- `--google_dir`: Path to downloaded Google Gemma Scope transcoders
- `--layer`: Layer number (default: 17)
- `--batches`: Number of evaluation batches (more = more accurate)
- `--device`: GPU device to use
- `--output_dir`: Where to save results

### Option 2: Evaluate Only Your Model

```bash
python src/eval/evaluate_interpretability_standalone.py \
    --checkpoint checkpoints/transcoder/gemma-2-2b/your_model_15000 \
    --layer 17 \
    --batches 1000 \
    --device cuda:0
```

## 📈 Understanding Your Results

### Example Output

```
📊 INTERPRETABILITY METRICS - Our Matryoshka Transcoder
================================================================================

📊 ABSORPTION SCORE (Lower is Better)
  Score: 0.0123
  Pairs above 0.9: 15,234
  Interpretation: ⚠️  High redundancy

🔧 FEATURE UTILIZATION
  Dead features: 1,842 / 18,432 (10.0%)
  Alive features: 16,590 (90.0%)
  Interpretation: ✅ Good utilization

✨ SPARSITY
  Mean L0: 96.3
  Median L0: 96.0
  Std L0: 12.4

🎯 RECONSTRUCTION QUALITY
  FVU: 0.3400 (lower is better)
  MSE: 0.001234
  MAE: 0.023456
  Cosine similarity: 0.9876
  Interpretation: ⚠️  Moderate reconstruction

🔥 ACTIVATION STATISTICS
  Max activation: 12.456
  Mean activation (non-zero): 2.345
================================================================================
```

### Interpretation Guide

#### Your FVU is 0.34 - Here's What This Means:

1. **Current State**: Your transcoder explains 66% of variance (1 - 0.34 = 0.66)
2. **Target**: Aim for FVU < 0.1 (explaining 90%+ of variance)
3. **Improvement Areas**:
   - Increase dictionary size
   - Adjust learning rate schedule
   - Optimize dead feature revival
   - Fine-tune top-k selection

#### Absorption Score Analysis:

- **Low absorption (< 0.01)**: Features learn distinct, interpretable concepts
- **High absorption (> 0.05)**: Features are redundant, reducing interpretability
- **Your goal**: Minimize absorption while maintaining reconstruction quality

## 🔍 Advanced: Standalone Interpretability Evaluation

For detailed feature-level analysis:

```python
from src.eval.interpretability_metrics import InterpretabilityEvaluator

# Create evaluator
evaluator = InterpretabilityEvaluator(
    absorption_threshold=0.9,  # SAE Bench standard
    absorption_subset_size=8192,  # For memory efficiency
    dead_threshold=20  # Features dead after 20 inactive batches
)

# Evaluate your transcoder
metrics = evaluator.evaluate_transcoder(
    transcoder=your_transcoder,
    source_activations=source_batch,
    target_activations=target_batch
)

# Print results
from src.eval.interpretability_metrics import print_metrics_summary
print_metrics_summary(metrics, "Your Model")
```

## 📚 How to Download Google's Gemma Scope Transcoders

Google's transcoders are available on HuggingFace:

```bash
# Install huggingface-cli if needed
pip install huggingface-hub

# Download Gemma Scope 2B transcoders
huggingface-cli download google/gemma-scope-2b-pt-trans \
    --repo-type model \
    --local-dir /path/to/gemma_scope_2b
```

## 🎓 Research Background

### Why These Metrics?

1. **Absorption Score** (SAE Bench):
   - Validated across multiple SAE papers
   - Directly measures interpretability through feature diversity
   - Standard threshold (0.9) established by research consensus

2. **FVU** (Fraction of Variance Unexplained):
   - Gold standard for reconstruction quality
   - Used in Anthropic, OpenAI, and academic papers
   - More interpretable than raw MSE/MAE

3. **Feature Utilization**:
   - Critical for efficiency evaluation
   - High dead % indicates training issues or capacity waste
   - Standard metric in neural network interpretability

### References

- **SAE Bench**: Standard benchmark for sparse autoencoders
- **Anthropic's Monosemantic Features**: [transformer-circuits.pub/2023/monosemantic-features](https://transformer-circuits.pub/2023/monosemantic-features)
- **Gemma Scope Paper**: Google's approach to transcoder interpretability
- **Feature Splitting & Absorption**: Established in SAE literature

## 🔧 Improving Your Metrics

### If FVU is High (> 0.3):

1. **Increase capacity**: Larger dictionary size
2. **Optimize learning**: 
   - Use warmup + cosine decay schedule
   - Tune learning rate (try 4e-4 to 6e-4)
3. **Reduce dead features**:
   - Increase top-k (more active features)
   - Adjust auxiliary loss penalty
   - Better dead feature revival (reduce top_k_aux)

### If Absorption is High (> 0.05):

1. **Diversity regularization**: Add penalty for similar features
2. **Different initialization**: Xavier/He initialization
3. **Architectural changes**: Adjust group sizes in Matryoshka structure

### If Dead % is High (> 30%):

1. **Training dynamics**:
   - Lower aux_penalty (try 1/64 instead of 1/32)
   - Increase warmup steps
   - Use larger learning rate initially
2. **Architecture**:
   - Reduce top_k_aux (256 instead of 512)
   - Adjust n_batches_to_dead threshold

## 📊 Output Files

After running evaluation, you'll get:

```
analysis_results/interpretability_comparison/
├── comparison.json          # Full comparison data
├── ours_metrics.json        # Your model's metrics
├── google_metrics.json      # Google's metrics
└── visualization/           # (Optional) Plots and charts
```

### JSON Structure

```json
{
  "ours": {
    "absorption": {
      "score": 0.0123,
      "pairs_above_threshold": 15234,
      "threshold": 0.9
    },
    "feature_utilization": {
      "dead_count": 1842,
      "dead_percentage": 10.0,
      "alive_count": 16590,
      "total_features": 18432
    },
    "reconstruction": {
      "mse": 0.001234,
      "mae": 0.023456,
      "fvu": 0.3400,
      "cosine_similarity": 0.9876
    }
  },
  "google": { ... },
  "comparison": {
    "fvu_winner": "google",
    "fvu_difference": 0.02,
    ...
  }
}
```

## 💡 Tips for Better Interpretability

1. **Focus on absorption first**: This is the most direct measure of interpretability
2. **Balance reconstruction vs. sparsity**: Don't sacrifice too much accuracy for sparsity
3. **Monitor dead features during training**: High dead % early = training issues
4. **Use activation samples**: Collect examples of what features activate on (see activation_samples.py)

## 🤔 FAQ

**Q: Why is my FVU 0.34? Is this bad?**
A: It's moderate. You're explaining 66% of variance. Target < 0.1 for excellent performance.

**Q: Why does Google have no absorption score?**
A: Their transcoder is a direct linear mapping without sparse features, so absorption doesn't apply.

**Q: How many batches should I use for evaluation?**
A: 1000+ for reliable results. More batches = more accurate metrics but longer evaluation time.

**Q: What if my absorption score is NaN?**
A: This means your model doesn't have sparse features (like Google's), or the feature vectors couldn't be extracted.

## 🎯 Next Steps

1. ✅ Run evaluation on your current checkpoint
2. 📊 Compare with Google's transcoder
3. 🔍 Analyze which metrics need improvement
4. ⚙️ Adjust training configuration based on results
5. 🔄 Re-train and re-evaluate
6. 📈 Track improvements over time

---

**Need help?** Check the implementation in:
- `src/eval/interpretability_metrics.py` - Core metric implementations
- `src/eval/compare_interpretability.py` - Comparison script
- `src/scripts/evaluate_features.py` - Feature-level analysis

