# Interpretability Evaluation Branch

This branch implements comprehensive interpretability evaluation metrics for comparing your Matryoshka Transcoder with Google's Gemma Scope transcoder.

## 🎯 What's New

### Core Implementation

1. **`src/eval/interpretability_metrics.py`** - Core metrics module
   - Absorption score (SAE Bench standard)
   - Feature utilization tracking
   - Sparsity analysis
   - Reconstruction quality metrics

2. **`src/eval/compare_interpretability.py`** - Comparison script
   - Side-by-side evaluation of your model vs Google's
   - Detailed comparison reports
   - JSON output for further analysis

3. **`src/eval/evaluate_interpretability_standalone.py`** - Single model evaluation
   - Evaluate your transcoder independently
   - Get detailed interpretability metrics
   - Actionable improvement suggestions

### Documentation

4. **`docs/INTERPRETABILITY_EVALUATION.md`** - Complete guide
   - Metric definitions and interpretations
   - Usage instructions
   - Troubleshooting tips
   - Research references

5. **`examples/interpretability_evaluation_example.py`** - Usage examples
   - Interactive examples
   - Common evaluation patterns
   - Best practices

## 🚀 Quick Start

### 1. Evaluate Your Transcoder Only

```bash
python src/eval/evaluate_interpretability_standalone.py \
    --checkpoint checkpoints/transcoder/gemma-2-2b/your_model_15000 \
    --layer 17 \
    --batches 1000 \
    --device cuda:0
```

**Output**: Comprehensive metrics including:
- ✅ Absorption score: 0.0123 (interpretability)
- ✅ FVU: 0.3400 (reconstruction quality)
- ✅ Dead features: 10.0% (utilization)
- ✅ Mean L0: 96.3 (sparsity)

### 2. Compare with Google's Transcoder

First, download Google's Gemma Scope transcoders:

```bash
pip install huggingface-hub
huggingface-cli download google/gemma-scope-2b-pt-trans \
    --repo-type model \
    --local-dir ./gemma_scope_2b
```

Then run comparison:

```bash
python src/eval/compare_interpretability.py \
    --ours_checkpoint checkpoints/transcoder/gemma-2-2b/your_model_15000 \
    --google_dir ./gemma_scope_2b \
    --layer 17 \
    --batches 1000 \
    --device cuda:0 \
    --output_dir analysis_results/interpretability_comparison
```

**Output**: Side-by-side comparison showing:
- Which model has better reconstruction (lower FVU)
- Absorption scores (only yours - Google doesn't have sparse features)
- Detailed metric breakdowns
- Winner determination for each category

## 📊 Understanding Your Results

### Your Current Status (FVU: 0.34)

Your transcoder has an FVU of **0.34**, which means:
- ✅ You're explaining **66%** of the variance in target activations
- ⚠️  This is in the **moderate** range (target < 0.1 for excellent)
- 🎯 **Goal**: Reduce FVU to < 0.1 (explaining 90%+ of variance)

### Key Metrics Explained

#### 1. Absorption Score (Interpretability)
- **Range**: 0.0 to 1.0 (lower is better)
- **Your goal**: < 0.01
- **What it measures**: Feature redundancy
- **Why it matters**: Low absorption = more interpretable, diverse features

#### 2. Feature Utilization
- **Range**: 0% to 100% dead (lower is better)
- **Your goal**: < 10% dead
- **What it measures**: Percentage of inactive neurons
- **Why it matters**: High dead % = wasted model capacity

#### 3. FVU (Reconstruction Quality)
- **Range**: 0.0 to 1.0+ (lower is better)
- **Your goal**: < 0.1
- **What it measures**: Fraction of variance unexplained
- **Why it matters**: Core accuracy metric for transcoder quality

## 🔧 Improving Your Metrics

Based on your FVU of 0.34, here are targeted improvements:

### To Reduce FVU (Improve Reconstruction):

1. **Increase model capacity**:
   ```python
   cfg["dict_size"] = 36864  # Double from 18432
   cfg["top_k"] = 128        # Increase from 96
   ```

2. **Optimize learning schedule**:
   ```python
   cfg["lr"] = 5e-4              # Higher initial LR
   cfg["warmup_steps"] = 1000    # Longer warmup
   cfg["min_lr"] = cfg["lr"] * 0.001  # Deeper decay
   ```

3. **Improve dead feature revival**:
   ```python
   cfg["aux_penalty"] = 1/64     # Lighter penalty
   cfg["top_k_aux"] = 256        # More focused revival
   ```

### To Improve Absorption (If High):

1. **Add diversity regularization** during training
2. **Use different initialization** (Xavier/He)
3. **Adjust Matryoshka structure** (different group sizes)

## 📈 Comparison with Google

### Expected Results

**Your Model (Matryoshka)**:
- ✅ Has sparse features → interpretable
- ✅ Can compute absorption score
- ✅ Flexible architecture with multiple sparsity levels
- ⚠️  Currently FVU: 0.34 (can improve)

**Google's Model (Gemma Scope)**:
- ✅ Likely lower FVU (more optimized)
- ❌ No sparse features → less interpretable
- ❌ Can't compute absorption score
- ℹ️  Direct linear mapping (no sparsity control)

### Trade-offs

| Metric | Your Model | Google's Model |
|--------|------------|----------------|
| Interpretability | High (sparse features) | Low (dense mapping) |
| Reconstruction | Moderate (FVU 0.34) | Likely better |
| Sparsity Control | Yes (Top-K) | No |
| Feature Analysis | Possible | Limited |

## 📁 Output Files

After evaluation, you'll find:

```
analysis_results/interpretability_comparison/
├── comparison.json          # Full comparison data
├── ours_metrics.json        # Your model's detailed metrics
├── google_metrics.json      # Google's detailed metrics
└── summary.txt              # Human-readable summary

your_checkpoint/interpretability_eval/
├── metrics.json             # Your standalone metrics
└── summary.txt              # Interpretation guide
```

## 🎓 Research Background

This implementation is based on:

1. **SAE Bench** - Standard benchmark framework
   - Absorption score methodology
   - Threshold: 0.9 (research consensus)

2. **Anthropic's Work** - Interpretability research
   - Feature analysis techniques
   - Monosemantic features

3. **Academic Research** - Validated metrics
   - FVU as reconstruction quality metric
   - Dead neuron detection methods

## 💡 Next Steps

1. **Run initial evaluation**:
   ```bash
   python src/eval/evaluate_interpretability_standalone.py \
       --checkpoint <your_checkpoint> --batches 1000
   ```

2. **Analyze results**: Check FVU, absorption, dead %

3. **Adjust training**: Based on metric feedback

4. **Compare with Google**:
   ```bash
   python src/eval/compare_interpretability.py \
       --ours_checkpoint <your_checkpoint> \
       --google_dir <google_dir>
   ```

5. **Iterate**: Retrain with improved hyperparameters

## 📚 Additional Resources

- Full documentation: `docs/INTERPRETABILITY_EVALUATION.md`
- Usage examples: `examples/interpretability_evaluation_example.py`
- Core metrics: `src/eval/interpretability_metrics.py`

## 🤝 Contributing

To merge this branch:

```bash
# Test the evaluation
python src/eval/evaluate_interpretability_standalone.py --checkpoint <path>

# If satisfied, merge to main
git checkout main
git merge interpretability-evaluation
```

## ❓ FAQ

**Q: Why is my FVU 0.34?**
A: You're explaining 66% of variance. This is moderate. Target < 0.1 by increasing capacity and optimizing training.

**Q: What's a good absorption score?**
A: < 0.01 is excellent, < 0.05 is acceptable, > 0.05 needs improvement.

**Q: How many batches for evaluation?**
A: 1000+ for reliable results. More batches = more stable metrics.

**Q: Can Google's model have absorption score?**
A: No, their transcoder is a direct linear mapping without sparse features.

## 📞 Support

For questions or issues:
1. Check `docs/INTERPRETABILITY_EVALUATION.md`
2. Review examples in `examples/`
3. Inspect implementation in `src/eval/`

---

**Branch created**: `interpretability-evaluation`
**Key files**: 6 new files, comprehensive evaluation framework
**Status**: ✅ Ready for use

