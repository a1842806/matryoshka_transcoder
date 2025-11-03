# Evaluation & Metrics

Consolidated reference for evaluating Matryoshka SAEs and transcoders. Covers internal feature-quality metrics, SAEBench-compatible procedures, and AutoInterp backends.

## Goals & Scope

- Track reconstruction quality, sparsity, and interpretability consistently across runs.
- Support internal tooling and external benchmarks (SAEBench, AutoInterp).
- Provide actionable workflows and consistent output locations.

## Core Metrics (Internal)

| Metric | What it measures | Healthy Range | Notes |
|--------|------------------|---------------|-------|
| FVU (Fraction of Variance Unexplained) | Normalized reconstruction error | `< 0.1` great, `0.1–0.3` OK | Preferable to raw L2; for Matryoshka report `fvu_min/mean/max` per group |
| Absorption (cosine) | Feature redundancy via decoder-weight similarity | `< 0.05` | Fraction of pairs with cosine > 0.9 |
| Splitting (co-activation) | Features activating together | `< 0.1` | High values suggest merging features |
| Dead features | Unused dictionary slots | `< 10%` | Monitor with moving averages during training |
| Monosemanticity | Specificity of activations | `> 0.7` | Combine sparsity + activation consistency |

Matryoshka models should report metrics per cumulative group to compare coarse vs fine representations.

## SAEBench-Compatible (Correct) Metrics

SAEBench metrics differ from simple internal proxies. Two important ones:

### 1) AutoInterp (LLM-based detection)

Pipeline (per latent):
1. Generate a short description from top-activating and importance-weighted samples (highlight with `<<token>>`).
2. Build a test set (10 random + 2 max + 2 importance-weighted sequences).
3. Ask the LLM to predict which sequences activate the latent.
4. Score = accuracy vs ground truth (activation thresholding).

Defaults (paper): ~1,000 non-dead latents; context length 128; GPT-4o-mini judge; OpenWebText samples.

### 2) Absorption (first-letter probe)

Measures compensation when “main” latents underfire relative to a probe of the residual stream.

Absorption (token-level):

```
Absorption = Σ(a_i * <d_i, p>) / [Σ(a_i * <d_i, p>) + Σ(a_j * <d_j, p>)]
             i∈S_abs                    i∈S_abs              j∈S_main
```

SAEBench reports `1 - mean(absorption)` (higher is better = less absorption).

Key steps: train letter-class probes; select main latents via k-sparse probing; compute compensation by absorbers on test tokens; aggregate.

### Implementation Notes

- The script skeleton `src/eval/eval_saebench_autointerp_absorption.py` reflects the correct methodology but requires full implementations (probe training, sequence collection, LLM integration).
- For production-grade results, use SAEBench directly or finish the skeleton; costs/time apply for LLM evaluation.

## AutoInterp Without GPT APIs (Free Backends)

You can reproduce AutoInterp flows without paid APIs using local Hugging Face models:

```bash
python src/eval/eval_saebench_autointerp_absorption.py \
    --checkpoint results/gemma-2-2b/8/7000/checkpoints/model.pt \
    --llm-backend huggingface \
    --llm-model microsoft/DialoGPT-medium
```

Other options: SAEBench backend when available, or local frameworks (Ollama) via adapter.

## Recommended Workflow

1. Pre-checks
   - Confirm training completed and metrics were logged.
   - Ensure activation samples exist if you plan qualitative reports.
2. Metric computation
   - Run evaluation scripts/notebooks to compute FVU, absorption (cosine), splitting, dead features.
   - Store outputs under `results/{model}/{layer}/{steps}/metrics.json`.
3. SAEBench-compatible analysis (optional)
   - Use the skeleton adapter or SAEBench directly for AutoInterp and probe-based Absorption.
4. Interpretation & reporting
   - Inspect worst-performing features (highest absorption/splitting).
   - Summarize metrics by group and compare against baselines; archive LLM-generated descriptions.

## Outputs

- Numeric metrics: `results/{model}/{layer}/{steps}/metrics.json`
- Optional artifacts: `analysis_results/...` for richer comparisons/plots.

Example JSON snippet:

```json
{
  "absorption": {"score": 0.0123, "threshold": 0.9},
  "dead_features": {"count": 1842, "pct": 10.0},
  "reconstruction": {"fvu": 0.3400, "mse": 0.001234}
}
```

## Tips

- Activation samples help validate whether low absorption corresponds to distinct concepts.
- Track how nested groups change semantics—coarse groups capture high-level structure; fine groups add detail.
- Prefer FVU over raw L2 for cross-run/model comparability.


