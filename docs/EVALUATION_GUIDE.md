# Evaluation Guide

This guide consolidates the interpretability, SAEBench, and feature-analysis notes into a single reference for assessing Matryoshka transcoders and SAEs.

## Goals & Scope

- Track reconstruction quality, sparsity, and interpretability consistently across runs.
- Support both internal tooling and external benchmarks (SAEBench, AutoInterp).
- Provide actionable guidance even while the evaluation code is being rewritten.

## Core Metrics

| Metric | What it measures | Healthy Range | Notes |
|--------|------------------|---------------|-------|
| **FVU** (Fraction of Variance Unexplained) | Normalized reconstruction error | `< 0.1` great, `0.1–0.3` OK | Preferable to raw L2; log `fvu_min`, `fvu_mean`, `fvu_max` per group |
| **Absorption** | Redundant features via cosine similarity | `< 0.05` | Counts fraction of feature pairs with cosine > 0.9 |
| **Splitting** | Co-activation frequency | `< 0.1` | High values suggest merging features |
| **Dead Features** | Unused dictionary slots | `< 10%` | Monitor with moving averages during training |
| **Monosemanticity** | Specificity of activations | `> 0.7` | Combine sparsity + activation consistency |

Nested-group models should report metrics per cumulative group so you can compare coarse vs fine representations.

## AutoInterp Without GPT APIs

AutoInterp scores can be reproduced without commercial APIs:

1. **Hugging Face conversational models** (DialoGPT, BlenderBot, etc.)
   ```bash
   python src/eval/eval_saebench_autointerp_absorption.py \
       --checkpoint results/gemma-2-2b/8/7000/checkpoints/model.pt \
       --llm-backend huggingface \
       --llm-model microsoft/DialoGPT-medium
   ```
2. **SAEBench backend** (when available) – drop-in replacement with `--llm-backend saebench`.
3. **Local frameworks** (Ollama, custom transformers) – integrate via the adapter interface in `src/adapters/`.

Collect top-activating and importance-weighted samples per latent, generate descriptions, then evaluate detection accuracy on a held-out set.

## SAEBench Compatibility

- Sample ~1,000 non-dead latents to mirror benchmark statistics.
- Format prompts with highlighted tokens (`<<token>>`) before sending to the LLM judge.
- Track accuracy, precision, and recall per latent; aggregate by percentile to compare against published baselines.

## Recommended Workflow

1. **Pre-Evaluation**
   - Confirm training completed and metrics were logged.
   - Ensure activation samples exist if you plan to generate qualitative reports.
2. **Metric Computation**
   - Run the rebuilt evaluation scripts (once restored) or notebooks that compute FVU, absorption, splitting, and dead feature counts.
   - Store outputs under `results/{model}/{layer}/{steps}/metrics.json`.
3. **Interpretation**
   - Inspect worst-performing features (highest absorption/splitting).
   - Use activation samples to verify monosemanticity and document interesting behaviors.
4. **Reporting**
   - Summarize metrics by group and compare against baselines.
   - Archive LLM-generated descriptions alongside numeric scores.

## Qualitative Analysis Tips

- Activation samples help validate whether low absorption actually corresponds to distinct concepts.
- Track how nested groups change feature semantics—coarse groups should capture high-level structure, fine groups fill in detail.
- Document anomalies (e.g., persistent dead features) so training configs can be adjusted in the next run.

## Next Steps

- Rebuild the evaluation scripts with modular components so they mirror the new results directory.
- Integrate SAEBench adapters under `src/adapters/` for a clean separation between external APIs and core logic.
- Extend metric computation to handle cross-layer transcoders and feature hierarchies automatically.

