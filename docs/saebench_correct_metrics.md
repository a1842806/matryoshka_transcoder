# SAEBench Correct Metrics: AutoInterp & Absorption

This document describes the **actual** AutoInterp and Absorption metrics from the SAEBench paper, not simplified approximations.

## ⚠️ Important Distinction

**What I initially implemented:** Simple feature redundancy via cosine similarity (NOT the real SAEBench metric)

**What SAEBench actually measures:** 
1. AutoInterp - LLM-based detection task
2. Absorption - First-letter probe task with compensation detection

## 1. AutoInterp (LLM-Based Detection)

### What It Measures
Feature interpretability via an LLM judge that:
1. Generates a description from top-activating examples
2. Predicts which sequences activate the feature (detection task)
3. Accuracy = interpretability score

### Methodology (SAEBench/Paulo et al., 2024)

**Phase 1: Description Generation**
- Input: Top-k (k=2) + importance-weighted samples for a latent
- Format: Highlight activating tokens with `<<token>>` syntax
- Prompt: "Given these examples, write a short description of what activates this latent"
- Output: Feature description

**Phase 2: Detection Task**
- Test set: 10 random + 2 max + 2 importance-weighted sequences
- Task: LLM predicts which sequences activate the latent
- Ground truth: Actual activation > threshold (e.g., top percentile)
- Score: Prediction accuracy

**Key Parameters (SAEBench Defaults)**
- Dataset: OpenWebText
- Context length: 128 tokens
- Latents sampled: 1,000 non-dead latents
- LLM judge: GPT-4o-mini
- Test composition: 10 random + 2 max + 2 IW per latent

### Implementation Requirements

```python
# 1. Collect activations
acts = encode_sequences_collect_latent_acts(model, sae, seqs)  # [N_seq, N_lat]

for latent_id in sampled_latents:
    # 2. Generate description (top-k + IW samples)
    gen_pool = select_top_and_iw_sequences(acts[:, latent_id], seqs, top_k=2)
    desc = llm_generate_description(gen_pool, highlight_tokens=True)
    
    # 3. Build test set
    test_set = build_test_set(
        acts[:, latent_id], seqs, 
        n_rand=10, n_max=2, n_iw=2
    )
    
    # 4. LLM predicts activations
    preds = llm_predict_activation(desc, test_set)  # list[bool]
    labels = compute_true_activation_labels(acts[:, latent_id], test_set)
    
    # 5. Score = accuracy
    score = accuracy(preds, labels)
```

### Dependencies
- **LLM API**: OpenAI GPT-4 (requires API key)
- **Cost**: ~$1-5 per 1,000 latents with GPT-4o-mini
- **Time**: ~30-60 minutes for 1,000 latents

---

## 2. Absorption Score (First-Letter Probe Task)

### What It Measures
Feature absorption/compensation via first-letter classification task.

**Concept:** Detects when SAE represents hierarchical concepts (e.g., "starts with S" → "is a letter") using:
- **Main latents**: Directly encode the concept
- **Absorbing latents**: Compensate when main latents underfire

### Methodology (SAEBench/Chanin et al.)

**Step 1: Train Logistic Probes**
- Filter vocabulary to single-letter tokens (a-z)
- Split: 80% train, 20% test
- Train logistic regression on residual stream to classify first letter
- Probe weight vector = `p` (unit-normalized)

**Step 2: Find Main Latents (K-Sparse Probing)**
- For each letter, greedily select latents:
  - Start with k=1 (best single latent)
  - Increase k if F1 improves by > τ_fs = 0.03
  - Stop at k_max = 10
- Result: Set of "main" latents S_main per letter

**Step 3: Detect Absorption on Test Split**

For each test token (correct probe prediction only):

**Condition 1: Main latents underfire**
```
Σ(a_i * <d_i, p>) < <a_model, p>
where i ∈ S_main
```

**Condition 2: Absorbers compensate**
```
Select A_max absorbing candidates with:
- Highest <d_i, p> 
- Cosine ≥ τ_ps (default: -1.0, no filter)
- Positive projection on p

Require: Σ(a_i * <d_i, p>) / <a_model, p> ≥ τ_pa (default: 0.0)
where i ∈ S_abs
```

**Absorption Score (Token-Level)**
```
Absorption = Σ(a_i * <d_i, p>) / [Σ(a_i * <d_i, p>) + Σ(a_j * <d_j, p>)]
             i∈S_abs                    i∈S_abs              j∈S_main
```

If conditions not met: absorption = 0

**Final Metric**
```
SAEBench reports: 1 - mean(absorption)
Higher is better (less absorption)
```

### SAEBench Defaults (Appendix D, Table 8)

| Parameter | Value | Description |
|-----------|-------|-------------|
| k_max | 10 | Max k for k-sparse probing |
| τ_fs | 0.03 | F1 improvement threshold for split |
| τ_pa | 0.0 | Compensation threshold |
| τ_ps | -1.0 | Cosine threshold (no filter) |
| A_max | dict_size | Max absorbing candidates |
| Train/Test | 80/20 | Split ratio |

### Implementation Requirements

```python
# 1. Build ground-truth probes
letters = ['a', 'b', ..., 'z']
train_tok, test_tok = split_letter_tokens(vocab, ratio=0.8)
probes = {L: train_logreg_probe(resid(train_tok[L]), labels=L) for L in letters}

# 2. Find main latents (k-sparse on train)
S_main = {}
for L in letters:
    S_main[L] = greedy_k_sparse_latents(
        sae_acts(train_tok[L]), labels=L,
        k_max=10, delta_f1=0.03
    )

# 3. Compute absorption (on test)
scores = []
for L in letters:
    p = unit(probes[L].weight)  # unit-norm probe direction
    
    for tok in test_tok[L]:
        if not probe_correct(probes[L], resid(tok)):
            continue
        
        # Main latents contribution
        main_sum = sum(a[i] * dot(d[i], p) for i in S_main[L])
        model_proj = dot(resid(tok), p)
        
        if main_sum >= model_proj:
            scores.append(0.0)
            continue
        
        # Find absorbers
        S_abs = top_absorbers_by_probe_projection(
            acts, decoder, p, A_max=dict_size,
            tau_ps=-1, require_positive=True
        )
        
        # Compute absorption
        abs_sum = sum(a[i] * dot(d[i], p) for i in S_abs)
        
        if abs_sum / model_proj < 0.0:  # tau_pa = 0
            scores.append(0.0)
            continue
        
        scores.append(abs_sum / (abs_sum + main_sum))

absorption = mean(scores)
reported_metric = 1.0 - absorption  # SAEBench convention
```

### Dependencies
- scikit-learn (LogisticRegression)
- Vocabulary filtering
- Residual stream activation collection
- K-sparse probing implementation

---

## Comparison: My Old Implementation vs SAEBench

| Metric | My Old Implementation | SAEBench Actual |
|--------|----------------------|-----------------|
| **Absorption** | Pairwise cosine similarity > 0.9 between decoder weights | First-letter probe task with compensation detection |
| **AutoInterp** | ❌ Not implemented | LLM-based detection task |
| **Complexity** | Simple (5 lines) | Complex (200+ lines each) |
| **Cost** | Free | ~$5 per evaluation (AutoInterp) |
| **Time** | Seconds | Hours |

## What My Original Code Actually Measured

My `compute_absorption_score()` function measured **feature redundancy** via cosine similarity:

```python
# This is NOT the SAEBench Absorption Score
V = F.normalize(feature_vectors, p=2, dim=-1)
S = torch.abs(V @ V.T)
absorption = (S > 0.9).sum() / total_pairs
```

This is a **valid interpretability metric** (used in some papers), but it's **not** the SAEBench Absorption Score.

## Implementation Status

The script `eval_saebench_autointerp_absorption.py` provides:

✅ **Skeleton structure** showing correct methodology
✅ **Correct formulas** and equations from SAEBench
✅ **SAEBench defaults** and hyperparameters
⚠️ **Placeholder implementations** (requires full implementation)

**To fully implement:**

1. **AutoInterp:**
   - Sequence collection and caching
   - LLM API integration (OpenAI)
   - Top-activating sample selection
   - Test set construction and evaluation

2. **Absorption:**
   - Vocabulary filtering (letter tokens)
   - Residual stream activation collection
   - Logistic regression probe training
   - K-sparse probing algorithm
   - Absorption computation with conditions

## Alternative: Use SAEBench Directly

For production use, consider using the official SAEBench repo:

```bash
git clone https://github.com/adamkarvonen/SAEBench
cd SAEBench

# Wrap your transcoder as a custom SAE
class CustomSAE:
    def encode(self, x): return your_transcoder.encode(x)
    def decode(self, x): return your_transcoder.decode(x)
    
# Run eval
python eval_autointerp.py --sae custom_sae --config eval_config.py
```

## References

- **SAEBench Paper:** arXiv:2503.09532
- **SAEBench GitHub:** https://github.com/adamkarvonen/SAEBench
- **AutoInterp (Paulo et al.):** Detection-style interpretability evaluation
- **Absorption (Chanin et al.):** First-letter probe task

## Summary

**Key Takeaway:** SAEBench's AutoInterp and Absorption are **complex, multi-step procedures** requiring:
- LLM API access (AutoInterp)
- Logistic regression probes (Absorption)
- K-sparse probing (Absorption)
- Careful test set construction

They are **not** simple similarity metrics. The skeleton implementation shows the correct structure, but full implementation requires significant additional code.

