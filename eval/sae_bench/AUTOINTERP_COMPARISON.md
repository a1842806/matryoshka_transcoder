# AutoInterp Implementation Comparison

Comparison between the AutoInterp paper methodology and the current implementation in `eval/sae_bench/`.

## Summary of Paper Methodology

The paper describes a 3-step pipeline:
1. **Activation Collection**: 10M tokens from RedPajama-v2, 256-token contexts
2. **Explanation Generation**: Llama-3.1-70B-Instruct shown activating contexts (top-activations, random, or stratified)
3. **Explanation Scoring**: 5 methods (Detection, Fuzzing, Surprisal, Embedding, Intervention)

## Implementation Comparison

### ✅ **What Matches the Paper**

1. **General Pipeline Structure**: The code follows the 3-step structure (activation collection → explanation → scoring)

2. **Activation Collection** (`run_autointerp.py`):
   - Uses `AutoInterpActivationCache` to collect activations
   - Computes actual sparsity using `get_feature_activation_sparsity` from sae_bench
   - Configurable token count and context length

3. **Explanation Generation**:
   - Supports configurable explainer LLM
   - Has parameters for top-activation sampling (`n_top_ex_for_generation`)
   - Supports random sampling (`n_random_ex_for_scoring`)

4. **Scoring Configuration**:
   - Can enable/disable scoring (`config.scoring`)
   - Uses sae_bench library which should implement all 5 scoring methods

### ⚠️ **Potential Issues and Differences**

#### 1. **Dataset and Scale** (Minor)
- **Paper**: 10M tokens from RedPajama-v2, 256-token contexts
- **Code**: 
  - Default 2M tokens (configurable via `--total-tokens`)
  - Context size: 128 tokens in `run_autointerp_simple.py`, configurable up to 512 in `run_autointerp.py`
  - Dataset: Fineweb-edu in simple script, configurable in main script
- **Impact**: Smaller scale might affect statistical power, but should still work

#### 2. **Sparsity Computation** (CRITICAL in `run_autointerp_simple.py`)
- **Paper**: Uses actual feature activation sparsity from the dataset
- **Code**: 
  - `run_autointerp.py`: ✅ Computes actual sparsity correctly
  - `run_autointerp_simple.py`: ❌ Uses uniform sparsity (0.01) as placeholder (line 281)
- **Impact**: **MAJOR** - Uniform sparsity is incorrect and will affect scoring accuracy
- **Fix Needed**: Compute actual sparsity like in `run_autointerp.py`

#### 3. **Sampling Strategy** (Moderate)
- **Paper**: Discusses top-activations, random, and **stratified by activation deciles**
- **Code**: 
  - Has `n_top_ex_for_generation` (top-activations)
  - Has `n_random_ex_for_scoring` (random)
  - ❌ **No explicit stratified sampling** by activation deciles
- **Impact**: Missing the stratified sampling strategy that the paper found useful for balancing specificity vs sensitivity

#### 4. **Explainer Model** (Minor)
- **Paper**: Uses Llama-3.1-70B-Instruct (or Claude 3.5)
- **Code**: 
  - `run_autointerp.py`: Default gpt-4o-mini (configurable)
  - `run_autointerp_simple.py`: Mistral-7B-Instruct (local)
- **Impact**: Smaller models may produce lower quality explanations, but methodology should still work

#### 5. **Scoring Methods** (CRITICAL - Major Mismatch)
- **Paper**: 5 methods - Detection, Fuzzing, Surprisal, Embedding, **Intervention**
- **Code**: Uses `sae_bench` library which only implements **Detection scoring** (sequence-level classification)
- **Evidence**: 
  - `get_scoring_prompts()` asks LLM to identify which sequences contain the feature (Detection)
  - `score_predictions()` computes accuracy of sequence-level predictions
  - Returns single "score" value, not multiple scores from different methods
- **Missing Methods**:
  - ❌ **Fuzzing**: Token-level prediction (which tokens activate)
  - ❌ **Surprisal**: Cross-entropy computation conditioned on explanation
  - ❌ **Embedding**: Semantic retrieval using embeddings
  - ❌ **Intervention**: Output-centric test with KL divergence (the novel method!)
- **Impact**: **MAJOR** - Only 1 out of 5 scoring methods is implemented. Missing the novel Intervention scoring which tests output effects.

#### 6. **Context Length Mismatch** (Minor)
- **Paper**: 256 tokens
- **Code**: Default 128 tokens (can be increased to 512 max)
- **Impact**: Shorter contexts might miss some activation patterns, but should still be informative

### 🔍 **Missing Features**

1. **Stratified Sampling**: No implementation of sampling by activation deciles
2. **Intervention Scoring Verification**: Cannot confirm if the novel Intervention scoring method is implemented
3. **Explicit Scoring Method Selection**: No way to enable/disable specific scoring methods (all or nothing)

## Recommendations

### High Priority Fixes

1. **⚠️ CRITICAL: Implement Missing Scoring Methods**:
   - The current implementation only uses Detection scoring
   - Missing: Fuzzing, Surprisal, Embedding, and **Intervention** (the novel method)
   - Consider implementing the full suite or using a different library that implements all methods
   - Intervention scoring is particularly important as it's the paper's novel contribution

2. **Fix sparsity computation in `run_autointerp_simple.py`**:
   ```python
   # Replace lines 277-282 with actual sparsity computation:
   from sae_bench.sae_bench_utils.activation_collection import (
       get_feature_activation_sparsity,
   )
   
   sparsity = get_feature_activation_sparsity(
       tokens=tokens,
       model=model,
       sae=sae,
       batch_size=config.llm_batch_size,
       layer=sae.cfg.hook_layer,
       hook_name=sae.cfg.hook_name,
       mask_bos_pad_eos_tokens=True,
   )
   ```

2. **✅ Verified**: sae_bench does NOT implement Intervention scoring (or Fuzzing, Surprisal, Embedding). Only Detection is implemented.

### Medium Priority Improvements

3. **Add Stratified Sampling**: Implement sampling by activation deciles for better explanation quality
4. **Increase Default Token Count**: Consider increasing default from 2M to closer to 10M tokens
5. **Increase Default Context Length**: Consider 256 tokens to match paper
6. **Add Scoring Method Selection**: Allow enabling/disabling specific scoring methods

### Low Priority

7. **Use Larger Explainer Model**: Consider using larger models (70B+) for better explanations
8. **Dataset Alignment**: Consider using RedPajama-v2 to match paper exactly

## Verification Steps

1. **✅ Completed**: Verified that sae_bench only implements Detection scoring
   - Checked `get_scoring_prompts()` and `score_predictions()` methods
   - Confirmed single score output (sequence-level classification accuracy)
   - No evidence of Fuzzing, Surprisal, Embedding, or Intervention methods

2. **Next Steps**:
   - Decide if Detection-only scoring is sufficient for your use case
   - If full methodology is needed, consider:
     - Implementing the missing 4 methods yourself
     - Using the official AutoInterp implementation from the paper authors
     - Using EleutherAI/delphi toolkit (mentioned in paper summary)

3. **For Intervention Scoring** (if needed):
   - Requires: feature intervention (amplify/clamp), KL divergence computation
   - Measures: whether explanation predicts output distribution changes
   - This is the paper's key contribution for "output features"

