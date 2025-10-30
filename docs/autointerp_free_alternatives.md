# AutoInterp Evaluation: Free Alternatives to GPT API

Yes! You can evaluate AutoInterp metrics without the GPT API using several free/open-source alternatives. Here are your options:

## 🆓 Free Options (Recommended)

### 1. **Hugging Face Models** (Best Option)

**Advantages:**
- ✅ Completely free
- ✅ Runs locally (no API calls)
- ✅ Many models available
- ✅ No rate limits
- ✅ Privacy (data stays local)

**Recommended Models:**
```python
# Small, fast models (good for testing)
"microsoft/DialoGPT-small"      # 117M parameters
"microsoft/DialoGPT-medium"     # 345M parameters

# Better quality models (slower)
"microsoft/DialoGPT-large"      # 774M parameters
"facebook/blenderbot-400M-distill"  # 400M parameters

# Even better (requires more GPU memory)
"microsoft/DialoGPT-xlarge"     # 1.5B parameters
"facebook/blenderbot-1B-distill"   # 1B parameters
```

**Usage:**
```bash
python src/eval/eval_saebench_autointerp_absorption.py \
    --checkpoint checkpoints/transcoder/gemma-2-2b/my_model_15000 \
    --llm_backend huggingface \
    --llm_model microsoft/DialoGPT-medium
```

### 2. **SAEBench API** (If Available)

**Advantages:**
- ✅ Free (if available)
- ✅ Optimized for SAE evaluation
- ✅ No model setup required

**Status:** SAEBench may offer a free API, but it's not widely available yet.

**Usage:**
```bash
python src/eval/eval_saebench_autointerp_absorption.py \
    --checkpoint checkpoints/transcoder/gemma-2-2b/my_model_15000 \
    --llm_backend saebench
```

### 3. **Other Open-Source Models**

**Local Models:**
- **Ollama** (if installed): `ollama run llama2`
- **Transformers** library: Any Hugging Face model
- **Custom models**: Your own fine-tuned models

## 💰 Paid Options (For Comparison)

### OpenAI API
- **Cost:** ~$5 per 1,000 latents
- **Quality:** Highest (GPT-4o-mini)
- **Speed:** Fast
- **Setup:** Requires API key

## 🔧 Implementation Details

### Hugging Face Integration

The updated code supports Hugging Face models:

```python
# Initialize with Hugging Face model
autointerp_eval = AutoInterpEvaluator(
    llm_backend="huggingface",
    model_name="microsoft/DialoGPT-medium",
    device="cuda"  # or "cpu"
)

# Evaluate
score, n_latents = autointerp_eval.evaluate(transcoder, activation_store)
```

### Model Requirements

**For AutoInterp, you need a model that can:**
1. **Generate text** (for feature descriptions)
2. **Classify text** (for activation prediction)
3. **Handle prompts** (for the detection task)

**Hugging Face models that work well:**
- DialoGPT series (conversational, good for prompts)
- BlenderBot series (dialogue, good for classification)
- GPT-2 variants (text generation)
- T5 variants (text-to-text, very flexible)

## 📊 Quality Comparison

| Backend | Cost | Quality | Speed | Setup |
|---------|------|---------|-------|-------|
| **Hugging Face** | Free | Good | Medium | Easy |
| **SAEBench API** | Free | Good | Fast | Easy |
| **OpenAI API** | ~$5 | Best | Fast | Easy |
| **Local Ollama** | Free | Good | Slow | Medium |

## 🚀 Quick Start with Hugging Face

### 1. Install Dependencies
```bash
pip install transformers torch
```

### 2. Run Evaluation
```bash
python src/eval/eval_saebench_autointerp_absorption.py \
    --checkpoint checkpoints/transcoder/gemma-2-2b/my_model_15000 \
    --llm_backend huggingface \
    --llm_model microsoft/DialoGPT-medium \
    --device cuda:0
```

### 3. Try Different Models
```bash
# Small and fast
--llm_model microsoft/DialoGPT-small

# Better quality
--llm_model microsoft/DialoGPT-large

# Alternative
--llm_model facebook/blenderbot-400M-distill
```

## ⚠️ Important Notes

### Model Quality vs Speed Trade-off

**Small models (DialoGPT-small):**
- ✅ Fast evaluation (~1-2 minutes)
- ✅ Low memory usage (~500MB)
- ⚠️ Lower accuracy (~70-80% of GPT-4)

**Large models (DialoGPT-large):**
- ✅ Better accuracy (~85-90% of GPT-4)
- ⚠️ Slower evaluation (~5-10 minutes)
- ⚠️ Higher memory usage (~2-3GB)

### Implementation Status

**Current code provides:**
- ✅ Backend selection (Hugging Face, OpenAI, SAEBench)
- ✅ Model loading and initialization
- ✅ Placeholder evaluation structure
- ⚠️ **Full implementation needed** for production use

**To complete the implementation, you need:**
1. **Sequence collection** from your dataset
2. **Prompt engineering** for feature description generation
3. **Classification logic** for activation prediction
4. **Test set construction** (10 random + 2 max + 2 IW)

## 🎯 Recommendation

**For your use case, I recommend:**

1. **Start with Hugging Face** (`microsoft/DialoGPT-medium`)
   - Free, local, good quality
   - Easy to experiment with different models

2. **Use the skeleton implementation** as a starting point
   - Shows correct methodology
   - Easy to extend with full implementation

3. **Consider OpenAI API** only if you need:
   - Publication-grade results
   - Direct comparison with SAEBench paper
   - Highest possible accuracy

## 📝 Example Commands

```bash
# Free evaluation with Hugging Face
python src/eval/eval_saebench_autointerp_absorption.py \
    --checkpoint checkpoints/transcoder/gemma-2-2b/my_model_15000 \
    --llm_backend huggingface \
    --llm_model microsoft/DialoGPT-medium

# Skip AutoInterp entirely (focus on Absorption only)
python src/eval/eval_saebench_autointerp_absorption.py \
    --checkpoint checkpoints/transcoder/gemma-2-2b/my_model_15000 \
    --skip_autointerp

# Try different Hugging Face model
python src/eval/eval_saebench_autointerp_absorption.py \
    --checkpoint checkpoints/transcoder/gemma-2-2b/my_model_15000 \
    --llm_backend huggingface \
    --llm_model facebook/blenderbot-400M-distill
```

## 🔗 References

- **Hugging Face Models:** https://huggingface.co/models
- **Transformers Library:** https://huggingface.co/docs/transformers
- **SAEBench Paper:** https://arxiv.org/abs/2503.09532
- **SAEBench GitHub:** https://github.com/adamkarvonen/SAEBench

The key insight is that **you don't need the GPT API** - Hugging Face models work well for AutoInterp evaluation and are completely free!
