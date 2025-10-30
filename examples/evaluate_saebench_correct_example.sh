#!/bin/bash

# Example: SAEBench AutoInterp & Absorption Evaluation
#
# This demonstrates the CORRECT SAEBench metrics:
# 1. AutoInterp - LLM-based detection task
# 2. Absorption - First-letter probe task
#
# Note: This is a skeleton implementation showing the methodology.
# Full implementation requires additional components (see documentation).

echo "======================================"
echo "SAEBench Evaluation (AutoInterp + Absorption)"
echo "======================================"

# Configuration
CHECKPOINT="checkpoints/transcoder/gemma-2-2b/my_model_15000"
LAYER=17
OUTPUT_DIR="analysis_results/saebench"

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT"
echo "  Layer: $LAYER"
echo "  Output: $OUTPUT_DIR"
echo ""

# Run evaluation with Hugging Face (free alternative to GPT API)
python src/eval/eval_saebench_autointerp_absorption.py \
    --checkpoint "$CHECKPOINT" \
    --layer $LAYER \
    --device cuda:0 \
    --dtype bfloat16 \
    --output_dir "$OUTPUT_DIR" \
    --llm_backend huggingface \
    --llm_model microsoft/DialoGPT-medium

echo ""
echo "======================================"
echo "⚠️  IMPLEMENTATION NOTE"
echo "======================================"
echo ""
echo "This is a SKELETON showing SAEBench methodology."
echo ""
echo "Full implementation requires:"
echo ""
echo "AutoInterp:"
echo "  - Sequence collection from OpenWebText (ctx=128)"
echo "  - LLM integration (Hugging Face models - FREE!)"
echo "  - Top-k + importance-weighted sampling"
echo "  - Test set: 10 random + 2 max + 2 IW per latent"
echo ""
echo "Free alternatives to GPT API:"
echo "  - Hugging Face models (microsoft/DialoGPT-medium)"
echo "  - SAEBench API (if available)"
echo "  - Local models (Ollama, etc.)"
echo ""
echo "Absorption:"
echo "  - Letter token filtering (a-z)"
echo "  - Residual stream activation collection"
echo "  - Logistic regression probes (per letter)"
echo "  - K-sparse probing (k_max=10, tau_fs=0.03)"
echo "  - Absorption computation on test split"
echo ""
echo "For production, consider using official SAEBench:"
echo "  https://github.com/adamkarvonen/SAEBench"
echo ""
echo "======================================"

