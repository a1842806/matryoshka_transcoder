#!/bin/bash
# Simple script to compare Matryoshka vs Google transcoders using AutoInterp
#
# Usage (with OpenAI):
#   ./eval/compare_transcoders.sh LAYER MATRYOSHKA_CHECKPOINT GOOGLE_NPZ [N_LATENTS] [TOTAL_TOKENS]
#
# Usage (with local Mixtral):
#   USE_LOCAL_LLM=1 ./eval/compare_transcoders.sh LAYER MATRYOSHKA_CHECKPOINT GOOGLE_NPZ [N_LATENTS] [TOTAL_TOKENS]

set -e

LAYER=${1:?Layer number required}
MATRYOSHKA=${2:?Matryoshka checkpoint path required}
GOOGLE=${3:?Google transcoder path required}
N_LATENTS=${4:-50}
TOTAL_TOKENS=${5:-2000000}

echo "=========================================="
echo "AutoInterp Transcoder Comparison"
echo "=========================================="
echo "Layer: $LAYER"
echo "Matryoshka: $MATRYOSHKA"
echo "Google: $GOOGLE"
echo "Features to evaluate: $N_LATENTS"
echo "Total tokens: $TOTAL_TOKENS"

# Check for LLM setup
if [ -n "$USE_LOCAL_LLM" ]; then
    echo "LLM: Local Mixtral (2x RTX 3090)"
    LOCAL_LLM_FLAG="--use-local-llm"
else
    echo "LLM: OpenAI API"
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "ERROR: OPENAI_API_KEY not set. Use 'export OPENAI_API_KEY=...' or set USE_LOCAL_LLM=1"
        exit 1
    fi
    LOCAL_LLM_FLAG=""
fi

echo "=========================================="

OUTPUT_DIR="autointerp_results/layer${LAYER}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Run evaluation
python eval/sae_bench/run_autointerp_simple.py \
    --matryoshka "$MATRYOSHKA" \
    --google "$GOOGLE" \
    --layer "$LAYER" \
    --n-latents "$N_LATENTS" \
    --total-tokens "$TOTAL_TOKENS" \
    --output-dir "$OUTPUT_DIR" \
    $LOCAL_LLM_FLAG

echo ""
echo "Results saved to: $OUTPUT_DIR"

