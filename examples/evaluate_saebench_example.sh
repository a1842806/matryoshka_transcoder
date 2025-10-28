#!/bin/bash

# Example: Evaluate Matryoshka Transcoder vs Google's Transcoder using SAEBench metrics
#
# This script demonstrates how to run interpretability evaluation comparing:
# - Your trained Matryoshka Transcoder
# - Google's Gemma Scope Transcoder
#
# Metrics evaluated:
# 1. Absorption Score - feature redundancy (lower = better)
# 2. FVU - reconstruction quality (lower = better)  
# 3. Sparsity - L0 distribution
# 4. Feature Utilization - dead vs alive features

echo "=================================="
echo "SAEBench Interpretability Evaluation"
echo "=================================="

# Configuration
OURS_CHECKPOINT="checkpoints/transcoder/gemma-2-2b/my_model_15000"
GOOGLE_DIR="/path/to/gemma_scope_2b"
LAYER=17
BATCHES=1000
OUTPUT_DIR="analysis_results/saebench_comparison"

# Check if checkpoint exists
if [ ! -d "$OURS_CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found: $OURS_CHECKPOINT"
    echo "Please provide valid checkpoint path"
    exit 1
fi

# Check if Google's transcoder exists
if [ ! -d "$GOOGLE_DIR" ]; then
    echo "❌ Error: Google transcoder directory not found: $GOOGLE_DIR"
    echo "Please download Google's Gemma Scope transcoders first"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Matryoshka Checkpoint: $OURS_CHECKPOINT"
echo "  Google Directory: $GOOGLE_DIR"
echo "  Layer: $LAYER"
echo "  Batches: $BATCHES"
echo "  Output: $OUTPUT_DIR"
echo ""

# Run evaluation
python src/eval/eval_interpretability_saebench.py \
    --ours_checkpoint "$OURS_CHECKPOINT" \
    --google_dir "$GOOGLE_DIR" \
    --layer $LAYER \
    --dataset HuggingFaceFW/fineweb-edu \
    --batches $BATCHES \
    --batch_size 256 \
    --seq_len 64 \
    --device cuda:0 \
    --dtype bfloat16 \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=================================="
echo "Evaluation Complete!"
echo "=================================="
echo "Results saved to: $OUTPUT_DIR/saebench_comparison.json"
echo ""
echo "Key metrics to check:"
echo "  1. Absorption Score (ours only): < 0.01 = excellent"
echo "  2. FVU (both): < 0.1 = excellent reconstruction"
echo "  3. Dead Features (ours only): < 10% = excellent utilization"

