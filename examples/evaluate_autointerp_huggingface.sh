#!/bin/bash

# Example: AutoInterp Evaluation with Hugging Face Models (FREE!)
#
# This demonstrates how to run AutoInterp evaluation using free
# Hugging Face models instead of the expensive GPT API.
#
# Models tested:
# - microsoft/DialoGPT-small (117M params, fast)
# - microsoft/DialoGPT-medium (345M params, balanced)
# - microsoft/DialoGPT-large (774M params, better quality)

echo "=========================================="
echo "AutoInterp Evaluation with Hugging Face"
echo "=========================================="

# Configuration
CHECKPOINT="checkpoints/transcoder/gemma-2-2b/my_model_15000"
LAYER=17
OUTPUT_DIR="analysis_results/autointerp_huggingface"

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
echo "  Backend: Hugging Face (FREE!)"
echo ""

# Test different Hugging Face models
MODELS=(
    "microsoft/DialoGPT-small"
    "microsoft/DialoGPT-medium" 
    "microsoft/DialoGPT-large"
)

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Testing model: $MODEL"
    echo "=========================================="
    
    # Create model-specific output directory
    MODEL_DIR="${OUTPUT_DIR}/$(basename $MODEL)"
    
    # Run evaluation
    python src/eval/eval_saebench_autointerp_absorption.py \
        --checkpoint "$CHECKPOINT" \
        --layer $LAYER \
        --device cuda:0 \
        --dtype bfloat16 \
        --output_dir "$MODEL_DIR" \
        --llm_backend huggingface \
        --llm_model "$MODEL"
    
    echo ""
    echo "✅ Results saved to: $MODEL_DIR"
done

echo ""
echo "=========================================="
echo "🎉 All evaluations complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Model comparison:"
echo "  - DialoGPT-small:  Fastest, lowest quality"
echo "  - DialoGPT-medium: Balanced speed/quality"
echo "  - DialoGPT-large:  Best quality, slowest"
echo ""
echo "💡 Tips:"
echo "  - Use DialoGPT-medium for most cases"
echo "  - Use DialoGPT-small for quick testing"
echo "  - Use DialoGPT-large for final evaluation"
echo ""
echo "🔧 To try other models:"
echo "  --llm_model facebook/blenderbot-400M-distill"
echo "  --llm_model microsoft/DialoGPT-xlarge"
echo "=========================================="
