#!/bin/bash

# Learning Rate Scheduling Experiments for Gemma-2-2B Layer 12 Transcoder
# Compare warmup, decay, and combined strategies against standard training

echo "=" * 80
echo "Learning Rate Scheduling Experiments"
echo "=" * 80
echo ""
echo "This script runs 3 different learning rate strategies:"
echo "1. Warmup Only (GPU 0)"
echo "2. Cosine Decay Only (GPU 1)" 
echo "3. Warmup + Decay (future comparison)"
echo ""
echo "Your standard run should already be completed and logged to W&B."
echo ""

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Please ensure CUDA is available."
    exit 1
fi

echo "✓ CUDA is available"
echo "Available GPUs:"
nvidia-smi --list-gpus
echo ""

# Function to run experiment
run_experiment() {
    local script_name=$1
    local gpu_id=$2
    local experiment_name=$3
    
    echo "🚀 Starting $experiment_name on GPU $gpu_id..."
    echo "Script: $script_name"
    echo ""
    
    CUDA_VISIBLE_DEVICES=$gpu_id python src/scripts/$script_name
    
    if [ $? -eq 0 ]; then
        echo "✅ $experiment_name completed successfully!"
    else
        echo "❌ $experiment_name failed!"
        return 1
    fi
    echo ""
}

# Ask user which experiments to run
echo "Which experiments would you like to run?"
echo "1) Warmup Only (GPU 0)"
echo "2) Cosine Decay Only (GPU 1)"
echo "3) Both experiments in parallel"
echo "4) Warmup + Decay (combined)"
echo "5) All experiments"
echo ""
read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        run_experiment "train_gemma_layer12_transcoder_warmup.py" 0 "Warmup Only"
        ;;
    2)
        run_experiment "train_gemma_layer12_transcoder_decay.py" 1 "Cosine Decay Only"
        ;;
    3)
        echo "🚀 Running both experiments in parallel..."
        echo ""
        
        # Run warmup on GPU 0 in background
        echo "Starting Warmup experiment on GPU 0..."
        CUDA_VISIBLE_DEVICES=0 python src/scripts/train_gemma_layer12_transcoder_warmup.py &
        WARMUP_PID=$!
        
        # Run decay on GPU 1 in background  
        echo "Starting Decay experiment on GPU 1..."
        CUDA_VISIBLE_DEVICES=1 python src/scripts/train_gemma_layer12_transcoder_decay.py &
        DECAY_PID=$!
        
        echo "Both experiments started. PIDs: Warmup=$WARMUP_PID, Decay=$DECAY_PID"
        echo "Waiting for both to complete..."
        echo ""
        
        # Wait for both to complete
        wait $WARMUP_PID
        WARMUP_EXIT=$?
        wait $DECAY_PID
        DECAY_EXIT=$?
        
        if [ $WARMUP_EXIT -eq 0 ] && [ $DECAY_EXIT -eq 0 ]; then
            echo "✅ Both experiments completed successfully!"
        else
            echo "❌ One or both experiments failed!"
            echo "Warmup exit code: $WARMUP_EXIT"
            echo "Decay exit code: $DECAY_EXIT"
        fi
        ;;
    4)
        run_experiment "train_gemma_layer12_transcoder_warmup_decay.py" 0 "Warmup + Decay"
        ;;
    5)
        echo "🚀 Running all experiments sequentially..."
        echo ""
        
        run_experiment "train_gemma_layer12_transcoder_warmup.py" 0 "Warmup Only"
        run_experiment "train_gemma_layer12_transcoder_decay.py" 1 "Cosine Decay Only"
        run_experiment "train_gemma_layer12_transcoder_warmup_decay.py" 0 "Warmup + Decay"
        ;;
    *)
        echo "Invalid choice. Please run the script again and select 1-5."
        exit 1
        ;;
esac

echo ""
echo "=" * 80
echo "Experiments Summary"
echo "=" * 80
echo ""
echo "W&B Projects to compare:"
echo "1. Standard Run: gemma-2-2b-layer12-transcoder (your existing run)"
echo "2. Warmup Only: gemma-2-2b-layer12-transcoder-warmup"
echo "3. Decay Only: gemma-2-2b-layer12-transcoder-decay"
echo "4. Warmup + Decay: gemma-2-2b-layer12-transcoder-warmup-decay"
echo ""
echo "Compare the following metrics on W&B:"
echo "- Training Loss curves"
echo "- Learning Rate schedules"
echo "- FVU (Fraction of Variance Unexplained)"
echo "- L0 (Sparsity)"
echo "- Final performance metrics"
echo ""
echo "Expected outcomes:"
echo "- Warmup: More stable early training, potentially better convergence"
echo "- Decay: Better final performance, reduced overfitting"
echo "- Combined: Best of both worlds"
echo ""
echo "Happy analyzing! 🎉"
