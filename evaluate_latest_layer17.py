#!/usr/bin/env python3
"""
Quick evaluation script for your latest Layer 17 model.

This script will:
1. Evaluate your latest layer 17 model using interpretability metrics
2. Compare it with Google's Gemma Scope transcoder
3. Generate comprehensive reports

Usage:
    python evaluate_latest_layer17.py
"""

import os
import sys
import subprocess
import argparse

def main():
    print("=" * 80)
    print("🔬 INTERPRETABILITY EVALUATION - LAYER 17")
    print("=" * 80)
    
    # Your latest layer 17 checkpoint
    checkpoint_path = "results/gemma_2_2b/layer17/2025-10-28_4k-steps"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        if os.path.exists("results/gemma_2_2b/layer17"):
            for item in os.listdir("results/gemma_2_2b/layer17"):
                print(f"  - {item}")
        return
    
    print(f"✅ Found checkpoint: {checkpoint_path}")
    
    # Check if Google's model is available
    google_dir = "./gemma_scope_2b_transcoders"
    if not os.path.exists(google_dir):
        print(f"\n⚠️  Google's model not found at: {google_dir}")
        print("To download Google's model, run:")
        print("huggingface-cli download google/gemma-scope-2b-pt-trans --local-dir ./gemma_scope_2b_transcoders")
        print("\nFor now, we'll only evaluate your model.")
        run_standalone_only = True
    else:
        print(f"✅ Found Google's model: {google_dir}")
        run_standalone_only = False
    
    # Step 1: Evaluate your model standalone
    print("\n" + "=" * 80)
    print("📊 STEP 1: EVALUATING YOUR MODEL")
    print("=" * 80)
    
    cmd_standalone = [
        "python", "src/eval/evaluate_interpretability_standalone.py",
        "--checkpoint", checkpoint_path,
        "--layer", "17",
        "--batches", "1000",
        "--device", "cuda:0" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu",
        "--output_dir", "analysis_results/layer17_evaluation"
    ]
    
    print("Running command:")
    print(" ".join(cmd_standalone))
    print()
    
    try:
        result = subprocess.run(cmd_standalone, check=True, capture_output=False)
        print("✅ Standalone evaluation completed!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Standalone evaluation failed: {e}")
        return
    
    # Step 2: Compare with Google (if available)
    if not run_standalone_only:
        print("\n" + "=" * 80)
        print("🏆 STEP 2: COMPARING WITH GOOGLE'S MODEL")
        print("=" * 80)
        
        cmd_compare = [
            "python", "src/eval/compare_interpretability.py",
            "--ours_checkpoint", checkpoint_path,
            "--google_dir", google_dir,
            "--layer", "17",
            "--batches", "1000",
            "--device", "cuda:0" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu",
            "--output_dir", "analysis_results/layer17_comparison"
        ]
        
        print("Running command:")
        print(" ".join(cmd_compare))
        print()
        
        try:
            result = subprocess.run(cmd_compare, check=True, capture_output=False)
            print("✅ Comparison completed!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Comparison failed: {e}")
            print("Continuing with standalone results...")
    
    # Step 3: Show results
    print("\n" + "=" * 80)
    print("📋 RESULTS SUMMARY")
    print("=" * 80)
    
    standalone_results = "analysis_results/layer17_evaluation/metrics.json"
    if os.path.exists(standalone_results):
        print(f"✅ Your model evaluation: {standalone_results}")
        
        # Try to show key metrics
        try:
            import json
            with open(standalone_results, 'r') as f:
                metrics = json.load(f)
            
            print("\n🎯 KEY METRICS:")
            print(f"  FVU: {metrics['reconstruction']['fvu']:.4f}")
            print(f"  Absorption Score: {metrics['absorption']['score']:.4f}")
            print(f"  Dead Features: {metrics['feature_utilization']['dead_percentage']:.1f}%")
            print(f"  Mean L0: {metrics['sparsity']['mean_l0']:.1f}")
            
        except Exception as e:
            print(f"Could not parse metrics: {e}")
    
    if not run_standalone_only:
        comparison_results = "analysis_results/layer17_comparison/comparison.json"
        if os.path.exists(comparison_results):
            print(f"✅ Comparison results: {comparison_results}")
    
    print("\n" + "=" * 80)
    print("🎉 EVALUATION COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review the metrics in the analysis_results/ directory")
    print("2. If FVU > 0.1, consider training with higher capacity")
    print("3. If absorption > 0.01, consider adjusting training parameters")
    print("4. Use the results to guide your next training run")

if __name__ == "__main__":
    main()
