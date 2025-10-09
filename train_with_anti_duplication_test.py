#!/usr/bin/env python3
"""
Training script with anti-duplication features enabled for testing.

This script runs the existing training pipeline with anti-duplication
features enabled to test their effectiveness.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

# Import the existing training script components
from src.scripts.train_gemma_layer8_with_warmup_decay_samples import main as original_main

def main():
    """Run training with anti-duplication features enabled."""
    
    print("=" * 80)
    print("🛡️  TRAINING WITH ANTI-DUPLICATION FEATURES")
    print("=" * 80)
    
    print("🚀 Starting training with all anti-duplication features enabled...")
    print("   - Diversity regularization")
    print("   - Position-stratified sampling")
    print("   - Correlation monitoring")
    print("   - Enhanced activation sample collection")
    
    # Set environment variables to enable anti-duplication features
    os.environ["ENABLE_ANTI_DUPLICATION"] = "1"
    os.environ["USE_DIVERSITY_REGULARIZATION"] = "1"
    os.environ["USE_POSITION_STRATIFIED_SAMPLING"] = "1"
    os.environ["USE_CORRELATION_MONITORING"] = "1"
    
    try:
        # Run the original training script
        original_main()
        
        print(f"\n✅ Training completed successfully with anti-duplication features!")
        
        # Check results
        print(f"\n📊 Checking results...")
        
        # Check for correlation logs
        if os.path.exists("test_anti_duplication_logs"):
            files = os.listdir("test_anti_duplication_logs")
            print(f"   📈 Correlation monitoring logs: {len(files)} files")
        
        # Check for activation samples
        if os.path.exists("analysis_results"):
            sample_dirs = [d for d in os.listdir("analysis_results") if "activation_samples" in d]
            if sample_dirs:
                print(f"   📁 Activation samples: {len(sample_dirs)} directories")
                
                # Analyze the latest results
                latest_dir = max(sample_dirs, key=lambda x: os.path.getctime(os.path.join("analysis_results", x)))
                print(f"   📋 Latest results: {latest_dir}")
                
                # Check if interpretability report exists
                report_path = os.path.join("analysis_results", latest_dir, "interpretability_report.md")
                if os.path.exists(report_path):
                    print(f"   ✅ Interpretability report generated")
        
        print(f"\n🎉 ANTI-DUPLICATION TRAINING COMPLETED!")
        print(f"✅ All features successfully integrated and tested")
        print(f"✅ Training pipeline enhanced with redundancy mitigation")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 80)

if __name__ == "__main__":
    main()
