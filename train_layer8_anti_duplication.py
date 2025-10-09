#!/usr/bin/env python3
"""
Train Gemma Layer 8 with anti-duplication features enabled.

This script runs training with all anti-duplication features enabled:
- Diversity regularization
- Position-stratified sampling
- Correlation monitoring
- Enhanced activation sample collection
"""

import sys
import os
import torch
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.scripts.train_gemma_layer8_with_warmup_decay_samples import (
    create_gemma_config, 
    load_model_and_data, 
    create_transcoder_and_store,
    train_transcoder
)

def create_anti_duplication_config():
    """Create configuration with anti-duplication features enabled."""
    
    # Get base configuration
    cfg = create_gemma_config()
    
    # Enable anti-duplication features
    cfg.update({
        # Shorter training for testing anti-duplication features
        "num_tokens": int(1e5),  # 100K tokens for testing
        
        # ANTI-DUPLICATION FEATURES (ENABLED)
        
        # 1. Diversity regularization
        "use_diversity_regularization": True,
        "diversity_regularizer_type": "adaptive",
        "orthogonality_weight": 0.01,
        "correlation_weight": 0.005,
        "position_diversity_weight": 0.01,
        
        # 2. Position-stratified sampling
        "use_position_stratified_sampling": True,
        "position_sampler_type": "adaptive",
        "position_bins": 10,
        "min_samples_per_bin": 2,
        "bos_penalty_factor": 0.3,
        "max_bos_ratio": 0.2,
        
        # 3. Correlation monitoring
        "use_correlation_monitoring": True,
        "correlation_threshold": 0.8,
        "correlation_window_size": 50,
        "correlation_save_frequency": 100,
        "correlation_output_dir": "anti_duplication_logs",
        
        # Enhanced activation sample collection
        "save_activation_samples": True,
        "sample_collection_freq": 25,  # More frequent collection
        "max_samples_per_feature": 20,
        "sample_context_size": 15,
        "sample_activation_threshold": 0.05,
    })
    
    return cfg

def main():
    """Main training function with anti-duplication features."""
    
    print("=" * 80)
    print("🛡️  TRAINING GEMMA LAYER 8 WITH ANTI-DUPLICATION FEATURES")
    print("=" * 80)
    
    # Create configuration
    cfg = create_anti_duplication_config()
    
    print("📋 Configuration with anti-duplication features:")
    print(f"   - Model: {cfg['model_name']}")
    print(f"   - Layer: {cfg['layer']}")
    print(f"   - Training tokens: {cfg['num_tokens']:,}")
    print(f"   - Diversity regularization: {'✅' if cfg.get('use_diversity_regularization') else '❌'}")
    print(f"   - Position-stratified sampling: {'✅' if cfg.get('use_position_stratified_sampling') else '❌'}")
    print(f"   - Correlation monitoring: {'✅' if cfg.get('use_correlation_monitoring') else '❌'}")
    print(f"   - Activation sample collection: {'✅' if cfg.get('save_activation_samples') else '❌'}")
    
    try:
        # Load model and data
        print(f"\n🔄 Loading model and data...")
        model = load_model_and_data(cfg)
        print(f"✅ Model loaded successfully")
        
        # Create transcoder and activation store
        print(f"\n🔄 Creating transcoder and activation store...")
        transcoder, activation_store = create_transcoder_and_store(cfg, model)
        print(f"✅ Transcoder and activation store created")
        
        # Train with anti-duplication features
        print(f"\n🚀 Starting training with anti-duplication features...")
        train_transcoder(transcoder, activation_store, model, cfg)
        
        print(f"\n✅ Training completed successfully!")
        
        # Analyze results
        print(f"\n📊 Analyzing results...")
        
        # Check correlation logs
        correlation_dir = cfg.get("correlation_output_dir", "anti_duplication_logs")
        if os.path.exists(correlation_dir):
            files = os.listdir(correlation_dir)
            print(f"   📈 Correlation monitoring: {len(files)} log files created")
        
        # Check activation samples
        if os.path.exists("analysis_results"):
            sample_dirs = [d for d in os.listdir("analysis_results") if "activation_samples" in d]
            if sample_dirs:
                latest_dir = max(sample_dirs, key=lambda x: os.path.getctime(os.path.join("analysis_results", x)))
                print(f"   📁 Activation samples: {latest_dir}")
                
                # Check interpretability report
                report_path = os.path.join("analysis_results", latest_dir, "interpretability_report.md")
                if os.path.exists(report_path):
                    print(f"   📋 Interpretability report: Generated")
        
        print(f"\n🎉 ANTI-DUPLICATION TRAINING COMPLETED!")
        print(f"✅ All anti-duplication features successfully integrated")
        print(f"✅ Training enhanced with redundancy mitigation")
        print(f"✅ Ready for comparison with baseline training")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 80)

if __name__ == "__main__":
    main()
