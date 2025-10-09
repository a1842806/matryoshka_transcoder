#!/usr/bin/env python3
"""
Simple test for anti-duplication features using existing training infrastructure.

This script tests the anti-duplication components by integrating them
into the existing training pipeline.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

from utils.config import get_default_cfg, post_init_cfg, create_gemma_mlp_transcoder_config
from utils.diversity_regularization import create_diversity_regularizer
from utils.position_stratified_sampling import create_position_sampler
from utils.feature_correlation_monitor import create_correlation_monitor
from src.training.training import train_transcoder

def create_anti_duplication_config():
    """Create configuration with anti-duplication features enabled."""
    
    # Get base config
    cfg = get_default_cfg()
    cfg = create_gemma_mlp_transcoder_config(cfg, layer=8)
    cfg = post_init_cfg(cfg)
    
    # Enable anti-duplication features
    cfg.update({
        # Shorter training for testing
        "num_tokens": 5000,
        "sample_collection_freq": 25,
        
        # ANTI-DUPLICATION FEATURES (ENABLED FOR TESTING)
        
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
        "correlation_output_dir": "test_anti_duplication_logs",
        
        # Enable activation sample collection
        "save_activation_samples": True,
        "max_samples_per_feature": 20,
        "sample_context_size": 15,
        "sample_activation_threshold": 0.05,
    })
    
    return cfg

def test_anti_duplication_components():
    """Test individual anti-duplication components."""
    
    print("🧪 Testing individual anti-duplication components...")
    
    cfg = create_anti_duplication_config()
    
    # Test 1: Diversity regularizer
    try:
        diversity_regularizer = create_diversity_regularizer(cfg)
        print("✅ Diversity regularizer created successfully")
    except Exception as e:
        print(f"❌ Diversity regularizer failed: {e}")
        return False
    
    # Test 2: Position sampler
    try:
        position_sampler = create_position_sampler(cfg)
        print("✅ Position-stratified sampler created successfully")
    except Exception as e:
        print(f"❌ Position sampler failed: {e}")
        return False
    
    # Test 3: Correlation monitor
    try:
        correlation_monitor = create_correlation_monitor(cfg)
        print("✅ Correlation monitor created successfully")
    except Exception as e:
        print(f"❌ Correlation monitor failed: {e}")
        return False
    
    print("✅ All anti-duplication components created successfully!")
    return True

def main():
    """Main test function."""
    
    print("=" * 80)
    print("🧪 TESTING ANTI-DUPLICATION COMPONENTS")
    print("=" * 80)
    
    # Test individual components
    if not test_anti_duplication_components():
        print("❌ Component tests failed")
        return
    
    # Create config for training
    cfg = create_anti_duplication_config()
    print(f"📋 Anti-duplication config created")
    print(f"   - Diversity regularization: {cfg.get('use_diversity_regularization', False)}")
    print(f"   - Position-stratified sampling: {cfg.get('use_position_stratified_sampling', False)}")
    print(f"   - Correlation monitoring: {cfg.get('use_correlation_monitoring', False)}")
    print(f"   - Activation sample collection: {cfg.get('save_activation_samples', False)}")
    
    print(f"\n🚀 Starting training with anti-duplication features...")
    print(f"   Training tokens: {cfg['num_tokens']}")
    print(f"   Sample collection frequency: {cfg['sample_collection_freq']}")
    
    try:
        # Run training with anti-duplication features
        train_transcoder(cfg)
        print(f"\n✅ Training completed successfully!")
        
        # Check if correlation logs were created
        correlation_dir = cfg.get("correlation_output_dir", "test_anti_duplication_logs")
        if os.path.exists(correlation_dir):
            files = os.listdir(correlation_dir)
            print(f"📊 Correlation logs created: {len(files)} files")
            for file in files[:3]:  # Show first 3 files
                print(f"   - {file}")
        
        # Check if activation samples were created
        if os.path.exists("analysis_results"):
            sample_dirs = [d for d in os.listdir("analysis_results") if "activation_samples" in d]
            if sample_dirs:
                print(f"📁 Activation samples created: {len(sample_dirs)} directories")
                for sample_dir in sample_dirs[:2]:  # Show first 2
                    print(f"   - {sample_dir}")
        
        print(f"\n🎉 ANTI-DUPLICATION TEST COMPLETED SUCCESSFULLY!")
        print(f"✅ All components integrated and working")
        print(f"✅ Training completed with anti-duplication features")
        print(f"✅ Check logs and samples for effectiveness analysis")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 80)

if __name__ == "__main__":
    main()
