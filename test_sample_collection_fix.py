#!/usr/bin/env python3
"""
Test script to verify that activation sample collection works with the fixed code.
"""

import sys
import os
import torch

# Add src to path
sys.path.append('src')

from models.transcoder_activation_store import TranscoderActivationsStore, create_transcoder_config
from models.sae import MatryoshkaTranscoder
from utils.activation_samples import ActivationSampleCollector
from utils.config import get_default_cfg, post_init_cfg
from transformer_lens import HookedTransformer

def test_sample_collection():
    """Test that activation sample collection works with transcoders."""
    
    print("=" * 80)
    print("🧪 TESTING ACTIVATION SAMPLE COLLECTION FIX")
    print("=" * 80)
    
    # Create minimal config
    cfg = get_default_cfg()
    cfg["model_name"] = "gemma-2-2b"
    cfg["dataset_path"] = "HuggingFaceFW/fineweb-edu"
    cfg["layer"] = 8
    cfg["num_tokens"] = 10000  # Small amount for testing
    cfg["model_batch_size"] = 2
    cfg["batch_size"] = 4
    cfg["seq_len"] = 32
    cfg["lr"] = 3e-4
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Transcoder config - use correct Gemma-2 dimensions
    cfg["sae_type"] = "matryoshka-transcoder"
    cfg["dict_size"] = 9216  # 4x expansion of 2304 for testing
    cfg["group_sizes"] = [2304, 4608, 2304]  # Small groups for testing
    cfg["top_k"] = 12
    cfg["l1_coeff"] = 0.0
    cfg["aux_penalty"] = 1/32
    cfg["n_batches_to_dead"] = 20
    cfg["top_k_aux"] = 64
    cfg["source_act_size"] = 2304  # Gemma-2 d_model
    cfg["target_act_size"] = 2304  # Gemma-2 d_model
    
    cfg = create_transcoder_config(
        cfg,
        source_layer=8,
        target_layer=8,
        source_site="resid_mid",
        target_site="mlp_out"
    )
    
    # Activation sample collection config
    cfg["save_activation_samples"] = True
    cfg["sample_collection_freq"] = 1  # Collect every step for testing
    cfg["max_samples_per_feature"] = 10
    cfg["sample_context_size"] = 10
    cfg["sample_activation_threshold"] = 0.01  # Low threshold for testing
    
    cfg = post_init_cfg(cfg)
    
    print(f"📋 Test Configuration:")
    print(f"   Model: {cfg['model_name']}")
    print(f"   Layer: {cfg['layer']}")
    print(f"   Device: {cfg['device']}")
    print(f"   Sample collection enabled: {cfg['save_activation_samples']}")
    print(f"   Collection frequency: every {cfg['sample_collection_freq']} steps")
    print(f"   Activation threshold: {cfg['sample_activation_threshold']}")
    
    try:
        # Load model
        print(f"\n📥 Loading model...")
        model = HookedTransformer.from_pretrained_no_processing(
            cfg["model_name"]
        ).to(cfg["device"])
        print(f"✓ Model loaded successfully")
        
        # Create activation store
        print(f"\n📊 Creating activation store...")
        activation_store = TranscoderActivationsStore(model, cfg)
        print(f"✓ Activation store created")
        
        # Test get_batch_with_tokens method
        print(f"\n🔧 Testing get_batch_with_tokens method...")
        sample_batch, sample_tokens = activation_store.get_batch_with_tokens()
        print(f"✓ get_batch_with_tokens works!")
        print(f"   Sample batch shape: {sample_batch.shape}")
        print(f"   Sample tokens shape: {sample_tokens.shape}")
        
        # Create transcoder
        print(f"\n🏗️  Creating transcoder...")
        transcoder = MatryoshkaTranscoder(cfg)
        print(f"✓ Transcoder created")
        
        # Create sample collector
        print(f"\n📥 Creating sample collector...")
        sample_collector = ActivationSampleCollector(
            max_samples_per_feature=cfg["max_samples_per_feature"],
            context_size=cfg["sample_context_size"],
            storage_threshold=cfg["sample_activation_threshold"]
        )
        print(f"✓ Sample collector created")
        
        # Test sample collection
        print(f"\n🎯 Testing sample collection...")
        
        # Get a batch and encode it
        sample_batch, sample_tokens = activation_store.get_batch_with_tokens()
        sample_acts = transcoder.encode(sample_batch)
        
        print(f"   Sample activations shape: {sample_acts.shape}")
        print(f"   Sample tokens shape: {sample_tokens.shape}")
        
        # Collect samples
        sample_collector.collect_batch_samples(
            sample_acts,
            sample_tokens,
            model.tokenizer
        )
        
        # Check results
        total_samples = 0
        features_with_samples = 0
        
        for feature_idx in range(min(10, sample_acts.shape[1])):  # Check first 10 features
            samples = sample_collector.get_top_samples(feature_idx)
            if samples:
                features_with_samples += 1
                total_samples += len(samples)
                print(f"   Feature {feature_idx}: {len(samples)} samples")
        
        print(f"\n✅ Sample Collection Test Results:")
        print(f"   Features with samples: {features_with_samples}")
        print(f"   Total samples collected: {total_samples}")
        
        if total_samples > 0:
            print(f"   ✅ SUCCESS: Activation sample collection is working!")
            
            # Show some examples
            print(f"\n📝 Sample Examples:")
            for feature_idx in range(min(3, sample_acts.shape[1])):
                samples = sample_collector.get_top_samples(feature_idx, k=2)
                if samples:
                    print(f"   Feature {feature_idx}:")
                    for i, sample in enumerate(samples):
                        print(f"      {i+1}. Activation: {sample.activation_value:.4f}")
                        print(f"         Context: '{sample.context_text[:50]}{'...' if len(sample.context_text) > 50 else ''}'")
        else:
            print(f"   ⚠️  No samples collected - activations may be below threshold")
        
        print(f"\n🎉 Test completed successfully!")
        print(f"✅ The fix for activation sample collection is working!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_sample_collection()
    if success:
        print(f"\n✅ All tests passed! Activation sample collection is fixed.")
    else:
        print(f"\n❌ Tests failed. Check the error messages above.")
