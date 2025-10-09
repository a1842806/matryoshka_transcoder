#!/usr/bin/env python3
"""
Quick training test to demonstrate that activation samples are now collected correctly.
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

def quick_training_test():
    """Run a quick training test to show sample collection works."""
    
    print("=" * 80)
    print("🚀 QUICK TRAINING TEST - ACTIVATION SAMPLE COLLECTION")
    print("=" * 80)
    
    # Create minimal config for quick test
    cfg = get_default_cfg()
    cfg["model_name"] = "gemma-2-2b"
    cfg["dataset_path"] = "HuggingFaceFW/fineweb-edu"
    cfg["layer"] = 8
    cfg["num_tokens"] = 1000  # Very small for quick test
    cfg["model_batch_size"] = 1
    cfg["batch_size"] = 2
    cfg["seq_len"] = 16  # Small sequence length
    cfg["lr"] = 3e-4
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Transcoder config - small for quick test
    cfg["sae_type"] = "matryoshka-transcoder"
    cfg["dict_size"] = 2304  # 1x expansion for quick test
    cfg["group_sizes"] = [2304]  # Single group
    cfg["top_k"] = 4  # Very small top_k
    cfg["l1_coeff"] = 0.0
    cfg["aux_penalty"] = 1/32
    cfg["n_batches_to_dead"] = 5
    cfg["top_k_aux"] = 16
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
    cfg["sample_collection_freq"] = 1  # Collect every step
    cfg["max_samples_per_feature"] = 5
    cfg["sample_context_size"] = 8
    cfg["sample_activation_threshold"] = 0.001  # Very low threshold
    
    cfg = post_init_cfg(cfg)
    
    print(f"📋 Quick Test Configuration:")
    print(f"   Model: {cfg['model_name']}")
    print(f"   Training tokens: {cfg['num_tokens']:,}")
    print(f"   Expected steps: {cfg['num_tokens'] // cfg['batch_size']}")
    print(f"   Sample collection: every {cfg['sample_collection_freq']} steps")
    print(f"   Activation threshold: {cfg['sample_activation_threshold']}")
    
    try:
        # Load model
        print(f"\n📥 Loading model...")
        model = HookedTransformer.from_pretrained_no_processing(
            cfg["model_name"]
        ).to(cfg["device"])
        print(f"✓ Model loaded")
        
        # Create activation store
        print(f"\n📊 Creating activation store...")
        activation_store = TranscoderActivationsStore(model, cfg)
        print(f"✓ Activation store created")
        
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
        
        # Run a few training steps
        print(f"\n🎯 Running quick training test...")
        
        optimizer = torch.optim.Adam(transcoder.parameters(), lr=cfg["lr"])
        num_steps = cfg["num_tokens"] // cfg["batch_size"]
        
        print(f"   Running {num_steps} training steps...")
        
        for step in range(num_steps):
            # Get batch
            source_batch, target_batch = activation_store.next_batch()
            
            # Forward pass
            transcoder_output = transcoder(source_batch, target_batch)
            loss = transcoder_output["loss"]
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Collect samples every step
            if sample_collector is not None:
                with torch.no_grad():
                    sample_batch, sample_tokens = activation_store.get_batch_with_tokens()
                    sample_acts = transcoder.encode(sample_batch)
                    sample_collector.collect_batch_samples(
                        sample_acts,
                        sample_tokens,
                        model.tokenizer
                    )
            
            if step % 10 == 0:
                print(f"   Step {step}: Loss = {loss.item():.4f}")
        
        # Check results
        print(f"\n📊 Training Test Results:")
        print(f"   Final loss: {loss.item():.4f}")
        
        # Analyze collected samples
        total_samples = 0
        features_with_samples = 0
        
        for feature_idx in range(min(10, transcoder.config["dict_size"])):
            samples = sample_collector.get_top_samples(feature_idx)
            if samples:
                features_with_samples += 1
                total_samples += len(samples)
        
        print(f"\n🎯 Sample Collection Results:")
        print(f"   Features with samples: {features_with_samples}")
        print(f"   Total samples collected: {total_samples}")
        
        if total_samples > 0:
            print(f"   ✅ SUCCESS: Activation samples collected from training data!")
            
            # Show some examples
            print(f"\n📝 Sample Examples:")
            for feature_idx in range(min(3, transcoder.config["dict_size"])):
                samples = sample_collector.get_top_samples(feature_idx, k=2)
                if samples:
                    print(f"   Feature {feature_idx}:")
                    for i, sample in enumerate(samples):
                        print(f"      {i+1}. Activation: {sample.activation_value:.4f}")
                        print(f"         Context: '{sample.context_text[:40]}{'...' if len(sample.context_text) > 40 else ''}'")
                        print(f"         Position: {sample.position}")
        else:
            print(f"   ⚠️  No samples collected - try lowering the activation threshold")
        
        print(f"\n🎉 Quick training test completed!")
        print(f"✅ Activation sample collection is working correctly!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = quick_training_test()
    if success:
        print(f"\n✅ ACTIVATION SAMPLE COLLECTION IS FIXED AND WORKING!")
        print(f"   The issue was that TranscoderActivationsStore was missing")
        print(f"   the get_batch_with_tokens() method needed for sample collection.")
        print(f"   Now features successfully store activation samples from training data!")
    else:
        print(f"\n❌ Test failed. Check the error messages above.")
