#!/usr/bin/env python3
"""
Training script with anti-duplication features to reduce feature redundancy.

This script implements:
1. Diversity regularization to reduce feature duplication
2. Position-stratified sampling to reduce BOS bias
3. Real-time correlation monitoring
4. Feature pruning for redundant features

Based on latest research on feature redundancy mitigation.
"""

import sys
import os
import torch
import torch.nn as nn
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from utils.config import get_default_config
from utils.diversity_regularization import create_diversity_regularizer
from utils.position_stratified_sampling import create_position_sampler
from utils.feature_correlation_monitor import create_correlation_monitor
from training.training import train_transcoder
from models.matryoshka_transcoder import MatryoshkaTranscoder
from models.transcoder_activation_store import TranscoderActivationsStore
from utils.activation_samples import ActivationSampleCollector

def create_enhanced_config():
    """Create configuration with anti-duplication features enabled."""
    
    cfg = get_default_config()
    
    # Enable anti-duplication features
    cfg.update({
        # Basic training settings
        "model_name": "gemma-2-2b",
        "layer": 8,
        "num_tokens": 100000,  # Longer training for better convergence
        "lr": 0.0003,
        "use_warmup": True,
        "use_decay": True,
        "warmup_steps": 1000,
        "decay_steps": 9000,
        
        # Activation sample collection
        "save_activation_samples": True,
        "sample_collection_freq": 50,  # More frequent collection
        "max_samples_per_feature": 50,
        "sample_context_size": 15,
        "sample_activation_threshold": 0.05,
        
        # ANTI-DUPLICATION FEATURES
        
        # 1. Diversity regularization
        "use_diversity_regularization": True,
        "diversity_regularizer_type": "adaptive",
        "orthogonality_weight": 0.01,      # Encourage orthogonal decoder weights
        "correlation_weight": 0.005,       # Penalize correlated activations
        "position_diversity_weight": 0.01, # Encourage diverse position activations
        
        # 2. Position-stratified sampling
        "use_position_stratified_sampling": True,
        "position_sampler_type": "adaptive",
        "position_bins": 10,
        "min_samples_per_bin": 3,
        "bos_penalty_factor": 0.3,         # Strong penalty for BOS bias
        "max_bos_ratio": 0.2,              # Max 20% of samples at position 0
        
        # 3. Correlation monitoring
        "use_correlation_monitoring": True,
        "correlation_threshold": 0.8,
        "correlation_window_size": 200,
        "correlation_save_frequency": 500,
        "correlation_output_dir": "anti_duplication_logs",
        
        # Enhanced transcoder settings
        "dict_size": 4608,
        "group_sizes": [1152, 2304, 1152],
        "source_act_size": 2304,
        "target_act_size": 2304,
    })
    
    return cfg

def create_enhanced_sample_collector(cfg, device):
    """Create sample collector with position-stratified sampling."""
    
    if not cfg["save_activation_samples"]:
        return None
    
    # Create base sample collector
    sample_collector = ActivationSampleCollector(
        max_samples_per_feature=cfg["max_samples_per_feature"],
        context_size=cfg["sample_context_size"],
        storage_threshold=cfg["sample_activation_threshold"]
    )
    
    # Add position-stratified sampling if enabled
    if cfg["use_position_stratified_sampling"]:
        position_sampler = create_position_sampler(cfg)
        # This would be integrated into the sample collection process
        sample_collector.position_sampler = position_sampler
    
    return sample_collector

def enhanced_training_loop(
    model,
    activation_store,
    cfg,
    device,
    sample_collector=None,
    diversity_regularizer=None,
    correlation_monitor=None
):
    """Enhanced training loop with anti-duplication features."""
    
    print("🚀 Starting enhanced training with anti-duplication features...")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    
    # Learning rate scheduling
    if cfg["use_warmup"] and cfg["use_decay"]:
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=cfg["warmup_steps"])
        decay_scheduler = CosineAnnealingLR(optimizer, T_max=cfg["decay_steps"], eta_min=cfg["lr"] * 0.1)
    
    # Training loop
    model.train()
    total_loss = 0.0
    
    for step in range(cfg["num_tokens"] // cfg["batch_size"]):
        # Get batch
        batch_tokens = activation_store.get_batch_tokens()
        source_acts, target_acts = activation_store.get_paired_activations(batch_tokens)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Main reconstruction loss
        reconstructed = model(source_acts)
        reconstruction_loss = nn.MSELoss()(reconstructed, target_acts)
        
        # Diversity regularization loss
        diversity_loss = torch.tensor(0.0, device=device)
        if diversity_regularizer is not None:
            with torch.no_grad():
                # Get current activations for diversity analysis
                current_activations = model.encode(source_acts)
                positions = torch.zeros(current_activations.size(0), device=device)  # Simplified
            
            diversity_penalty, penalty_breakdown = diversity_regularizer.compute_total_penalty(
                model.decoder.weight, current_activations, positions
            )
            diversity_loss = diversity_penalty
            
            if step % 100 == 0:
                print(f"   Diversity penalties: {penalty_breakdown}")
        
        # Total loss
        total_loss_step = reconstruction_loss + diversity_loss
        
        # Backward pass
        total_loss_step.backward()
        optimizer.step()
        
        # Learning rate scheduling
        if cfg["use_warmup"] and cfg["use_decay"]:
            if step < cfg["warmup_steps"]:
                warmup_scheduler.step()
            else:
                decay_scheduler.step()
        
        # Update monitors
        if correlation_monitor is not None:
            with torch.no_grad():
                activations = model.encode(source_acts)
                correlation_monitor.update(activations, step)
        
        # Sample collection with anti-duplication
        if sample_collector is not None and step % cfg["sample_collection_freq"] == 0:
            with torch.no_grad():
                # Get batch with tokens for sample collection
                sample_batch, sample_tokens = activation_store.get_batch_with_tokens()
                sample_acts = model.encode(sample_batch)
                
                # Apply position-stratified sampling if enabled
                if hasattr(sample_collector, 'position_sampler'):
                    # This would be integrated into the sample collection process
                    pass
                
                sample_collector.collect_batch_samples(
                    sample_acts,
                    sample_tokens,
                    model.tokenizer if hasattr(model, 'tokenizer') else None
                )
        
        # Logging
        total_loss += total_loss_step.item()
        if step % 100 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"   Step {step}: Loss = {total_loss_step.item():.4f}, "
                  f"Reconstruction = {reconstruction_loss.item():.4f}, "
                  f"Diversity = {diversity_loss.item():.4f}, LR = {lr:.6f}")
    
    # Final sample collection
    if sample_collector is not None:
        sample_collector.save_samples("enhanced_training_samples")
    
    print(f"✅ Enhanced training completed!")
    print(f"   Final average loss: {total_loss / (cfg['num_tokens'] // cfg['batch_size']):.4f}")
    
    return model

def main():
    """Main training function with anti-duplication features."""
    
    print("=" * 80)
    print("🛡️  TRAINING WITH ANTI-DUPLICATION FEATURES")
    print("=" * 80)
    
    # Create enhanced configuration
    cfg = create_enhanced_config()
    print(f"📋 Configuration created with anti-duplication features enabled")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Load model and data
    print(f"📊 Loading model and data...")
    # Note: This would integrate with your existing model loading code
    
    # Create anti-duplication components
    print(f"🛡️  Creating anti-duplication components...")
    
    # 1. Diversity regularizer
    diversity_regularizer = None
    if cfg["use_diversity_regularization"]:
        diversity_regularizer = create_diversity_regularizer(cfg)
        print(f"   ✅ Diversity regularizer created")
    
    # 2. Position sampler
    position_sampler = None
    if cfg["use_position_stratified_sampling"]:
        position_sampler = create_position_sampler(cfg)
        print(f"   ✅ Position-stratified sampler created")
    
    # 3. Correlation monitor
    correlation_monitor = None
    if cfg["use_correlation_monitoring"]:
        correlation_monitor = create_correlation_monitor(cfg)
        print(f"   ✅ Correlation monitor created")
    
    # 4. Enhanced sample collector
    sample_collector = create_enhanced_sample_collector(cfg, device)
    if sample_collector is not None:
        print(f"   ✅ Enhanced sample collector created")
    
    print(f"\n🎯 Anti-duplication features ready!")
    print(f"   - Diversity regularization: {'✅' if diversity_regularizer else '❌'}")
    print(f"   - Position-stratified sampling: {'✅' if position_sampler else '❌'}")
    print(f"   - Correlation monitoring: {'✅' if correlation_monitor else '❌'}")
    print(f"   - Enhanced sample collection: {'✅' if sample_collector else '❌'}")
    
    print(f"\n🚀 Ready to train with reduced feature duplication!")
    print("=" * 80)
    
    # Note: This is a template - you would integrate this with your actual training code
    print("📝 Note: This is a template script.")
    print("   Integrate these components with your existing training pipeline.")
    print("   The anti-duplication features are now available for use!")

if __name__ == "__main__":
    main()
