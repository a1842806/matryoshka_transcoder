#!/usr/bin/env python3
"""
Analyze actual model weights to detect feature duplication.

This script loads the trained model and examines the actual feature weights
to determine if features are truly duplicating each other.
"""

import torch
import numpy as np
import json
import os
from pathlib import Path

def load_model_weights(checkpoint_path):
    """Load model weights from checkpoint."""
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ Loaded checkpoint from: {checkpoint_path}")
        return checkpoint
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None

def analyze_encoder_weights(encoder_weights, suspicious_features):
    """Analyze encoder weights for feature duplication."""
    
    print("=" * 80)
    print("🔍 ENCODER WEIGHT ANALYSIS")
    print("=" * 80)
    
    # Get weight matrix (assuming it's a linear layer)
    if 'weight' in encoder_weights:
        weight_matrix = encoder_weights['weight']
        print(f"📊 Weight matrix shape: {weight_matrix.shape}")
        
        # Extract weights for suspicious features
        feature_weights = {}
        for feature_idx in suspicious_features:
            if feature_idx < weight_matrix.shape[0]:
                feature_weights[feature_idx] = weight_matrix[feature_idx].detach().numpy()
                print(f"   Feature {feature_idx} weight vector shape: {feature_weights[feature_idx].shape}")
        
        # Calculate cosine similarity between feature weights
        print(f"\n🔗 Feature Weight Similarity (Cosine Similarity):")
        
        similarities = {}
        feature_list = list(feature_weights.keys())
        
        for i, feat1 in enumerate(feature_list):
            for j, feat2 in enumerate(feature_list):
                if i < j:
                    w1 = feature_weights[feat1]
                    w2 = feature_weights[feat2]
                    
                    # Calculate cosine similarity
                    dot_product = np.dot(w1, w2)
                    norm1 = np.linalg.norm(w1)
                    norm2 = np.linalg.norm(w2)
                    
                    if norm1 > 0 and norm2 > 0:
                        cosine_sim = dot_product / (norm1 * norm2)
                        similarities[(feat1, feat2)] = cosine_sim
                        
                        print(f"   Features {feat1} & {feat2}: {cosine_sim:.6f}")
                        
                        if cosine_sim > 0.95:
                            print(f"      🚨 EXTREMELY HIGH SIMILARITY - Likely duplication!")
                        elif cosine_sim > 0.9:
                            print(f"      ⚠️  VERY HIGH SIMILARITY - Possible duplication!")
                        elif cosine_sim > 0.8:
                            print(f"      ⚠️  HIGH SIMILARITY - Check for redundancy")
        
        # Analyze weight magnitudes
        print(f"\n📊 Weight Magnitude Analysis:")
        for feature_idx, weights in feature_weights.items():
            weight_norm = np.linalg.norm(weights)
            weight_mean = np.mean(np.abs(weights))
            weight_std = np.std(weights)
            
            print(f"   Feature {feature_idx}:")
            print(f"      L2 norm: {weight_norm:.6f}")
            print(f"      Mean absolute: {weight_mean:.6f}")
            print(f"      Std: {weight_std:.6f}")
        
        return similarities
    
    else:
        print("❌ No 'weight' key found in encoder weights")
        return {}

def analyze_decoder_weights(decoder_weights, suspicious_features):
    """Analyze decoder weights for feature duplication."""
    
    print("\n" + "=" * 80)
    print("🔍 DECODER WEIGHT ANALYSIS")
    print("=" * 80)
    
    if 'weight' in decoder_weights:
        weight_matrix = decoder_weights['weight']
        print(f"📊 Decoder weight matrix shape: {weight_matrix.shape}")
        
        # For decoder, we look at columns (features) instead of rows
        feature_weights = {}
        for feature_idx in suspicious_features:
            if feature_idx < weight_matrix.shape[1]:  # Columns for decoder
                feature_weights[feature_idx] = weight_matrix[:, feature_idx].detach().numpy()
                print(f"   Feature {feature_idx} decoder weight vector shape: {feature_weights[feature_idx].shape}")
        
        # Calculate cosine similarity between decoder feature weights
        print(f"\n🔗 Decoder Feature Weight Similarity:")
        
        feature_list = list(feature_weights.keys())
        
        for i, feat1 in enumerate(feature_list):
            for j, feat2 in enumerate(feature_list):
                if i < j:
                    w1 = feature_weights[feat1]
                    w2 = feature_weights[feat2]
                    
                    # Calculate cosine similarity
                    dot_product = np.dot(w1, w2)
                    norm1 = np.linalg.norm(w1)
                    norm2 = np.linalg.norm(w2)
                    
                    if norm1 > 0 and norm2 > 0:
                        cosine_sim = dot_product / (norm1 * norm2)
                        
                        print(f"   Features {feat1} & {feat2}: {cosine_sim:.6f}")
                        
                        if cosine_sim > 0.95:
                            print(f"      🚨 EXTREMELY HIGH SIMILARITY - Likely duplication!")
                        elif cosine_sim > 0.9:
                            print(f"      ⚠️  VERY HIGH SIMILARITY - Possible duplication!")
        
        return feature_weights
    
    else:
        print("❌ No 'weight' key found in decoder weights")
        return {}

def find_latest_checkpoint():
    """Find the latest checkpoint file."""
    
    checkpoint_dir = Path("checkpoints/transcoder/gemma-2-2b")
    
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return None
    
    # Look for the most recent checkpoint
    checkpoint_files = list(checkpoint_dir.glob("**/sae.pt"))
    
    if not checkpoint_files:
        print(f"❌ No sae.pt files found in {checkpoint_dir}")
        return None
    
    # Get the most recently modified file
    latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
    print(f"📁 Found latest checkpoint: {latest_checkpoint}")
    
    return str(latest_checkpoint)

def main():
    """Main function to analyze model weights for duplication."""
    
    print("🔍 Analyzing model weights for feature duplication...")
    
    # Find latest checkpoint
    checkpoint_path = find_latest_checkpoint()
    if checkpoint_path is None:
        return
    
    # Load checkpoint
    checkpoint = load_model_weights(checkpoint_path)
    if checkpoint is None:
        return
    
    # Define suspicious features
    suspicious_features = [1936, 830, 2177, 2149]
    
    print(f"\n🎯 Analyzing weights for suspicious features: {suspicious_features}")
    
    # Analyze encoder weights
    if 'encoder' in checkpoint:
        encoder_similarities = analyze_encoder_weights(checkpoint['encoder'], suspicious_features)
    elif 'model' in checkpoint and 'encoder' in checkpoint['model']:
        encoder_similarities = analyze_encoder_weights(checkpoint['model']['encoder'], suspicious_features)
    else:
        print("❌ No encoder weights found in checkpoint")
        encoder_similarities = {}
    
    # Analyze decoder weights
    if 'decoder' in checkpoint:
        decoder_weights = analyze_decoder_weights(checkpoint['decoder'], suspicious_features)
    elif 'model' in checkpoint and 'decoder' in checkpoint['model']:
        decoder_weights = analyze_decoder_weights(checkpoint['model']['decoder'], suspicious_features)
    else:
        print("❌ No decoder weights found in checkpoint")
        decoder_weights = {}
    
    # Summary
    print("\n" + "=" * 80)
    print("🎉 WEIGHT ANALYSIS SUMMARY")
    print("=" * 80)
    
    if encoder_similarities:
        high_similarity_pairs = []
        for (feat1, feat2), sim in encoder_similarities.items():
            if sim > 0.9:
                high_similarity_pairs.append(((feat1, feat2), sim))
        
        if high_similarity_pairs:
            print(f"🚨 ENCODER DUPLICATION DETECTED:")
            for (feat1, feat2), sim in high_similarity_pairs:
                print(f"   Features {feat1} & {feat2}: {sim:.6f} similarity")
        else:
            print(f"✅ No high encoder similarity detected")
    
    print(f"\n💡 Interpretation:")
    print(f"   - Cosine similarity > 0.95: Almost identical weights (duplication)")
    print(f"   - Cosine similarity > 0.9: Very similar weights (possible duplication)")
    print(f"   - Cosine similarity > 0.8: Similar weights (check for redundancy)")
    print(f"   - Both encoder and decoder similarities should be considered")
    print("=" * 80)

if __name__ == "__main__":
    main()
