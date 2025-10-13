#!/usr/bin/env python3
"""
Analyze actual feature weights to detect duplication.

This script loads the trained SAE weights and examines the actual feature weights
to determine if features are truly duplicating each other.
"""

import torch
import numpy as np
import json
import os
from pathlib import Path

def load_sae_weights(checkpoint_path):
    """Load SAE weights from checkpoint."""
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ Loaded SAE checkpoint from: {checkpoint_path}")
        return checkpoint
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None

def analyze_encoder_weights(W_enc, suspicious_features):
    """Analyze encoder weights for feature duplication."""
    
    print("=" * 80)
    print("🔍 ENCODER WEIGHT ANALYSIS")
    print("=" * 80)
    
    print(f"📊 Encoder weight matrix shape: {W_enc.shape}")
    print(f"   - Input dimension: {W_enc.shape[0]}")
    print(f"   - Feature dimension: {W_enc.shape[1]}")
    
    # Extract weights for suspicious features
    feature_weights = {}
    for feature_idx in suspicious_features:
        if feature_idx < W_enc.shape[1]:
            # For encoder, each feature is a column
            feature_weights[feature_idx] = W_enc[:, feature_idx].float().numpy()
            print(f"   Feature {feature_idx} encoder weight vector shape: {feature_weights[feature_idx].shape}")
        else:
            print(f"   ⚠️  Feature {feature_idx} index out of range (max: {W_enc.shape[1]-1})")
    
    if not feature_weights:
        print("❌ No valid features found")
        return {}
    
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
        weight_max = np.max(np.abs(weights))
        
        print(f"   Feature {feature_idx}:")
        print(f"      L2 norm: {weight_norm:.6f}")
        print(f"      Mean absolute: {weight_mean:.6f}")
        print(f"      Std: {weight_std:.6f}")
        print(f"      Max absolute: {weight_max:.6f}")
    
    return similarities

def analyze_decoder_weights(W_dec, suspicious_features):
    """Analyze decoder weights for feature duplication."""
    
    print("\n" + "=" * 80)
    print("🔍 DECODER WEIGHT ANALYSIS")
    print("=" * 80)
    
    print(f"📊 Decoder weight matrix shape: {W_dec.shape}")
    print(f"   - Feature dimension: {W_dec.shape[0]}")
    print(f"   - Output dimension: {W_dec.shape[1]}")
    
    # Extract weights for suspicious features
    feature_weights = {}
    for feature_idx in suspicious_features:
        if feature_idx < W_dec.shape[0]:
            # For decoder, each feature is a row
            feature_weights[feature_idx] = W_dec[feature_idx, :].float().numpy()
            print(f"   Feature {feature_idx} decoder weight vector shape: {feature_weights[feature_idx].shape}")
        else:
            print(f"   ⚠️  Feature {feature_idx} index out of range (max: {W_dec.shape[0]-1})")
    
    if not feature_weights:
        print("❌ No valid features found")
        return {}
    
    # Calculate cosine similarity between decoder feature weights
    print(f"\n🔗 Decoder Feature Weight Similarity:")
    
    feature_list = list(feature_weights.keys())
    decoder_similarities = {}
    
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
                    decoder_similarities[(feat1, feat2)] = cosine_sim
                    
                    print(f"   Features {feat1} & {feat2}: {cosine_sim:.6f}")
                    
                    if cosine_sim > 0.95:
                        print(f"      🚨 EXTREMELY HIGH SIMILARITY - Likely duplication!")
                    elif cosine_sim > 0.9:
                        print(f"      ⚠️  VERY HIGH SIMILARITY - Possible duplication!")
    
    # Analyze decoder weight magnitudes
    print(f"\n📊 Decoder Weight Magnitude Analysis:")
    for feature_idx, weights in feature_weights.items():
        weight_norm = np.linalg.norm(weights)
        weight_mean = np.mean(np.abs(weights))
        weight_std = np.std(weights)
        weight_max = np.max(np.abs(weights))
        
        print(f"   Feature {feature_idx}:")
        print(f"      L2 norm: {weight_norm:.6f}")
        print(f"      Mean absolute: {weight_mean:.6f}")
        print(f"      Std: {weight_std:.6f}")
        print(f"      Max absolute: {weight_max:.6f}")
    
    return decoder_similarities

def analyze_weight_statistics(W_enc, W_dec, suspicious_features):
    """Analyze overall weight statistics."""
    
    print("\n" + "=" * 80)
    print("📊 OVERALL WEIGHT STATISTICS")
    print("=" * 80)
    
    # Encoder statistics
    print(f"🔍 Encoder Weight Statistics:")
    print(f"   Shape: {W_enc.shape}")
    print(f"   Mean: {W_enc.float().mean():.6f}")
    print(f"   Std: {W_enc.float().std():.6f}")
    print(f"   Min: {W_enc.float().min():.6f}")
    print(f"   Max: {W_enc.float().max():.6f}")
    
    # Decoder statistics
    print(f"\n🔍 Decoder Weight Statistics:")
    print(f"   Shape: {W_dec.shape}")
    print(f"   Mean: {W_dec.float().mean():.6f}")
    print(f"   Std: {W_dec.float().std():.6f}")
    print(f"   Min: {W_dec.float().min():.6f}")
    print(f"   Max: {W_dec.float().max():.6f}")
    
    # Check if suspicious features have unusual weight patterns
    print(f"\n🎯 Suspicious Feature Weight Analysis:")
    
    for feature_idx in suspicious_features:
        if feature_idx < W_enc.shape[1] and feature_idx < W_dec.shape[0]:
            enc_weights = W_enc[:, feature_idx].float().numpy()
            dec_weights = W_dec[feature_idx, :].float().numpy()
            
            print(f"\n   Feature {feature_idx}:")
            print(f"      Encoder - Mean: {np.mean(np.abs(enc_weights)):.6f}, Std: {np.std(enc_weights):.6f}")
            print(f"      Decoder - Mean: {np.mean(np.abs(dec_weights)):.6f}, Std: {np.std(dec_weights):.6f}")

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
    """Main function to analyze feature weights for duplication."""
    
    print("🔍 Analyzing SAE feature weights for duplication...")
    
    # Find latest checkpoint
    checkpoint_path = find_latest_checkpoint()
    if checkpoint_path is None:
        return
    
    # Load checkpoint
    checkpoint = load_sae_weights(checkpoint_path)
    if checkpoint is None:
        return
    
    # Extract weights
    W_enc = checkpoint['W_enc']  # [input_dim, feature_dim]
    W_dec = checkpoint['W_dec']  # [feature_dim, output_dim]
    
    print(f"📊 SAE Architecture:")
    print(f"   Input dimension: {W_enc.shape[0]}")
    print(f"   Feature dimension: {W_enc.shape[1]}")
    print(f"   Output dimension: {W_dec.shape[1]}")
    
    # Define suspicious features
    suspicious_features = [1936, 830, 2177, 2149]
    
    print(f"\n🎯 Analyzing weights for suspicious features: {suspicious_features}")
    
    # Analyze encoder weights
    encoder_similarities = analyze_encoder_weights(W_enc, suspicious_features)
    
    # Analyze decoder weights
    decoder_similarities = analyze_decoder_weights(W_dec, suspicious_features)
    
    # Analyze overall statistics
    analyze_weight_statistics(W_enc, W_dec, suspicious_features)
    
    # Summary
    print("\n" + "=" * 80)
    print("🎉 WEIGHT ANALYSIS SUMMARY")
    print("=" * 80)
    
    print(f"🔍 Encoder Weight Similarities:")
    high_encoder_similarity = False
    if encoder_similarities:
        for (feat1, feat2), sim in encoder_similarities.items():
            if sim > 0.9:
                high_encoder_similarity = True
                print(f"   🚨 Features {feat1} & {feat2}: {sim:.6f} similarity")
            else:
                print(f"   ✅ Features {feat1} & {feat2}: {sim:.6f} similarity")
    
    print(f"\n🔍 Decoder Weight Similarities:")
    high_decoder_similarity = False
    if decoder_similarities:
        for (feat1, feat2), sim in decoder_similarities.items():
            if sim > 0.9:
                high_decoder_similarity = True
                print(f"   🚨 Features {feat1} & {feat2}: {sim:.6f} similarity")
            else:
                print(f"   ✅ Features {feat1} & {feat2}: {sim:.6f} similarity")
    
    # Final conclusion
    print(f"\n🎯 FINAL CONCLUSION:")
    if high_encoder_similarity and high_decoder_similarity:
        print(f"   🚨 FEATURE DUPLICATION CONFIRMED!")
        print(f"      - Both encoder and decoder weights show high similarity")
        print(f"      - Features are likely duplicating each other")
        print(f"      - Consider feature regularization or pruning")
    elif high_encoder_similarity or high_decoder_similarity:
        print(f"   ⚠️  PARTIAL DUPLICATION DETECTED!")
        print(f"      - Some weight components show high similarity")
        print(f"      - Features may be partially redundant")
    else:
        print(f"   ✅ NO WEIGHT DUPLICATION DETECTED!")
        print(f"      - Features have distinct weight patterns")
        print(f"      - Similar activations may be due to similar input patterns")
    
    print(f"\n💡 Interpretation:")
    print(f"   - Cosine similarity > 0.95: Almost identical weights (duplication)")
    print(f"   - Cosine similarity > 0.9: Very similar weights (possible duplication)")
    print(f"   - Cosine similarity > 0.8: Similar weights (check for redundancy)")
    print(f"   - Both encoder and decoder similarities should be considered")
    print("=" * 80)

if __name__ == "__main__":
    main()
