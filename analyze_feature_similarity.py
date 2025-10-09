#!/usr/bin/env python3
"""
Analyze feature similarity to detect potential feature duplication.

This script examines features that have similar activation patterns to determine
if they are truly duplicating each other or if they have subtle differences.
"""

import json
import os
import numpy as np
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def load_analysis_results():
    """Load the analysis results from the JSON file."""
    
    analysis_file = "analysis_results/gemma-2-2b_blocks.8.hook_resid_pre_4608_matryoshka-transcoder_8_0.0003_activation_samples/analysis_results.json"
    
    if not os.path.exists(analysis_file):
        print(f"❌ Analysis file not found: {analysis_file}")
        return None
    
    with open(analysis_file, 'r') as f:
        return json.load(f)

def extract_contexts_for_feature(feature_data):
    """Extract all contexts for a given feature."""
    contexts = []
    for example in feature_data['top_examples']:
        contexts.append(example['context'])
    return contexts

def clean_context(context):
    """Clean and normalize context text for comparison."""
    # Remove <bos> tokens and normalize whitespace
    cleaned = context.replace('<bos>', '').strip()
    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned

def calculate_context_similarity(contexts1, contexts2):
    """Calculate similarity between two sets of contexts."""
    
    # Clean contexts
    clean1 = [clean_context(ctx) for ctx in contexts1]
    clean2 = [clean_context(ctx) for ctx in contexts2]
    
    # Combine all contexts for TF-IDF
    all_contexts = clean1 + clean2
    
    if not all_contexts:
        return 0.0
    
    # Create TF-IDF vectors
    try:
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(all_contexts)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Get similarity between the two feature sets
        n1 = len(clean1)
        n2 = len(clean2)
        
        # Average similarity between contexts from feature1 and feature2
        cross_similarities = []
        for i in range(n1):
            for j in range(n1, n1 + n2):
                cross_similarities.append(similarity_matrix[i][j])
        
        return np.mean(cross_similarities) if cross_similarities else 0.0
        
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0.0

def analyze_feature_duplication(analysis_results):
    """Analyze potential feature duplication."""
    
    print("=" * 80)
    print("🔍 FEATURE DUPLICATION ANALYSIS")
    print("=" * 80)
    
    # Focus on the suspicious features mentioned
    suspicious_features = [1936, 830, 2177, 2149]
    
    print(f"🎯 Analyzing suspicious features: {suspicious_features}")
    
    # Create feature lookup
    feature_lookup = {}
    for feature in analysis_results['top_features']:
        feature_lookup[feature['feature_idx']] = feature
    
    # Check if all suspicious features exist
    missing_features = [f for f in suspicious_features if f not in feature_lookup]
    if missing_features:
        print(f"❌ Missing features: {missing_features}")
        return
    
    print(f"\n📊 Feature Details:")
    for feature_idx in suspicious_features:
        feature = feature_lookup[feature_idx]
        print(f"   Feature {feature_idx}:")
        print(f"      Max activation: {feature['max_activation']:.4f}")
        print(f"      Mean activation: {feature['mean_activation']:.4f}")
        print(f"      Sample count: {feature['sample_count']}")
        
        # Show contexts
        contexts = extract_contexts_for_feature(feature)
        print(f"      Contexts:")
        for i, ctx in enumerate(contexts[:3], 1):
            print(f"         {i}. '{ctx}'")
    
    # Calculate pairwise similarities
    print(f"\n🔗 Pairwise Similarity Analysis:")
    
    similarities = {}
    for i, feat1_idx in enumerate(suspicious_features):
        for j, feat2_idx in enumerate(suspicious_features):
            if i < j:  # Avoid duplicate comparisons
                feat1 = feature_lookup[feat1_idx]
                feat2 = feature_lookup[feat2_idx]
                
                contexts1 = extract_contexts_for_feature(feat1)
                contexts2 = extract_contexts_for_feature(feat2)
                
                similarity = calculate_context_similarity(contexts1, contexts2)
                similarities[(feat1_idx, feat2_idx)] = similarity
                
                print(f"   Feature {feat1_idx} vs Feature {feat2_idx}: {similarity:.4f}")
    
    # Analyze similarity patterns
    print(f"\n🧠 Similarity Analysis:")
    
    high_similarity_pairs = []
    for (feat1, feat2), sim in similarities.items():
        if sim > 0.7:  # High similarity threshold
            high_similarity_pairs.append(((feat1, feat2), sim))
    
    if high_similarity_pairs:
        print(f"   🚨 HIGH SIMILARITY DETECTED:")
        for (feat1, feat2), sim in high_similarity_pairs:
            print(f"      Features {feat1} & {feat2}: {sim:.4f} similarity")
            print(f"         → Potential duplication!")
    else:
        print(f"   ✅ No high similarity detected (all < 0.7)")
    
    # Detailed context analysis
    print(f"\n📝 Detailed Context Analysis:")
    
    for feature_idx in suspicious_features:
        feature = feature_lookup[feature_idx]
        contexts = extract_contexts_for_feature(feature)
        
        print(f"\n   Feature {feature_idx} Context Patterns:")
        
        # Clean contexts
        clean_contexts = [clean_context(ctx) for ctx in contexts]
        
        # Find common patterns
        words = []
        for ctx in clean_contexts:
            words.extend(ctx.lower().split())
        
        word_freq = Counter(words)
        common_words = word_freq.most_common(5)
        
        print(f"      Most common words:")
        for word, count in common_words:
            print(f"         '{word}': {count} times")
        
        # Check for exact duplicates
        unique_contexts = set(clean_contexts)
        if len(unique_contexts) < len(clean_contexts):
            print(f"      ⚠️  {len(clean_contexts) - len(unique_contexts)} duplicate contexts found")
        
        print(f"      Unique contexts: {len(unique_contexts)}/{len(clean_contexts)}")

def analyze_activation_patterns(analysis_results):
    """Analyze activation patterns to detect duplication."""
    
    print(f"\n" + "=" * 80)
    print("⚡ ACTIVATION PATTERN ANALYSIS")
    print("=" * 80)
    
    # Focus on suspicious features
    suspicious_features = [1936, 830, 2177, 2149]
    
    feature_lookup = {}
    for feature in analysis_results['top_features']:
        feature_lookup[feature['feature_idx']] = feature
    
    print(f"🎯 Activation Pattern Comparison:")
    
    # Extract activation values for each feature
    feature_activations = {}
    for feature_idx in suspicious_features:
        if feature_idx in feature_lookup:
            feature = feature_lookup[feature_idx]
            activations = [ex['activation'] for ex in feature['top_examples']]
            feature_activations[feature_idx] = activations
    
    # Compare activation distributions
    print(f"\n📊 Activation Statistics:")
    for feature_idx, activations in feature_activations.items():
        print(f"   Feature {feature_idx}:")
        print(f"      Mean: {np.mean(activations):.4f}")
        print(f"      Std: {np.std(activations):.4f}")
        print(f"      Min: {np.min(activations):.4f}")
        print(f"      Max: {np.max(activations):.4f}")
        print(f"      Range: {np.max(activations) - np.min(activations):.4f}")
    
    # Check for identical activation patterns
    print(f"\n🔍 Activation Pattern Similarity:")
    
    # Calculate correlation between activation patterns
    feature_list = list(feature_activations.keys())
    for i, feat1 in enumerate(feature_list):
        for j, feat2 in enumerate(feature_list):
            if i < j:
                activations1 = feature_activations[feat1]
                activations2 = feature_activations[feat2]
                
                # Pad shorter list with zeros for comparison
                max_len = max(len(activations1), len(activations2))
                padded1 = activations1 + [0] * (max_len - len(activations1))
                padded2 = activations2 + [0] * (max_len - len(activations2))
                
                correlation = np.corrcoef(padded1, padded2)[0, 1]
                print(f"   Features {feat1} & {feat2}: correlation = {correlation:.4f}")
                
                if correlation > 0.9:
                    print(f"      🚨 VERY HIGH CORRELATION - Potential duplication!")

def analyze_position_patterns(analysis_results):
    """Analyze position patterns to detect duplication."""
    
    print(f"\n" + "=" * 80)
    print("📍 POSITION PATTERN ANALYSIS")
    print("=" * 80)
    
    suspicious_features = [1936, 830, 2177, 2149]
    
    feature_lookup = {}
    for feature in analysis_results['top_features']:
        feature_lookup[feature['feature_idx']] = feature
    
    print(f"🎯 Position Pattern Comparison:")
    
    for feature_idx in suspicious_features:
        if feature_idx in feature_lookup:
            feature = feature_lookup[feature_idx]
            positions = [ex['position'] for ex in feature['top_examples']]
            
            print(f"\n   Feature {feature_idx}:")
            print(f"      Positions: {positions}")
            
            # Count position frequency
            position_counts = Counter(positions)
            print(f"      Position frequency:")
            for pos, count in sorted(position_counts.items()):
                print(f"         Position {pos}: {count} times")

def check_feature_weights():
    """Check if we can access actual feature weights to detect duplication."""
    
    print(f"\n" + "=" * 80)
    print("⚖️  FEATURE WEIGHT ANALYSIS")
    print("=" * 80)
    
    # Look for saved model files
    checkpoint_dir = "checkpoints/transcoder/gemma-2-2b"
    
    if os.path.exists(checkpoint_dir):
        print(f"📁 Checkpoint directory found: {checkpoint_dir}")
        
        # List checkpoint files
        for root, dirs, files in os.walk(checkpoint_dir):
            for file in files:
                if file.endswith('.pt') or file.endswith('.pth'):
                    print(f"   Found model file: {file}")
    else:
        print(f"❌ No checkpoint directory found")

def main():
    """Main function to analyze feature duplication."""
    
    print("🔍 Loading analysis results for feature duplication detection...")
    
    # Load analysis results
    analysis_results = load_analysis_results()
    if analysis_results is None:
        return
    
    # Run all analyses
    analyze_feature_duplication(analysis_results)
    analyze_activation_patterns(analysis_results)
    analyze_position_patterns(analysis_results)
    check_feature_weights()
    
    print(f"\n" + "=" * 80)
    print("🎉 FEATURE DUPLICATION ANALYSIS COMPLETE!")
    print("=" * 80)
    
    print(f"\n💡 Recommendations:")
    print(f"   1. If high similarity detected (>0.7), features may be duplicating")
    print(f"   2. If correlation >0.9, features have very similar activation patterns")
    print(f"   3. Check if features activate at same positions consistently")
    print(f"   4. Consider feature regularization or pruning for duplicated features")
    print(f"   5. Analyze actual model weights to confirm duplication")
    print("=" * 80)

if __name__ == "__main__":
    main()
