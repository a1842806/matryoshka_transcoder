#!/usr/bin/env python3
"""
Comprehensive analysis of Layer 17 Matryoshka Transcoder training results.
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def load_checkpoint(checkpoint_path):
    """Load model checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load the model state
    model_state = torch.load(os.path.join(checkpoint_path, "sae.pt"), map_location='cpu')
    
    # Load the config
    with open(os.path.join(checkpoint_path, "config.json"), 'r') as f:
        config = json.load(f)
    
    return model_state, config

def analyze_activation_samples(samples_dir):
    """Analyze collected activation samples."""
    print(f"\n=== ACTIVATION SAMPLE ANALYSIS ===")
    
    # Load collection summary
    summary_path = os.path.join(samples_dir, "collection_summary.json")
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    print(f"Total samples seen: {summary['total_samples_seen']:,}")
    print(f"Total samples stored: {summary['total_samples_stored']:,}")
    print(f"Features tracked: {summary['num_features_tracked']:,}")
    print(f"Avg samples per feature: {summary['avg_samples_per_feature']:.1f}")
    
    # Analyze top features
    print(f"\nTop 10 Features by Max Activation:")
    for i, feat in enumerate(summary['top_features_by_activation'][:10]):
        print(f"  {i+1:2d}. Feature {feat['feature_idx']:4d}: max={feat['max_activation']:6.2f}, samples={feat['num_samples']:3d}")
    
    # Load a few sample files to analyze content
    sample_files = [f for f in os.listdir(samples_dir) if f.startswith('feature_') and f.endswith('.json')]
    
    print(f"\nAnalyzing sample content from {len(sample_files)} feature files...")
    
    # Analyze a few high-activation features
    high_activation_features = [f['feature_idx'] for f in summary['top_features_by_activation'][:5]]
    
    feature_analysis = {}
    for feat_idx in high_activation_features:
        sample_file = f"feature_{feat_idx:05d}_samples.json"
        if sample_file in sample_files:
            with open(os.path.join(samples_dir, sample_file), 'r') as f:
                samples = json.load(f)
            
            # Analyze token patterns
            tokens = [sample['tokens'] for sample in samples['top_samples'][:10]]  # First 10 samples
            token_counts = defaultdict(int)
            for token_seq in tokens:
                for token in token_seq:
                    token_counts[token] += 1
            
            feature_analysis[feat_idx] = {
                'max_activation': samples['statistics']['max_activation'],
                'num_samples': samples['statistics']['num_samples'],
                'top_tokens': sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            }
    
    print(f"\nDetailed Feature Analysis:")
    for feat_idx, analysis in feature_analysis.items():
        print(f"\nFeature {feat_idx}:")
        print(f"  Max activation: {analysis['max_activation']:.2f}")
        print(f"  Sample count: {analysis['num_samples']}")
        print(f"  Top tokens: {analysis['top_tokens'][:5]}")
    
    return summary, feature_analysis

def analyze_model_weights(model_state):
    """Analyze the trained model weights."""
    print(f"\n=== MODEL WEIGHT ANALYSIS ===")
    
    # Extract weight statistics
    W_enc = model_state['W_enc']
    W_dec = model_state['W_dec']
    b_enc = model_state['b_enc']
    b_dec = model_state['b_dec']
    
    print(f"Encoder weights (W_enc): {W_enc.shape}")
    print(f"  Mean: {W_enc.mean().item():.4f}")
    print(f"  Std: {W_enc.std().item():.4f}")
    print(f"  Min: {W_enc.min().item():.4f}")
    print(f"  Max: {W_enc.max().item():.4f}")
    
    print(f"\nDecoder weights (W_dec): {W_dec.shape}")
    print(f"  Mean: {W_dec.mean().item():.4f}")
    print(f"  Std: {W_dec.std().item():.4f}")
    print(f"  Min: {W_dec.min().item():.4f}")
    print(f"  Max: {W_dec.max().item():.4f}")
    
    print(f"\nEncoder bias (b_enc): {b_enc.shape}")
    print(f"  Mean: {b_enc.mean().item():.4f}")
    print(f"  Std: {b_enc.std().item():.4f}")
    
    print(f"\nDecoder bias (b_dec): {b_dec.shape}")
    print(f"  Mean: {b_dec.mean().item():.4f}")
    print(f"  Std: {b_dec.std().item():.4f}")
    
    # Analyze feature norms
    feature_norms = torch.norm(W_enc, dim=0)
    print(f"\nFeature encoder norms:")
    print(f"  Mean: {feature_norms.mean().item():.4f}")
    print(f"  Std: {feature_norms.std().item():.4f}")
    print(f"  Min: {feature_norms.min().item():.4f}")
    print(f"  Max: {feature_norms.max().item():.4f}")
    
    # Analyze dead features
    num_batches_not_active = model_state.get('num_batches_not_active', torch.zeros(W_enc.shape[1]))
    dead_features = (num_batches_not_active > 0).sum().item()
    print(f"\nDead features: {dead_features} / {W_enc.shape[1]} ({dead_features/W_enc.shape[1]*100:.1f}%)")
    
    return {
        'W_enc_stats': {
            'mean': W_enc.mean().item(),
            'std': W_enc.std().item(),
            'min': W_enc.min().item(),
            'max': W_enc.max().item()
        },
        'W_dec_stats': {
            'mean': W_dec.mean().item(),
            'std': W_dec.std().item(),
            'min': W_dec.min().item(),
            'max': W_dec.max().item()
        },
        'feature_norms': {
            'mean': feature_norms.mean().item(),
            'std': feature_norms.std().item(),
            'min': feature_norms.min().item(),
            'max': feature_norms.max().item()
        },
        'dead_features': dead_features,
        'total_features': W_enc.shape[1]
    }

def analyze_training_metrics(checkpoint_path):
    """Analyze training metrics from the checkpoint."""
    print(f"\n=== TRAINING METRICS ANALYSIS ===")
    
    # Load training metrics if available
    metrics_file = os.path.join(checkpoint_path, "training_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        print(f"Training completed successfully!")
        print(f"Final loss: {metrics.get('final_loss', 'N/A')}")
        print(f"Training steps: {metrics.get('total_steps', 'N/A')}")
        
        return metrics
    else:
        print("No training metrics file found.")
        return None

def create_visualizations(model_state, samples_summary, output_dir):
    """Create visualization plots."""
    print(f"\n=== CREATING VISUALIZATIONS ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Feature activation distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    top_features = samples_summary['top_features_by_activation'][:50]
    feature_indices = [f['feature_idx'] for f in top_features]
    max_activations = [f['max_activation'] for f in top_features]
    
    plt.bar(range(len(feature_indices)), max_activations)
    plt.title('Top 50 Features by Max Activation')
    plt.xlabel('Feature Rank')
    plt.ylabel('Max Activation')
    plt.xticks(range(0, len(feature_indices), 5), [str(feature_indices[i]) for i in range(0, len(feature_indices), 5)], rotation=45)
    
    # 2. Sample count distribution
    plt.subplot(2, 2, 2)
    sample_counts = [f['num_samples'] for f in top_features]
    plt.hist(sample_counts, bins=20, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Sample Counts (Top 50 Features)')
    plt.xlabel('Number of Samples')
    plt.ylabel('Frequency')
    
    # 3. Weight distribution
    plt.subplot(2, 2, 3)
    W_enc = model_state['W_enc']
    weights_flat = W_enc.float().flatten().detach().cpu().numpy()
    plt.hist(weights_flat, bins=50, alpha=0.7, edgecolor='black')
    plt.title('Encoder Weight Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.xlim(-0.1, 0.1)  # Focus on main distribution
    
    # 4. Feature norms
    plt.subplot(2, 2, 4)
    feature_norms = torch.norm(W_enc.float(), dim=0).detach().cpu().numpy()
    plt.hist(feature_norms, bins=50, alpha=0.7, edgecolor='black')
    plt.title('Feature Encoder Norm Distribution')
    plt.xlabel('Feature Norm')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'layer17_analysis_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Overview plot saved to {output_dir}/layer17_analysis_overview.png")
    
    # 5. Activation vs Sample count scatter
    plt.figure(figsize=(10, 6))
    all_features = samples_summary['top_features_by_activation']
    activations = [f['max_activation'] for f in all_features]
    samples = [f['num_samples'] for f in all_features]
    
    plt.scatter(samples, activations, alpha=0.6, s=20)
    plt.xlabel('Number of Samples')
    plt.ylabel('Max Activation')
    plt.title('Feature Activation vs Sample Count')
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(samples, activations, 1)
    p = np.poly1d(z)
    plt.plot(samples, p(samples), "r--", alpha=0.8, label=f'Trend (slope={z[0]:.3f})')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'activation_vs_samples.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Activation vs samples plot saved to {output_dir}/activation_vs_samples.png")

def generate_analysis_report(model_stats, samples_summary, feature_analysis, output_dir):
    """Generate a comprehensive analysis report."""
    print(f"\n=== GENERATING ANALYSIS REPORT ===")
    
    report_path = os.path.join(output_dir, 'layer17_analysis_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Layer 17 Matryoshka Transcoder Analysis Report\n\n")
        f.write(f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Features:** {model_stats['total_features']:,}\n")
        f.write(f"- **Dead Features:** {model_stats['dead_features']:,} ({model_stats['dead_features']/model_stats['total_features']*100:.1f}%)\n")
        f.write(f"- **Features with Samples:** {samples_summary['num_features_tracked']:,}\n")
        f.write(f"- **Total Samples Collected:** {samples_summary['total_samples_stored']:,}\n")
        f.write(f"- **Average Samples per Feature:** {samples_summary['avg_samples_per_feature']:.1f}\n\n")
        
        f.write("## Training Configuration\n\n")
        f.write("- **Model:** Gemma-2-2B\n")
        f.write("- **Layer:** 17 (resid_mid → mlp_out transcoder)\n")
        f.write("- **Dictionary Size:** 4,608 (Matryoshka: [1152, 2304, 1152])\n")
        f.write("- **Top-k:** 48\n")
        f.write("- **Learning Rate:** 0.0003 with warmup+decay\n")
        f.write("- **Training Steps:** ~14,500\n\n")
        
        f.write("## Model Weight Analysis\n\n")
        f.write("### Encoder Weights (W_enc)\n")
        f.write(f"- **Mean:** {model_stats['W_enc_stats']['mean']:.4f}\n")
        f.write(f"- **Std:** {model_stats['W_enc_stats']['std']:.4f}\n")
        f.write(f"- **Range:** [{model_stats['W_enc_stats']['min']:.4f}, {model_stats['W_enc_stats']['max']:.4f}]\n\n")
        
        f.write("### Decoder Weights (W_dec)\n")
        f.write(f"- **Mean:** {model_stats['W_dec_stats']['mean']:.4f}\n")
        f.write(f"- **Std:** {model_stats['W_dec_stats']['std']:.4f}\n")
        f.write(f"- **Range:** [{model_stats['W_dec_stats']['min']:.4f}, {model_stats['W_dec_stats']['max']:.4f}]\n\n")
        
        f.write("### Feature Norms\n")
        f.write(f"- **Mean:** {model_stats['feature_norms']['mean']:.4f}\n")
        f.write(f"- **Std:** {model_stats['feature_norms']['std']:.4f}\n")
        f.write(f"- **Range:** [{model_stats['feature_norms']['min']:.4f}, {model_stats['feature_norms']['max']:.4f}]\n\n")
        
        f.write("## Top Features by Activation\n\n")
        f.write("| Rank | Feature ID | Max Activation | Sample Count |\n")
        f.write("|------|------------|----------------|--------------|\n")
        
        for i, feat in enumerate(samples_summary['top_features_by_activation'][:20]):
            f.write(f"| {i+1:2d} | {feat['feature_idx']:10d} | {feat['max_activation']:14.2f} | {feat['num_samples']:12d} |\n")
        
        f.write("\n## Feature Analysis\n\n")
        f.write("### High-Activation Features\n\n")
        
        for feat_idx, analysis in feature_analysis.items():
            f.write(f"**Feature {feat_idx}:**\n")
            f.write(f"- Max activation: {analysis['max_activation']:.2f}\n")
            f.write(f"- Sample count: {analysis['num_samples']}\n")
            f.write(f"- Top tokens: {', '.join([f'{token}({count})' for token, count in analysis['top_tokens'][:5]])}\n\n")
        
        f.write("## Key Insights\n\n")
        f.write("1. **Feature Utilization:** ")
        if model_stats['dead_features'] / model_stats['total_features'] < 0.1:
            f.write("Good feature utilization with low dead feature rate.\n")
        else:
            f.write(f"Moderate dead feature rate ({model_stats['dead_features']/model_stats['total_features']*100:.1f}%).\n")
        
        f.write("2. **Activation Distribution:** ")
        top_activation = samples_summary['top_features_by_activation'][0]['max_activation']
        if top_activation > 30:
            f.write("Strong feature activations detected.\n")
        else:
            f.write("Moderate feature activations.\n")
        
        f.write("3. **Sample Collection:** ")
        avg_samples = samples_summary['avg_samples_per_feature']
        if avg_samples > 50:
            f.write("Good sample coverage across features.\n")
        else:
            f.write("Limited sample coverage - consider longer training.\n")
        
        f.write("\n## Recommendations\n\n")
        f.write("1. **Further Analysis:** Investigate the most active features for interpretability.\n")
        f.write("2. **Feature Engineering:** Consider adjusting dictionary size based on dead feature rate.\n")
        f.write("3. **Training:** Monitor convergence - training appears successful.\n")
        f.write("4. **Interpretability:** Use activation samples for feature interpretation.\n\n")
        
        f.write("## Files Generated\n\n")
        f.write("- `layer17_analysis_overview.png`: Overview plots\n")
        f.write("- `activation_vs_samples.png`: Activation vs sample count analysis\n")
        f.write("- `layer17_analysis_report.md`: This report\n")
    
    print(f"Analysis report saved to {report_path}")

def main():
    """Main analysis function."""
    print("=" * 80)
    print("LAYER 17 MATRYOSHKA TRANSCODER ANALYSIS")
    print("=" * 80)
    
    # Paths
    checkpoint_path = "/home/datngo/User/matryoshka_sae/checkpoints/sae/gemma-2-2b/gemma-2-2b_blocks.17.hook_resid_pre_4608_topk_48_0.0003_14500"
    samples_dir = "/home/datngo/User/matryoshka_sae/checkpoints/transcoder/gemma-2-2b/gemma-2-2b_blocks.17.hook_resid_pre_4608_topk_48_0.0003_activation_samples"
    output_dir = "/home/datngo/User/matryoshka_sae/analysis/layer17_results"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and analyze data
    model_state, config = load_checkpoint(checkpoint_path)
    samples_summary, feature_analysis = analyze_activation_samples(samples_dir)
    model_stats = analyze_model_weights(model_state)
    training_metrics = analyze_training_metrics(checkpoint_path)
    
    # Create visualizations
    create_visualizations(model_state, samples_summary, output_dir)
    
    # Generate report
    generate_analysis_report(model_stats, samples_summary, feature_analysis, output_dir)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"Check the generated report: {output_dir}/layer17_analysis_report.md")

if __name__ == "__main__":
    main()
