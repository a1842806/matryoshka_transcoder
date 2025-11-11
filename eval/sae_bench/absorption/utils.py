"""
Utility functions for absorption evaluation.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json


def get_token_first_letter(token_str: str) -> Optional[str]:
    """
    Extract the first letter from a token string.
    
    Args:
        token_str: Token string (may have special prefixes)
    
    Returns:
        First letter (lowercase) or None
    """
    # Clean up token string (remove spaces, special chars at start)
    token_clean = token_str.strip().lstrip("▁").lstrip("Ġ").lstrip()
    
    if len(token_clean) == 0:
        return None
    
    first_char = token_clean[0].lower()
    
    if first_char.isalpha():
        return first_char
    
    return None


def compute_cosine_similarity_matrix(
    vectors_a: torch.Tensor,
    vectors_b: torch.Tensor,
) -> torch.Tensor:
    """
    Compute pairwise cosine similarities between two sets of vectors.
    
    Args:
        vectors_a: (n, d) tensor
        vectors_b: (m, d) tensor
    
    Returns:
        (n, m) similarity matrix
    """
    # Normalize
    vectors_a_norm = torch.nn.functional.normalize(vectors_a, dim=1)
    vectors_b_norm = torch.nn.functional.normalize(vectors_b, dim=1)
    
    # Compute similarities
    similarities = torch.matmul(vectors_a_norm, vectors_b_norm.t())
    
    return similarities


def find_token_aligned_features(
    transcoder,
    token_id: int,
    tokenizer,
    layer: int,
    model,
    threshold: float = 0.1,
) -> List[int]:
    """
    Find features that are aligned with a specific token.
    
    Args:
        transcoder: Matryoshka transcoder
        token_id: Token ID
        tokenizer: Tokenizer
        layer: Layer number
        model: Language model
        threshold: Activation threshold
    
    Returns:
        List of feature indices that fire for this token
    """
    # Get model activations for this token
    token_tensor = torch.tensor([[token_id]], device=transcoder.device)
    
    with torch.no_grad():
        _, cache = model.run_with_cache(token_tensor, stop_at_layer=layer+1)
        hook_name = f"blocks.{layer}.hook_resid_pre"
        activations = cache[hook_name][0, 0, :]  # (d_model,)
        
        # Encode with transcoder
        features, _ = transcoder.compute_activations(activations.unsqueeze(0))
        features = features.squeeze(0)  # (n_features,)
    
    # Find active features
    active_indices = torch.where(features > threshold)[0].tolist()
    
    return active_indices


def load_results(results_path: str) -> Dict:
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def save_results(results: Dict, output_path: str):
    """Save results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def aggregate_layer_results(
    results_by_layer: Dict[int, Dict]
) -> Dict:
    """
    Aggregate results across multiple layers.
    
    Args:
        results_by_layer: Dictionary mapping layer to results
    
    Returns:
        Aggregated statistics
    """
    all_f1 = []
    all_precision = []
    all_recall = []
    all_absorption_rates = []
    
    for layer, results in results_by_layer.items():
        summary = results['summary']
        all_f1.append(summary['mean_f1'])
        all_precision.append(summary['mean_precision'])
        all_recall.append(summary['mean_recall'])
        
        if 'mean_absorption_rate' in summary:
            all_absorption_rates.append(summary['mean_absorption_rate'])
    
    aggregated = {
        'n_layers': len(results_by_layer),
        'layers': sorted(results_by_layer.keys()),
        'mean_f1': float(np.mean(all_f1)),
        'std_f1': float(np.std(all_f1)),
        'mean_precision': float(np.mean(all_precision)),
        'std_precision': float(np.std(all_precision)),
        'mean_recall': float(np.mean(all_recall)),
        'std_recall': float(np.std(all_recall)),
    }
    
    if all_absorption_rates:
        aggregated.update({
            'mean_absorption_rate': float(np.mean(all_absorption_rates)),
            'std_absorption_rate': float(np.std(all_absorption_rates)),
        })
    
    return aggregated


def print_results_table(results_by_layer: Dict[int, Dict]):
    """Print a formatted table of results across layers."""
    print("\n" + "="*100)
    print(f"{'Layer':<8} {'Mean F1':<12} {'Mean Prec':<12} {'Mean Recall':<12} {'Absorption Rate':<18}")
    print("="*100)
    
    for layer in sorted(results_by_layer.keys()):
        summary = results_by_layer[layer]['summary']
        f1 = summary['mean_f1']
        prec = summary['mean_precision']
        rec = summary['mean_recall']
        
        line = f"{layer:<8} {f1:<12.3f} {prec:<12.3f} {rec:<12.3f}"
        
        if 'mean_absorption_rate' in summary:
            abs_rate = summary['mean_absorption_rate']
            line += f" {abs_rate:<18.3f}"
        else:
            line += f" {'N/A':<18}"
        
        print(line)
    
    print("="*100)


def generate_report(
    results: Dict,
    output_file: str,
):
    """
    Generate a text report summarizing the evaluation results.
    
    Args:
        results: Evaluation results dictionary
        output_file: Path to output text file
    """
    lines = []
    
    lines.append("="*80)
    lines.append("Absorption Score Evaluation Report")
    lines.append("="*80)
    lines.append("")
    
    # Model info
    if 'transcoder_info' in results:
        info = results['transcoder_info']
        lines.append("Model Information:")
        lines.append(f"  Layer: {info['layer']}")
        lines.append(f"  Number of features: {info['n_features']}")
        lines.append(f"  Total features: {info['total_features']}")
        lines.append(f"  Prefix level: {info['prefix_level']}")
        lines.append("")
    
    # Summary statistics
    if 'summary' in results:
        summary = results['summary']
        lines.append("Summary Statistics:")
        lines.append(f"  Mean F1: {summary['mean_f1']:.3f} ± {summary['std_f1']:.3f}")
        lines.append(f"  Mean Precision: {summary['mean_precision']:.3f} ± {summary['std_precision']:.3f}")
        lines.append(f"  Mean Recall: {summary['mean_recall']:.3f} ± {summary['std_recall']:.3f}")
        
        if 'mean_absorption_rate' in summary:
            lines.append("")
            lines.append("Absorption Analysis:")
            lines.append(f"  Mean absorption rate: {summary['mean_absorption_rate']:.3f} ± {summary['std_absorption_rate']:.3f}")
            lines.append(f"  Max absorption rate: {summary['max_absorption_rate']:.3f}")
            lines.append(f"  Min absorption rate: {summary['min_absorption_rate']:.3f}")
        
        lines.append("")
    
    # Per-letter results
    if 'results_by_letter' in results:
        lines.append("Per-Letter Results:")
        lines.append("-"*80)
        
        for letter, letter_results in sorted(results['results_by_letter'].items()):
            lines.append(f"\nLetter '{letter}':")
            lines.append(f"  Feature index: {letter_results['feature_idx']}")
            
            cls_metrics = letter_results['classification']
            lines.append(f"  Classification metrics:")
            lines.append(f"    Precision: {cls_metrics['precision']:.3f}")
            lines.append(f"    Recall: {cls_metrics['recall']:.3f}")
            lines.append(f"    F1: {cls_metrics['f1']:.3f}")
            
            if 'absorption' in letter_results:
                abs_data = letter_results['absorption']
                lines.append(f"  Absorption analysis:")
                lines.append(f"    Absorption rate: {abs_data['absorption_rate']:.3f}")
                lines.append(f"    Absorbed tokens: {abs_data['num_absorbed_tokens']}/{abs_data['num_total_tokens']}")
                lines.append(f"    Number of absorbing features: {len(abs_data['absorbing_features'])}")
    
    lines.append("")
    lines.append("="*80)
    
    # Write to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Report saved to: {output_path}")


def compare_with_baseline(
    results: Dict,
    baseline_f1: float = 0.95,
) -> Dict:
    """
    Compare transcoder performance with baseline (linear probe).
    
    Args:
        results: Evaluation results
        baseline_f1: Expected F1 score for linear probe
    
    Returns:
        Comparison statistics
    """
    summary = results['summary']
    transcoder_f1 = summary['mean_f1']
    
    gap = baseline_f1 - transcoder_f1
    relative_gap = gap / baseline_f1
    
    comparison = {
        'baseline_f1': baseline_f1,
        'transcoder_f1': transcoder_f1,
        'absolute_gap': gap,
        'relative_gap': relative_gap,
        'relative_performance': transcoder_f1 / baseline_f1,
    }
    
    return comparison

