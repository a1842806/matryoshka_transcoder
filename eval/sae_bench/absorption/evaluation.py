"""
Main evaluation script for absorption score.

Runs complete absorption evaluation on Matryoshka transcoders.
"""

import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from dataclasses import asdict

from .config import AbsorptionConfig
from .first_letter_task import (
    FirstLetterTask,
    train_linear_probe,
    evaluate_feature_classification,
    select_feature_by_cosine_similarity,
    select_features_k_sparse,
)
from .absorption_metric import AbsorptionMetric, compute_absorption_score
from .matryoshka_wrapper import MatryoshkaTranscoderWrapper, load_matryoshka_transcoder


def run_absorption_evaluation(
    config: AbsorptionConfig,
    transcoder_wrapper: MatryoshkaTranscoderWrapper,
    save_results: bool = True,
) -> Dict:
    """
    Run full absorption evaluation on a Matryoshka transcoder.
    
    Args:
        config: Evaluation configuration
        transcoder_wrapper: Wrapped Matryoshka transcoder
        save_results: Whether to save results to disk
    
    Returns:
        Dictionary with evaluation results
    """
    print("="*80)
    print(f"Absorption Score Evaluation")
    print(f"Model: {config.model_name}, Layer: {transcoder_wrapper.layer}")
    print(f"Transcoder features: {transcoder_wrapper.n_features}")
    print("="*80)
    
    # Load model
    print("\n[1/6] Loading model...")
    model = HookedTransformer.from_pretrained_no_processing(
        config.model_name,
        dtype=getattr(torch, config.dtype),
    ).to(config.device)
    
    # Create first-letter task
    print("\n[2/6] Creating first-letter classification dataset...")
    task = FirstLetterTask(
        model=model,
        tokenizer=model.tokenizer,
        letters=config.letters,
        min_tokens_per_letter=config.min_tokens_per_letter,
        device=config.device,
    )
    
    dataset = task.create_dataset(num_samples=config.num_samples)
    print(f"  Created dataset with {len(dataset.tokens)} tokens")
    print(f"  Letters: {len(config.letters)}")
    
    # Get model activations
    print("\n[3/6] Extracting model activations...")
    activations = task.get_activations(
        dataset.tokens,
        layer=transcoder_wrapper.layer,
        batch_size=config.batch_size,
    )
    print(f"  Activations shape: {activations.shape}")
    
    # Train linear probe
    print("\n[4/6] Training linear probe...")
    probe_weights, probe_metrics = train_linear_probe(
        activations,
        dataset.labels,
        l1_penalty=config.probe_l1_penalty,
    )
    print(f"  Probe accuracy: {probe_metrics['accuracy']:.3f}")
    print(f"  Probe F1: {probe_metrics['f1']:.3f}")
    
    # Get transcoder feature activations
    print("\n[5/6] Computing transcoder feature activations...")
    feature_activations = transcoder_wrapper.encode(activations)
    print(f"  Feature activations shape: {feature_activations.shape}")
    
    # Evaluate each letter
    print("\n[6/6] Evaluating absorption for each letter...")
    results_by_letter = {}
    
    for letter_idx, letter in enumerate(tqdm(config.letters, desc="Letters")):
        # Get probe direction for this letter
        probe_direction = probe_weights[letter_idx]  # (d_model,)
        
        # Get binary labels for this letter
        letter_labels = dataset.labels[:, letter_idx]
        
        # Select best feature for this letter
        if config.selection_method == "cosine_similarity":
            feature_idx, cosine_sim = select_feature_by_cosine_similarity(
                probe_direction,
                transcoder_wrapper.W_enc,
            )
            print(f"\n  Letter '{letter}': Feature {feature_idx} (cosine={cosine_sim:.3f})")
        else:
            # K-sparse selection
            selected_features = select_features_k_sparse(
                feature_activations,
                dataset.labels,
                k=config.k_sparse,
                l1_penalty=config.probe_l1_penalty,
            )
            feature_idx = selected_features[0]
            print(f"\n  Letter '{letter}': Feature {feature_idx} (k-sparse)")
        
        # Evaluate feature classification performance
        feature_acts_letter = feature_activations[:, feature_idx]
        classification_metrics = evaluate_feature_classification(
            feature_acts_letter,
            letter_labels,
        )
        
        print(f"    Precision: {classification_metrics['precision']:.3f}")
        print(f"    Recall: {classification_metrics['recall']:.3f}")
        print(f"    F1: {classification_metrics['f1']:.3f}")
        
        # Detect absorption
        if config.calculate_absorption_rate:
            print(f"    Detecting absorption...")
            absorption_analysis = compute_absorption_score(
                model=model,
                transcoder=transcoder_wrapper.transcoder,
                layer=transcoder_wrapper.layer,
                tokens=dataset.tokens,
                labels=letter_labels,
                probe_direction=probe_direction,
                main_feature_idx=feature_idx,
                tokenizer=model.tokenizer,
                cosine_threshold=config.absorption_threshold,
                ablation_threshold=config.ablation_threshold,
                check_causality=True,
            )
            absorption_analysis.letter = letter
            
            print(f"    Absorption rate: {absorption_analysis.absorption_rate:.3f}")
            print(f"    Absorbed tokens: {absorption_analysis.num_absorbed_tokens}/{absorption_analysis.num_total_tokens}")
            print(f"    Absorbing features: {len(absorption_analysis.absorbing_features)}")
            
            # Store results
            results_by_letter[letter] = {
                'feature_idx': feature_idx,
                'classification': classification_metrics,
                'absorption': asdict(absorption_analysis),
                'probe_metrics': {
                    'cosine_similarity': cosine_sim if config.selection_method == "cosine_similarity" else None,
                }
            }
        else:
            results_by_letter[letter] = {
                'feature_idx': feature_idx,
                'classification': classification_metrics,
                'probe_metrics': {
                    'cosine_similarity': cosine_sim if config.selection_method == "cosine_similarity" else None,
                }
            }
    
    # Aggregate results
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)
    
    all_f1_scores = [results_by_letter[l]['classification']['f1'] for l in config.letters]
    all_precisions = [results_by_letter[l]['classification']['precision'] for l in config.letters]
    all_recalls = [results_by_letter[l]['classification']['recall'] for l in config.letters]
    
    print(f"\nClassification Performance:")
    print(f"  Mean F1: {np.mean(all_f1_scores):.3f} ± {np.std(all_f1_scores):.3f}")
    print(f"  Mean Precision: {np.mean(all_precisions):.3f} ± {np.std(all_precisions):.3f}")
    print(f"  Mean Recall: {np.mean(all_recalls):.3f} ± {np.std(all_recalls):.3f}")
    
    if config.calculate_absorption_rate:
        all_absorption_rates = [
            results_by_letter[l]['absorption']['absorption_rate'] 
            for l in config.letters
        ]
        print(f"\nAbsorption Analysis:")
        print(f"  Mean absorption rate: {np.mean(all_absorption_rates):.3f} ± {np.std(all_absorption_rates):.3f}")
        print(f"  Max absorption rate: {np.max(all_absorption_rates):.3f}")
        print(f"  Min absorption rate: {np.min(all_absorption_rates):.3f}")
    
    # Compile final results
    final_results = {
        'config': asdict(config),
        'transcoder_info': transcoder_wrapper.get_feature_stats(),
        'probe_metrics': probe_metrics,
        'results_by_letter': results_by_letter,
        'summary': {
            'mean_f1': float(np.mean(all_f1_scores)),
            'std_f1': float(np.std(all_f1_scores)),
            'mean_precision': float(np.mean(all_precisions)),
            'std_precision': float(np.std(all_precisions)),
            'mean_recall': float(np.mean(all_recalls)),
            'std_recall': float(np.std(all_recalls)),
        }
    }
    
    if config.calculate_absorption_rate:
        final_results['summary'].update({
            'mean_absorption_rate': float(np.mean(all_absorption_rates)),
            'std_absorption_rate': float(np.std(all_absorption_rates)),
            'max_absorption_rate': float(np.max(all_absorption_rates)),
            'min_absorption_rate': float(np.min(all_absorption_rates)),
        })
    
    # Save results
    if save_results:
        output_dir = Path(config.results_dir) / config.model_name / f"layer{transcoder_wrapper.layer}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "absorption_results.json"
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)
    
    return final_results


def evaluate_multiple_layers(
    config: AbsorptionConfig,
    layers: List[int],
    prefix_level: Optional[int] = None,
) -> Dict[int, Dict]:
    """
    Evaluate absorption across multiple layers.
    
    Args:
        config: Evaluation configuration
        layers: List of layer numbers to evaluate
        prefix_level: Prefix level to use for all layers
    
    Returns:
        Dictionary mapping layer number to results
    """
    all_results = {}
    
    for layer in layers:
        print(f"\n{'='*80}")
        print(f"Evaluating Layer {layer}")
        print(f"{'='*80}\n")
        
        try:
            # Load transcoder for this layer
            transcoder_wrapper = load_matryoshka_transcoder(
                model_name=config.model_name,
                layer=layer,
                prefix_level=prefix_level,
                device=config.device,
            )
            
            # Run evaluation
            results = run_absorption_evaluation(
                config=config,
                transcoder_wrapper=transcoder_wrapper,
                save_results=True,
            )
            
            all_results[layer] = results
            
        except Exception as e:
            print(f"Error evaluating layer {layer}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save combined results
    output_file = Path(config.results_dir) / config.model_name / "absorption_all_layers.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Combined results saved to: {output_file}")
    
    return all_results


def compare_probe_vs_transcoder(
    config: AbsorptionConfig,
    transcoder_wrapper: MatryoshkaTranscoderWrapper,
) -> Dict:
    """
    Compare linear probe performance vs transcoder feature performance.
    
    This replicates Figure 2 from the absorption paper.
    
    Args:
        config: Evaluation configuration
        transcoder_wrapper: Wrapped transcoder
    
    Returns:
        Comparison results
    """
    print("\nComparing Linear Probe vs Transcoder Features...")
    
    # Load model
    model = HookedTransformer.from_pretrained_no_processing(
        config.model_name,
        dtype=getattr(torch, config.dtype),
    ).to(config.device)
    
    # Create task
    task = FirstLetterTask(model, model.tokenizer, device=config.device)
    dataset = task.create_dataset(config.num_samples)
    
    # Get activations
    activations = task.get_activations(dataset.tokens, transcoder_wrapper.layer)
    
    # Train probe
    probe_weights, probe_metrics = train_linear_probe(activations, dataset.labels)
    
    # Get transcoder features
    feature_activations = transcoder_wrapper.encode(activations)
    
    # Compare performance
    results = {
        'probe_performance': probe_metrics,
        'transcoder_features': {},
    }
    
    for letter_idx, letter in enumerate(config.letters):
        probe_dir = probe_weights[letter_idx]
        letter_labels = dataset.labels[:, letter_idx]
        
        # Select transcoder feature
        feature_idx, _ = select_feature_by_cosine_similarity(
            probe_dir,
            transcoder_wrapper.W_enc,
        )
        
        # Evaluate transcoder feature
        feature_metrics = evaluate_feature_classification(
            feature_activations[:, feature_idx],
            letter_labels,
        )
        
        results['transcoder_features'][letter] = {
            'feature_idx': feature_idx,
            'metrics': feature_metrics,
        }
    
    return results

