"""
Main evaluation script for Sparse Probing.

Trains linear probes for each SAE feature to predict concepts,
evaluating how well individual features correspond to interpretable concepts.
"""

import torch
from transformer_lens import HookedTransformer
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from dataclasses import asdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from .config import SparseProbingConfig
from ..absorption.first_letter_task import FirstLetterTask
from ..absorption.matryoshka_wrapper import MatryoshkaTranscoderWrapper, load_matryoshka_transcoder


def train_feature_probe(
    feature_activations: torch.Tensor,
    labels: torch.Tensor,
    l1_penalty: float = 0.01,
) -> Tuple[LogisticRegression, Dict[str, float]]:
    """
    Train a linear probe for a single feature to predict binary labels.
    
    Args:
        feature_activations: Feature activation values (n_samples,)
        labels: Binary labels (n_samples,)
        l1_penalty: L1 regularization strength
    
    Returns:
        Tuple of (trained_classifier, metrics_dict)
    """
    # Convert to numpy for sklearn (convert bfloat16 to float32 first)
    X = feature_activations.cpu().float().numpy().reshape(-1, 1)
    y = labels.cpu().float().numpy()
    
    # Train logistic regression
    clf = LogisticRegression(
        penalty='l1',
        C=1.0 / l1_penalty,
        solver='liblinear',
        max_iter=1000,
        random_state=42,
    )
    clf.fit(X, y)
    
    # Evaluate on training set
    y_pred = clf.predict(X)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    accuracy = accuracy_score(y, y_pred)
    
    metrics = {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
        'weight': float(clf.coef_[0, 0]),
        'bias': float(clf.intercept_[0]),
    }
    
    return clf, metrics


def evaluate_feature_probing(
    config: SparseProbingConfig,
    transcoder_wrapper: MatryoshkaTranscoderWrapper,
    save_results: bool = True,
) -> Dict:
    """
    Evaluate sparse probing on a Matryoshka transcoder.
    
    For each feature, trains a linear probe to predict each concept (letter),
    then reports which features are most predictive for each concept.
    
    Args:
        config: Evaluation configuration
        transcoder_wrapper: Wrapped Matryoshka transcoder
        save_results: Whether to save results to disk
    
    Returns:
        Dictionary with evaluation results
    """
    print("="*80)
    print(f"Sparse Probing Evaluation")
    print(f"Model: {config.model_name}, Layer: {transcoder_wrapper.layer}")
    print(f"Transcoder features: {transcoder_wrapper.n_features}")
    print("="*80)
    
    # Load model
    print("\n[1/5] Loading model...")
    model = HookedTransformer.from_pretrained_no_processing(
        config.model_name,
        dtype=getattr(torch, config.dtype),
    ).to(config.device)
    
    # Create first-letter task
    print("\n[2/5] Creating first-letter classification dataset...")
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
    print("\n[3/5] Extracting model activations...")
    activations = task.get_activations(
        dataset.tokens,
        layer=transcoder_wrapper.layer,
        batch_size=config.batch_size,
    )
    print(f"  Activations shape: {activations.shape}")
    
    # Get transcoder feature activations
    print("\n[4/5] Computing transcoder feature activations...")
    feature_activations = transcoder_wrapper.encode(activations)
    print(f"  Feature activations shape: {feature_activations.shape}")
    
    n_features = feature_activations.shape[1]
    n_samples = feature_activations.shape[0]
    
    # Filter features by activation rate if needed
    if config.min_feature_activation_rate > 0:
        activation_rates = (feature_activations > config.activation_threshold).float().mean(dim=0)
        active_features = (activation_rates >= config.min_feature_activation_rate).nonzero(as_tuple=True)[0]
        print(f"  Active features (activation rate >= {config.min_feature_activation_rate}): {len(active_features)}/{n_features}")
    else:
        active_features = torch.arange(n_features, device=feature_activations.device)
    
    # Limit to max_features if specified
    if config.max_features is not None and len(active_features) > config.max_features:
        # Select top features by activation frequency
        activation_counts = (feature_activations > config.activation_threshold).float().sum(dim=0)
        _, top_indices = torch.topk(activation_counts[active_features], k=config.max_features)
        active_features = active_features[top_indices]
        print(f"  Limited to top {config.max_features} features by activation frequency")
    
    active_features = active_features.cpu().tolist()
    n_features_to_eval = len(active_features)
    print(f"  Evaluating {n_features_to_eval} features")
    
    # Train/test split
    n_train = int(n_samples * config.train_test_split)
    train_indices = torch.arange(n_train)
    test_indices = torch.arange(n_train, n_samples)
    
    # Evaluate each feature on each concept
    print("\n[5/5] Training probes for each feature-concept pair...")
    results_by_letter = {}
    feature_results = {}  # Store results by feature for summary
    
    for letter_idx, letter in enumerate(tqdm(config.letters, desc="Concepts")):
        # Get binary labels for this letter
        letter_labels = dataset.labels[:, letter_idx]
        train_labels = letter_labels[train_indices]
        test_labels = letter_labels[test_indices]
        
        letter_results = []
        
        for feature_idx in tqdm(active_features, desc=f"  Letter '{letter}'", leave=False):
            # Get feature activations
            feature_acts = feature_activations[:, feature_idx]
            train_acts = feature_acts[train_indices]
            test_acts = feature_acts[test_indices]
            
            # Check if we have both classes in training set
            train_labels_np = train_labels.cpu().numpy()
            unique_classes = np.unique(train_labels_np)
            if len(unique_classes) < 2:
                # Skip this feature-concept pair if only one class present
                continue
            
            # Train probe on training set
            probe_clf, train_metrics = train_feature_probe(
                train_acts,
                train_labels,
                l1_penalty=config.probe_l1_penalty,
            )
            
            # Evaluate on test set using trained probe (convert bfloat16 to float32 first)
            test_acts_np = test_acts.cpu().float().numpy().reshape(-1, 1)
            test_labels_np = test_labels.cpu().float().numpy()
            test_preds = probe_clf.predict(test_acts_np)
            
            # Calculate test metrics
            test_precision = precision_score(test_labels_np, test_preds, zero_division=0)
            test_recall = recall_score(test_labels_np, test_preds, zero_division=0)
            test_f1 = f1_score(test_labels_np, test_preds, zero_division=0)
            test_accuracy = accuracy_score(test_labels_np, test_preds)
            
            # Calculate activation statistics
            activation_rate = (feature_acts > config.activation_threshold).float().mean().item()
            mean_activation = feature_acts.mean().item()
            max_activation = feature_acts.max().item()
            
            result = {
                'feature_idx': feature_idx,
                'train_metrics': train_metrics,
                'test_metrics': {
                    'precision': float(test_precision),
                    'recall': float(test_recall),
                    'f1': float(test_f1),
                    'accuracy': float(test_accuracy),
                },
                'activation_stats': {
                    'activation_rate': float(activation_rate),
                    'mean_activation': float(mean_activation),
                    'max_activation': float(max_activation),
                },
            }
            
            letter_results.append(result)
            
            # Store in feature_results for summary
            if feature_idx not in feature_results:
                feature_results[feature_idx] = {}
            feature_results[feature_idx][letter] = {
                'test_f1': float(test_f1),
                'test_precision': float(test_precision),
                'test_recall': float(test_recall),
            }
        
        # Sort by test F1 score
        letter_results.sort(key=lambda x: x['test_metrics']['f1'], reverse=True)
        
        # Store top k results (handle empty case)
        if len(letter_results) > 0:
            results_by_letter[letter] = {
                'top_features': letter_results[:config.top_k_features],
                'all_features_count': len(letter_results),
                'mean_f1': float(np.mean([r['test_metrics']['f1'] for r in letter_results])),
                'max_f1': float(np.max([r['test_metrics']['f1'] for r in letter_results])),
            }
        else:
            # No valid features for this letter
            results_by_letter[letter] = {
                'top_features': [],
                'all_features_count': 0,
                'mean_f1': 0.0,
                'max_f1': 0.0,
            }
    
    # Aggregate statistics
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)
    
    all_max_f1s = [results_by_letter[l]['max_f1'] for l in config.letters if results_by_letter[l]['all_features_count'] > 0]
    all_mean_f1s = [results_by_letter[l]['mean_f1'] for l in config.letters if results_by_letter[l]['all_features_count'] > 0]
    
    print(f"\nPer-Concept Performance:")
    if len(all_max_f1s) > 0:
        print(f"  Mean max F1: {np.mean(all_max_f1s):.3f} ± {np.std(all_max_f1s):.3f}")
        print(f"  Mean average F1: {np.mean(all_mean_f1s):.3f} ± {np.std(all_mean_f1s):.3f}")
        print(f"  Concepts with valid features: {len(all_max_f1s)}/{len(config.letters)}")
    else:
        print(f"  No valid features found for any concept")
    
    # Find best features overall (features that are top predictors for multiple concepts)
    feature_concept_count = {}
    for letter in config.letters:
        top_feature_indices = [r['feature_idx'] for r in results_by_letter[letter]['top_features']]
        for feat_idx in top_feature_indices:
            feature_concept_count[feat_idx] = feature_concept_count.get(feat_idx, 0) + 1
    
    multi_concept_features = {k: v for k, v in feature_concept_count.items() if v > 1}
    print(f"\nFeature Statistics:")
    print(f"  Features predicting multiple concepts: {len(multi_concept_features)}")
    if len(multi_concept_features) > 0:
        print(f"  Max concepts per feature: {max(multi_concept_features.values())}")
    
    # Compile final results
    final_results = {
        'config': asdict(config),
        'transcoder_info': transcoder_wrapper.get_feature_stats(),
        'results_by_letter': results_by_letter,
        'summary': {
            'mean_max_f1': float(np.mean(all_max_f1s)),
            'std_max_f1': float(np.std(all_max_f1s)),
            'mean_avg_f1': float(np.mean(all_mean_f1s)),
            'std_avg_f1': float(np.std(all_mean_f1s)),
            'n_features_evaluated': n_features_to_eval,
            'n_features_total': n_features,
            'multi_concept_features_count': len(multi_concept_features),
        }
    }
    
    # Save results
    if save_results:
        output_dir = Path(config.results_dir) / config.model_name / f"layer{transcoder_wrapper.layer}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "sparse_probing_results.json"
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)
    
    return final_results


def evaluate_multiple_layers(
    config: SparseProbingConfig,
    layers: List[int],
    prefix_level: Optional[int] = None,
) -> Dict[int, Dict]:
    """
    Evaluate sparse probing across multiple layers.
    
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
            results = evaluate_feature_probing(
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
    output_file = Path(config.results_dir) / config.model_name / "sparse_probing_all_layers.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Combined results saved to: {output_file}")
    
    return all_results

