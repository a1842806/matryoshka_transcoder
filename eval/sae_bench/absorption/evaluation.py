"""
Main evaluation script for absorption score.

Runs complete absorption evaluation on Matryoshka transcoders using SAE Bench's methodology.
"""

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from typing import Dict, List, Optional
from pathlib import Path
import json
import numpy as np
import random
from tqdm import tqdm
from dataclasses import asdict

from .config import AbsorptionConfig
from .feature_absorption_calculator import (
    FeatureAbsorptionCalculator,
    AbsorptionResults,
)
from .prompting import first_letter_formatter
from .vocab import LETTERS, get_alpha_tokens
from .matryoshka_wrapper import MatryoshkaTranscoderWrapper, load_matryoshka_transcoder


def select_main_feature_by_cosine(
    probe_direction: torch.Tensor,
    encoder_weights: torch.Tensor,
) -> tuple[int, float]:
    """
    Select main feature by cosine similarity with probe direction.
    
    Args:
        probe_direction: Probe weight vector for a specific class (d_model,)
        encoder_weights: Encoder weight matrix (d_model, n_features)
    
    Returns:
        Tuple of (best_feature_idx, cosine_similarity)
    """
    # Ensure same dtype for computation
    if probe_direction.dtype != encoder_weights.dtype:
        # Convert to float32 for stable computation
        probe_direction = probe_direction.float()
        encoder_weights = encoder_weights.float()
    
    # Normalize vectors
    probe_norm = F.normalize(probe_direction.unsqueeze(0), dim=1)  # (1, d_model)
    encoder_norm = F.normalize(encoder_weights, dim=0)  # (d_model, n_features)
    
    # Compute cosine similarities
    cosine_sims = torch.matmul(probe_norm, encoder_norm).squeeze(0)  # (n_features,)
    
    # Select feature with highest cosine similarity
    best_idx = cosine_sims.argmax().item()
    best_sim = cosine_sims[best_idx].item()
    
    return best_idx, best_sim


def train_probe_on_activations(
    activations: torch.Tensor,
    labels: torch.Tensor,
    device: str = "cuda",
    lr: float = 0.01,
    num_epochs: int = 100,
) -> torch.Tensor:
    """
    Train a simple linear probe on activations.
    
    Args:
        activations: (n_samples, d_model)
        labels: (n_samples, n_classes) one-hot
        device: Device to train on
        lr: Learning rate
        num_epochs: Number of epochs
    
    Returns:
        Probe weights (n_classes, d_model)
    """
    from sklearn.linear_model import LogisticRegression
    
    # Convert to float32 for numpy compatibility (bfloat16 not supported)
    X = activations.float().cpu().numpy()
    y = labels.cpu().numpy().argmax(axis=1)
    
    clf = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=num_epochs,
        random_state=42,
        multi_class='multinomial',
    )
    
    clf.fit(X, y)
    
    # Convert to torch tensor - use float32 for probe weights (more stable than bfloat16)
    probe_weights = torch.tensor(clf.coef_, dtype=torch.float32, device=device)
    
    return probe_weights


def run_absorption_evaluation(
    config: AbsorptionConfig,
    transcoder_wrapper: MatryoshkaTranscoderWrapper,
    save_results: bool = True,
) -> Dict:
    """
    Run full absorption evaluation on a Matryoshka transcoder using SAE Bench methodology.
    
    Args:
        config: Evaluation configuration
        transcoder_wrapper: Wrapped Matryoshka transcoder
        save_results: Whether to save results to disk
    
    Returns:
        Dictionary with evaluation results
    """
    print("="*80)
    print(f"Absorption Score Evaluation (SAE Bench Style)")
    print(f"Model: {config.model_name}, Layer: {transcoder_wrapper.layer}")
    print(f"Transcoder features: {transcoder_wrapper.n_features}")
    print("="*80)
    
    # Load model
    print("\n[1/5] Loading model...")
    model = HookedTransformer.from_pretrained_no_processing(
        config.model_name,
        dtype=getattr(torch, config.dtype),
    ).to(config.device)
    
    # Get vocabulary for ICL prompts
    print("\n[2/5] Loading vocabulary...")
    vocab = get_alpha_tokens(model.tokenizer)
    print(f"  Found {len(vocab)} alpha tokens")
    
    # Get model activations for probe training
    print("\n[3/5] Training linear probe...")
    # Sample tokens for probe training - get token IDs from vocabulary
    vocab_tokens = []
    vocab_labels = []
    
    # Convert vocabulary strings to token IDs
    tokenizer = model.tokenizer
    for token_str in vocab[:min(config.num_samples, len(vocab))]:
        try:
            # Encode token string to get token ID(s)
            token_ids = tokenizer.encode(token_str, add_special_tokens=False)
            # Only use single-token words
            if len(token_ids) == 1:
                token_id = token_ids[0]
                vocab_tokens.append(token_id)
                # Find first letter from token string
                token_clean = token_str.strip().lstrip("▁").lstrip("Ġ").lstrip()
                if len(token_clean) > 0:
                    first_char = token_clean[0].lower()
                    if first_char in LETTERS:
                        vocab_labels.append(LETTERS.index(first_char))
                    else:
                        vocab_tokens.pop()  # Remove if no valid label
                else:
                    vocab_tokens.pop()  # Remove if no valid token
        except Exception:
            continue
    
    if len(vocab_tokens) == 0:
        raise ValueError("No valid tokens found for probe training")
    
    print(f"  Using {len(vocab_tokens)} tokens for probe training")
    
    # Get activations at the correct hook point
    tokens_tensor = torch.tensor(vocab_tokens, device=config.device).unsqueeze(1)
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens_tensor,
            stop_at_layer=transcoder_wrapper.layer + 1,
        )
        # SAE Bench uses hook_resid_post (after the layer)
        hook_name = f"blocks.{transcoder_wrapper.layer}.hook_resid_post"
        activations = cache[hook_name][:, 0, :]  # (n_samples, d_model)
    
    # Create labels
    labels = torch.zeros(len(vocab_labels), len(LETTERS), device=config.device)
    for i, letter_idx in enumerate(vocab_labels):
        labels[i, letter_idx] = 1.0
    
    # Train probe
    probe_weights = train_probe_on_activations(
        activations,
        labels,
        device=config.device,
        lr=config.probe_lr,
        num_epochs=config.probe_epochs,
    )
    print(f"  Probe weights shape: {probe_weights.shape}")
    
    # Create absorption calculator
    print("\n[4/5] Creating absorption calculator...")
    calculator = FeatureAbsorptionCalculator(
        model=model,
        icl_word_list=vocab,
        max_icl_examples=10,
        base_template="{word}:",
        answer_formatter=first_letter_formatter(),
        word_token_pos=-2,
        batch_size=config.batch_size,
        full_absorption_probe_cos_sim_threshold=0.025,
        absorption_fraction_probe_cos_sim_threshold=0.1,
        probe_projection_proportion_threshold=0.4,
        absorption_fraction_max_absorbing_latents=3,
    )
    
    # Evaluate each letter
    print("\n[5/5] Evaluating absorption for each letter...")
    results_by_letter = {}
    
    # Filter vocabulary by first letter for each letter
    vocab_by_letter = {letter: [] for letter in LETTERS}
    for word in vocab:
        # Get first letter from word
        word_clean = word.strip().lstrip("▁").lstrip("Ġ").lstrip()
        if len(word_clean) > 0:
            first_char = word_clean[0].lower()
            if first_char in vocab_by_letter:
                vocab_by_letter[first_char].append(word)
    
    # Determine how many words to use per letter
    words_per_letter = min(500, min(len(words) for words in vocab_by_letter.values() if len(words) > 0))
    print(f"  Using {words_per_letter} words per letter for evaluation")
    
    for letter_idx, letter in enumerate(tqdm(LETTERS, desc="Letters")):
        # Get words that start with this letter
        letter_words = vocab_by_letter[letter]
        if len(letter_words) == 0:
            print(f"  Warning: No words found for letter '{letter}', skipping...")
            results_by_letter[letter] = {
                'main_feature_idx': -1,
                'cosine_similarity': 0.0,
                'mean_absorption_fraction': 0.0,
                'full_absorption_rate': 0.0,
                'n_words_evaluated': 0,
                'absorption_fractions': [],
                'full_absorptions': [],
            }
            continue
        
        # Sample words for this letter (use more words for better statistics)
        if len(letter_words) > words_per_letter:
            eval_words = random.sample(letter_words, words_per_letter)
        else:
            eval_words = letter_words
        
        # Get probe direction for this letter
        probe_direction = probe_weights[letter_idx]  # (d_model,)
        
        # Select main feature
        main_feature_idx, cosine_sim = select_main_feature_by_cosine(
            probe_direction,
            transcoder_wrapper.W_enc,
        )
        
        # Calculate absorption
        absorption_results = calculator.calculate_absorption(
            transcoder=transcoder_wrapper,
            words=eval_words,
            probe_direction=probe_direction,
            main_feature_ids=[main_feature_idx],
            layer=transcoder_wrapper.layer,
            show_progress=False,
        )
        
        # Aggregate results
        absorption_fractions = [
            result.absorption_fraction for result in absorption_results.word_results
        ]
        full_absorptions = [
            result.is_full_absorption for result in absorption_results.word_results
        ]
        probe_projections = [
            result.probe_projection for result in absorption_results.word_results
        ]
        
        # Calculate statistics
        mean_absorption_fraction = np.mean(absorption_fractions)
        full_absorption_rate = np.mean(full_absorptions)
        
        # Count words with non-zero absorption
        non_zero_absorption = [f for f in absorption_fractions if f > 0]
        n_non_zero = len(non_zero_absorption)
        mean_non_zero_absorption = np.mean(non_zero_absorption) if n_non_zero > 0 else 0.0
        
        # Count words with positive probe projection (where absorption could occur)
        positive_probe_proj = [p for p in probe_projections if p > 0]
        n_positive_probe = len(positive_probe_proj)
        
        # Calculate median and max for better understanding
        median_absorption_fraction = np.median(absorption_fractions)
        max_absorption_fraction = np.max(absorption_fractions) if len(absorption_fractions) > 0 else 0.0
        
        results_by_letter[letter] = {
            'main_feature_idx': main_feature_idx,
            'cosine_similarity': float(cosine_sim),
            'mean_absorption_fraction': float(mean_absorption_fraction),
            'median_absorption_fraction': float(median_absorption_fraction),
            'max_absorption_fraction': float(max_absorption_fraction),
            'full_absorption_rate': float(full_absorption_rate),
            'n_words_evaluated': len(eval_words),
            'n_non_zero_absorption': n_non_zero,
            'n_positive_probe_proj': n_positive_probe,
            'mean_non_zero_absorption': float(mean_non_zero_absorption),
            'absorption_fractions': [float(f) for f in absorption_fractions],
            'full_absorptions': [bool(f) for f in full_absorptions],
        }
    
    # Aggregate statistics
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)
    
    all_absorption_fractions = [
        results_by_letter[l]['mean_absorption_fraction'] for l in LETTERS
    ]
    all_full_absorption_rates = [
        results_by_letter[l]['full_absorption_rate'] for l in LETTERS
    ]
    all_cosines = [
        results_by_letter[l]['cosine_similarity'] for l in LETTERS
    ]
    
    print(f"\nAbsorption Analysis:")
    print(f"  Mean absorption fraction: {np.mean(all_absorption_fractions):.4f} ± {np.std(all_absorption_fractions):.4f}")
    print(f"  Median absorption fraction: {np.median(all_absorption_fractions):.4f}")
    print(f"  Max absorption fraction: {np.max(all_absorption_fractions):.4f}")
    print(f"  Mean full absorption rate: {np.mean(all_full_absorption_rates):.3f} ± {np.std(all_full_absorption_rates):.3f}")
    print(f"  Mean cosine similarity: {np.mean(all_cosines):.3f} ± {np.std(all_cosines):.3f}")
    
    # Calculate aggregate stats for non-zero absorption
    all_non_zero_counts = [
        results_by_letter[l].get('n_non_zero_absorption', 0) for l in LETTERS
    ]
    all_word_counts = [
        results_by_letter[l].get('n_words_evaluated', 0) for l in LETTERS
    ]
    total_non_zero = sum(all_non_zero_counts)
    total_words = sum(all_word_counts)
    print(f"\nDetailed Statistics:")
    print(f"  Words with non-zero absorption: {total_non_zero}/{total_words} ({100*total_non_zero/max(total_words,1):.2f}%)")
    if total_non_zero > 0:
        all_non_zero_absorptions = [
            results_by_letter[l].get('mean_non_zero_absorption', 0) for l in LETTERS
            if results_by_letter[l].get('n_non_zero_absorption', 0) > 0
        ]
        if all_non_zero_absorptions:
            print(f"  Mean absorption (non-zero only): {np.mean(all_non_zero_absorptions):.4f}")
    
    # Compile final results
    summary_dict = {
        'mean_absorption_fraction': float(np.mean(all_absorption_fractions)),
        'std_absorption_fraction': float(np.std(all_absorption_fractions)),
        'median_absorption_fraction': float(np.median(all_absorption_fractions)),
        'max_absorption_fraction': float(np.max(all_absorption_fractions)),
        'mean_full_absorption_rate': float(np.mean(all_full_absorption_rates)),
        'std_full_absorption_rate': float(np.std(all_full_absorption_rates)),
        'mean_cosine_similarity': float(np.mean(all_cosines)),
        'std_cosine_similarity': float(np.std(all_cosines)),
        'total_words_evaluated': int(total_words),
        'total_non_zero_absorption': int(total_non_zero),
        'fraction_with_absorption': float(total_non_zero / max(total_words, 1)),
    }
    
    final_results = {
        'config': asdict(config),
        'transcoder_info': transcoder_wrapper.get_feature_stats(),
        'results_by_letter': results_by_letter,
        'summary': summary_dict,
    }
    
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
