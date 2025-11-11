"""
First-letter identification task for evaluating feature absorption.

This module implements the methodology from the absorption paper for creating
a ground-truth classification task where we know exactly which tokens should
activate for each feature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score


@dataclass
class FirstLetterDataset:
    """Dataset for first-letter classification task."""
    tokens: torch.Tensor  # (n_samples,) token IDs
    letters: List[str]  # (n_samples,) first letters
    labels: torch.Tensor  # (n_samples, 26) one-hot labels
    token_strings: List[str]  # (n_samples,) decoded token strings
    

class FirstLetterTask:
    """
    First-letter identification task.
    
    Creates a classification task where we have ground truth labels for
    which tokens start with which letters. Used to evaluate if SAE/transcoder
    features track this concept accurately.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        letters: List[str] = list("abcdefghijklmnopqrstuvwxyz"),
        min_tokens_per_letter: int = 100,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.letters = letters
        self.min_tokens_per_letter = min_tokens_per_letter
        self.device = device
        
        # Letter to index mapping
        self.letter_to_idx = {letter: idx for idx, letter in enumerate(letters)}
        self.idx_to_letter = {idx: letter for letter, idx in self.letter_to_idx.items()}
        
    def create_dataset(self, num_samples: int = 10000) -> FirstLetterDataset:
        """
        Create a balanced dataset of tokens for first-letter classification.
        
        Args:
            num_samples: Total number of samples to collect (will be balanced across letters)
        
        Returns:
            FirstLetterDataset with tokens and labels
        """
        vocab_size = self.tokenizer.vocab_size
        
        # Collect tokens by first letter
        tokens_by_letter: Dict[str, List[int]] = {letter: [] for letter in self.letters}
        token_strings: Dict[str, List[str]] = {letter: [] for letter in self.letters}
        
        # Scan vocabulary for tokens starting with each letter
        for token_id in range(vocab_size):
            token_str = self.tokenizer.decode([token_id])
            
            # Clean up token string (remove spaces, special chars at start)
            token_str_clean = token_str.strip().lstrip("▁").lstrip("Ġ").lstrip()
            
            if len(token_str_clean) == 0:
                continue
            
            first_char = token_str_clean[0].lower()
            if first_char in self.letter_to_idx:
                tokens_by_letter[first_char].append(token_id)
                token_strings[first_char].append(token_str)
        
        # Balance dataset across letters
        samples_per_letter = num_samples // len(self.letters)
        samples_per_letter = max(samples_per_letter, self.min_tokens_per_letter)
        
        all_tokens = []
        all_letters = []
        all_token_strings = []
        
        for letter in self.letters:
            letter_tokens = tokens_by_letter[letter]
            letter_strings = token_strings[letter]
            
            if len(letter_tokens) < self.min_tokens_per_letter:
                print(f"Warning: Only found {len(letter_tokens)} tokens for letter '{letter}'")
            
            # Sample with replacement if needed
            n_samples = min(samples_per_letter, len(letter_tokens))
            indices = np.random.choice(len(letter_tokens), size=n_samples, replace=False)
            
            sampled_tokens = [letter_tokens[i] for i in indices]
            sampled_strings = [letter_strings[i] for i in indices]
            
            all_tokens.extend(sampled_tokens)
            all_letters.extend([letter] * len(sampled_tokens))
            all_token_strings.extend(sampled_strings)
        
        # Convert to tensors
        tokens_tensor = torch.tensor(all_tokens, dtype=torch.long, device=self.device)
        
        # Create one-hot labels
        labels = torch.zeros(len(all_tokens), len(self.letters), device=self.device)
        for i, letter in enumerate(all_letters):
            labels[i, self.letter_to_idx[letter]] = 1.0
        
        return FirstLetterDataset(
            tokens=tokens_tensor,
            letters=all_letters,
            labels=labels,
            token_strings=all_token_strings,
        )
    
    def get_activations(
        self,
        tokens: torch.Tensor,
        layer: int,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Get model activations for tokens at a specific layer.
        
        Args:
            tokens: Token IDs (n_samples,)
            layer: Layer number to extract activations from
            batch_size: Batch size for processing
        
        Returns:
            Activations tensor (n_samples, d_model)
        """
        all_activations = []
        
        with torch.no_grad():
            for i in range(0, len(tokens), batch_size):
                batch_tokens = tokens[i:i+batch_size].unsqueeze(1)  # (batch, 1)
                
                # Run model and capture activations
                _, cache = self.model.run_with_cache(batch_tokens, stop_at_layer=layer+1)
                
                # Get residual stream activations after layer
                hook_name = f"blocks.{layer}.hook_resid_post"
                activations = cache[hook_name][:, 0, :]  # (batch, d_model)
                
                all_activations.append(activations)
        
        return torch.cat(all_activations, dim=0)


def train_linear_probe(
    activations: torch.Tensor,
    labels: torch.Tensor,
    l1_penalty: float = 0.01,
    max_iter: int = 1000,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Train a linear probe for classification task.
    
    Args:
        activations: Input activations (n_samples, d_model)
        labels: One-hot labels (n_samples, n_classes)
        l1_penalty: L1 regularization strength
        max_iter: Maximum iterations for training
    
    Returns:
        Tuple of (probe_weights, metrics_dict)
    """
    X = activations.cpu().numpy()
    y = labels.cpu().numpy().argmax(axis=1)
    
    # Train logistic regression with L1 penalty
    clf = LogisticRegression(
        penalty='l1',
        C=1.0 / l1_penalty,
        solver='saga',
        max_iter=max_iter,
        multi_class='multinomial',
        random_state=42,
    )
    
    clf.fit(X, y)
    
    # Get predictions
    y_pred = clf.predict(X)
    
    # Calculate metrics
    precision = precision_score(y, y_pred, average='macro', zero_division=0)
    recall = recall_score(y, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y, y_pred, average='macro', zero_division=0)
    accuracy = (y == y_pred).mean()
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
    }
    
    # Convert weights to torch tensor
    probe_weights = torch.tensor(clf.coef_, dtype=activations.dtype, device=activations.device)
    
    return probe_weights, metrics


def evaluate_feature_classification(
    feature_activations: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.0,
) -> Dict[str, float]:
    """
    Evaluate a feature's classification performance.
    
    Args:
        feature_activations: Feature activation values (n_samples,)
        labels: Binary labels (n_samples,)
        threshold: Activation threshold for classification
    
    Returns:
        Dictionary with precision, recall, f1 scores
    """
    predictions = (feature_activations > threshold).float()
    
    # True positives, false positives, false negatives
    tp = ((predictions == 1) & (labels == 1)).sum().item()
    fp = ((predictions == 1) & (labels == 0)).sum().item()
    fn = ((predictions == 0) & (labels == 1)).sum().item()
    tn = ((predictions == 0) & (labels == 0)).sum().item()
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
    }


def select_feature_by_cosine_similarity(
    probe_direction: torch.Tensor,
    encoder_weights: torch.Tensor,
) -> Tuple[int, float]:
    """
    Select SAE/transcoder feature by cosine similarity with probe direction.
    
    Args:
        probe_direction: Probe weight vector for a specific class (d_model,)
        encoder_weights: Encoder weight matrix (d_model, n_features)
    
    Returns:
        Tuple of (best_feature_idx, cosine_similarity)
    """
    # Normalize vectors
    probe_norm = F.normalize(probe_direction.unsqueeze(0), dim=1)  # (1, d_model)
    encoder_norm = F.normalize(encoder_weights, dim=0)  # (d_model, n_features)
    
    # Compute cosine similarities
    cosine_sims = torch.matmul(probe_norm, encoder_norm).squeeze(0)  # (n_features,)
    
    # Select feature with highest cosine similarity
    best_idx = cosine_sims.argmax().item()
    best_sim = cosine_sims[best_idx].item()
    
    return best_idx, best_sim


def select_features_k_sparse(
    activations: torch.Tensor,
    labels: torch.Tensor,
    k: int = 1,
    l1_penalty: float = 0.01,
) -> List[int]:
    """
    Select top k features using k-sparse probing.
    
    Args:
        activations: SAE/transcoder feature activations (n_samples, n_features)
        labels: One-hot labels (n_samples, n_classes)
        k: Number of features to select
        l1_penalty: L1 regularization strength
    
    Returns:
        List of selected feature indices
    """
    probe_weights, _ = train_linear_probe(activations, labels, l1_penalty=l1_penalty)
    
    # Get top k features by absolute weight across all classes
    feature_importance = probe_weights.abs().sum(dim=0)  # Sum across classes
    top_k_indices = torch.topk(feature_importance, k=k).indices.tolist()
    
    return top_k_indices

