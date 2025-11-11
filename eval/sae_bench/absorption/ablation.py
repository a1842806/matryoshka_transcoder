"""
Ablation experiments for causal analysis of SAE/transcoder features.

Implements the ablation methodology from the absorption paper to verify
that features have causal impact on model behavior.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class AblationResult:
    """Result from a feature ablation experiment."""
    feature_idx: int
    token_idx: int
    baseline_metric: float
    ablated_metric: float
    ablation_effect: float  # baseline - ablated (positive = feature increases metric)


class FeatureAblator:
    """
    Perform ablation experiments on SAE/transcoder features.
    
    Follows Algorithm 1 from the absorption paper:
    1. Insert SAE/transcoder in model computation path
    2. Define scalar metric on model output
    3. Calculate baseline metric
    4. For each feature: set activation to 0, recalculate metric
    5. Compute ablation effect
    """
    
    def __init__(
        self,
        model,
        transcoder,
        layer: int,
        device: str = "cuda",
    ):
        self.model = model
        self.transcoder = transcoder
        self.layer = layer
        self.device = device
        
    def ablate_feature(
        self,
        tokens: torch.Tensor,
        feature_idx: int,
        token_position: int,
        metric_fn: Callable[[torch.Tensor], float],
    ) -> AblationResult:
        """
        Ablate a single feature and measure the effect.
        
        Args:
            tokens: Input tokens (batch, seq_len)
            feature_idx: Index of feature to ablate
            token_position: Token position to ablate at
            metric_fn: Function that computes scalar metric from logits
        
        Returns:
            AblationResult with baseline and ablated metrics
        """
        # Get baseline metric (with feature active)
        with torch.no_grad():
            baseline_metric = self._compute_metric_with_transcoder(
                tokens, metric_fn, ablate_features=None
            )
        
        # Get ablated metric (with feature set to 0)
        with torch.no_grad():
            ablated_metric = self._compute_metric_with_transcoder(
                tokens, metric_fn, ablate_features=[(feature_idx, token_position)]
            )
        
        ablation_effect = baseline_metric - ablated_metric
        
        return AblationResult(
            feature_idx=feature_idx,
            token_idx=token_position,
            baseline_metric=baseline_metric,
            ablated_metric=ablated_metric,
            ablation_effect=ablation_effect,
        )
    
    def _compute_metric_with_transcoder(
        self,
        tokens: torch.Tensor,
        metric_fn: Callable[[torch.Tensor], float],
        ablate_features: Optional[List[Tuple[int, int]]] = None,
    ) -> float:
        """
        Run model with transcoder intervention and compute metric.
        
        Args:
            tokens: Input tokens
            metric_fn: Metric function
            ablate_features: List of (feature_idx, token_pos) to ablate
        
        Returns:
            Metric value
        """
        # Hook to intervene with transcoder
        def transcoder_hook(activations, hook):
            # activations: (batch, seq_len, d_model)
            batch_size, seq_len, d_model = activations.shape
            
            # Get source activations (e.g., mlp_in)
            x_source = activations
            
            # Encode to features
            features, features_topk = self.transcoder.compute_activations(
                x_source.view(-1, d_model)
            )
            features = features.view(batch_size, seq_len, -1)
            features_topk = features_topk.view(batch_size, seq_len, -1)
            
            # Ablate specified features
            if ablate_features is not None:
                for feature_idx, token_pos in ablate_features:
                    features_topk[:, token_pos, feature_idx] = 0.0
            
            # Decode to target reconstruction
            x_reconstruct = (
                features_topk.view(-1, features_topk.shape[-1]) @ self.transcoder.W_dec 
                + self.transcoder.b_dec
            )
            x_reconstruct = x_reconstruct.view(batch_size, seq_len, d_model)
            
            # For transcoder, we replace the activations with the reconstruction
            # This allows the model to continue with transcoded features
            return x_reconstruct
        
        # Insert hook at the appropriate layer
        hook_name = f"blocks.{self.layer}.hook_mlp_in"
        
        with self.model.hooks([(hook_name, transcoder_hook)]):
            logits = self.model(tokens)
        
        return metric_fn(logits)


def integrated_gradients_ablation(
    model,
    transcoder,
    tokens: torch.Tensor,
    feature_idx: int,
    token_position: int,
    metric_fn: Callable[[torch.Tensor], float],
    layer: int,
    n_steps: int = 10,
) -> float:
    """
    Approximate ablation effect using integrated gradients.
    
    This is faster than full ablation and provides a good approximation.
    
    Args:
        model: Language model
        transcoder: Matryoshka transcoder
        tokens: Input tokens
        feature_idx: Feature to ablate
        token_position: Token position
        metric_fn: Metric function
        layer: Layer number
        n_steps: Number of integration steps
    
    Returns:
        Approximate ablation effect
    """
    def compute_with_feature_scale(scale: float) -> float:
        """Compute metric with feature scaled by given amount."""
        def hook_fn(activations, hook):
            batch_size, seq_len, d_model = activations.shape
            x_source = activations
            
            # Encode
            features, features_topk = transcoder.compute_activations(
                x_source.view(-1, d_model)
            )
            features = features.view(batch_size, seq_len, -1)
            features_topk = features_topk.view(batch_size, seq_len, -1)
            
            # Scale feature
            features_topk[:, token_position, feature_idx] *= scale
            
            # Decode
            x_reconstruct = (
                features_topk.view(-1, features_topk.shape[-1]) @ transcoder.W_dec 
                + transcoder.b_dec
            )
            x_reconstruct = x_reconstruct.view(batch_size, seq_len, d_model)
            
            return x_reconstruct
        
        hook_name = f"blocks.{layer}.hook_mlp_in"
        with model.hooks([(hook_name, hook_fn)]):
            logits = model(tokens)
        
        return metric_fn(logits)
    
    # Compute integrated gradients
    with torch.no_grad():
        baseline = compute_with_feature_scale(0.0)  # Feature ablated
        full = compute_with_feature_scale(1.0)  # Feature active
        
        # Integrate over path
        integrated_value = 0.0
        for i in range(1, n_steps):
            alpha = i / n_steps
            integrated_value += compute_with_feature_scale(alpha)
        
        # Average and scale
        integrated_value /= (n_steps - 1)
    
    # Ablation effect approximation
    ablation_effect = full - baseline
    
    return ablation_effect


def compute_ablation_effect(
    model,
    transcoder,
    tokens: torch.Tensor,
    feature_idx: int,
    token_position: int,
    target_letter_idx: int,
    layer: int,
    use_integrated_gradients: bool = True,
) -> float:
    """
    Compute ablation effect for first-letter task.
    
    Measures how much removing a feature reduces the logit for the correct letter.
    
    Args:
        model: Language model
        transcoder: Matryoshka transcoder  
        tokens: Input tokens (should be ICL prompt)
        feature_idx: Feature to ablate
        token_position: Position of subject token
        target_letter_idx: Index of correct letter token
        layer: Layer number
        use_integrated_gradients: Use IG approximation for speed
    
    Returns:
        Ablation effect (positive = feature increases correct letter logit)
    """
    def metric_fn(logits: torch.Tensor) -> float:
        """Extract logit for target letter at last position."""
        last_position_logits = logits[0, -1, :]  # (vocab_size,)
        return last_position_logits[target_letter_idx].item()
    
    if use_integrated_gradients:
        return integrated_gradients_ablation(
            model, transcoder, tokens, feature_idx, token_position, 
            metric_fn, layer, n_steps=10
        )
    else:
        ablator = FeatureAblator(model, transcoder, layer)
        result = ablator.ablate_feature(tokens, feature_idx, token_position, metric_fn)
        return result.ablation_effect


def batch_ablation_analysis(
    model,
    transcoder,
    tokens_list: List[torch.Tensor],
    features_to_test: List[int],
    token_positions: List[int],
    target_letter_indices: List[int],
    layer: int,
    use_integrated_gradients: bool = True,
) -> Dict[int, List[float]]:
    """
    Perform ablation analysis on multiple examples.
    
    Args:
        model: Language model
        transcoder: Matryoshka transcoder
        tokens_list: List of token sequences
        features_to_test: List of feature indices to test
        token_positions: Token positions for each example
        target_letter_indices: Target letter indices for each example
        layer: Layer number
        use_integrated_gradients: Use IG approximation
    
    Returns:
        Dictionary mapping feature_idx to list of ablation effects
    """
    results = {feat_idx: [] for feat_idx in features_to_test}
    
    for tokens, token_pos, target_idx in zip(tokens_list, token_positions, target_letter_indices):
        for feat_idx in features_to_test:
            effect = compute_ablation_effect(
                model, transcoder, tokens, feat_idx, token_pos,
                target_idx, layer, use_integrated_gradients
            )
            results[feat_idx].append(effect)
    
    return results

