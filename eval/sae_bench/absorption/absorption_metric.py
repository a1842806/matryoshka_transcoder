"""
Feature absorption detection and measurement.

Implements the core absorption metric from the paper:
"A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders"

Feature absorption occurs when a seemingly monosemantic latent fails to fire
on arbitrary tokens where token-aligned latents activate instead.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class AbsorptionAnalysis:
    """Results from absorption analysis for a single feature."""
    main_feature_idx: int
    letter: str
    
    # Classification metrics
    precision: float
    recall: float
    f1: float
    
    # Absorption metrics  
    absorption_rate: float
    num_absorbed_tokens: int
    num_total_tokens: int
    
    # Absorbing features
    absorbing_features: List[int] = field(default_factory=list)
    absorbing_feature_cosines: List[float] = field(default_factory=list)
    absorbing_feature_ablations: List[float] = field(default_factory=list)
    
    # Examples of absorption
    absorbed_token_ids: List[int] = field(default_factory=list)
    absorbed_token_strings: List[str] = field(default_factory=list)


class AbsorptionMetric:
    """
    Compute absorption metrics for SAE/transcoder features.
    
    The absorption metric identifies cases where:
    1. A feature appears to track a concept (e.g., starts with 'S')
    2. But fails to activate on certain tokens
    3. Instead, token-aligned features activate and contribute the probe direction
    4. These absorbing features have causal impact (verified by ablation)
    """
    
    def __init__(
        self,
        model,
        transcoder,
        layer: int,
        probe_direction: torch.Tensor,
        cosine_threshold: float = 0.3,
        ablation_threshold: float = 0.01,
        device: str = "cuda",
    ):
        """
        Args:
            model: Language model
            transcoder: Matryoshka transcoder
            layer: Layer to analyze
            probe_direction: Linear probe direction for the concept (d_model,)
            cosine_threshold: Minimum cosine similarity to consider absorption
            ablation_threshold: Minimum ablation effect to verify causality
            device: Device to run on
        """
        self.model = model
        self.transcoder = transcoder
        self.layer = layer
        self.probe_direction = probe_direction
        self.cosine_threshold = cosine_threshold
        self.ablation_threshold = ablation_threshold
        self.device = device
        
    def detect_absorption(
        self,
        tokens: torch.Tensor,
        token_labels: torch.Tensor,
        main_feature_idx: int,
        tokenizer,
        check_causality: bool = True,
    ) -> AbsorptionAnalysis:
        """
        Detect absorption for a specific feature.
        
        Args:
            tokens: Token IDs (n_samples,)
            token_labels: Binary labels (n_samples,) indicating if token has concept
            main_feature_idx: Index of main feature that should track concept
            tokenizer: Tokenizer for decoding
            check_causality: Whether to run ablation to verify causality
        
        Returns:
            AbsorptionAnalysis with detailed results
        """
        # Get feature activations
        feature_acts = self._get_feature_activations(tokens, main_feature_idx)
        
        # Find tokens where main feature should fire but doesn't
        positive_tokens = (token_labels == 1)
        main_active = (feature_acts > 0.1)  # Threshold for "active"
        
        # Absorbed tokens: should fire but doesn't
        absorbed_mask = positive_tokens & (~main_active)
        absorbed_indices = torch.where(absorbed_mask)[0]
        
        # For each absorbed token, find absorbing features
        absorbing_features_dict: Dict[int, List[float]] = {}  # feature_idx -> list of cosines
        absorbing_feature_ablations: Dict[int, List[float]] = {}  # feature_idx -> list of ablation effects
        
        for idx in absorbed_indices:
            token_id = tokens[idx].item()
            
            # Get all feature activations for this token
            all_feature_acts = self._get_all_feature_activations(tokens[idx:idx+1])
            
            # Find features that are active (besides main feature)
            active_features = torch.where(all_feature_acts[0] > 0.1)[0].tolist()
            active_features = [f for f in active_features if f != main_feature_idx]
            
            # Check cosine similarity with probe direction for each active feature
            for feat_idx in active_features:
                # Get decoder direction for this feature
                decoder_direction = self.transcoder.W_dec[feat_idx]
                
                # Compute cosine similarity with probe
                cosine_sim = F.cosine_similarity(
                    decoder_direction.unsqueeze(0),
                    self.probe_direction.unsqueeze(0),
                ).item()
                
                if abs(cosine_sim) >= self.cosine_threshold:
                    if feat_idx not in absorbing_features_dict:
                        absorbing_features_dict[feat_idx] = []
                        absorbing_feature_ablations[feat_idx] = []
                    
                    absorbing_features_dict[feat_idx].append(cosine_sim)
                    
                    # Optionally verify causality with ablation
                    if check_causality:
                        # Create simple ICL prompt for this token
                        prompt = self._create_icl_prompt(token_id, tokenizer)
                        ablation_effect = self._compute_ablation_for_token(
                            prompt, feat_idx, token_id
                        )
                        absorbing_feature_ablations[feat_idx].append(ablation_effect)
        
        # Filter absorbing features by causality if checked
        if check_causality:
            verified_absorbing_features = []
            verified_cosines = []
            verified_ablations = []
            
            for feat_idx, cosines in absorbing_features_dict.items():
                ablations = absorbing_feature_ablations.get(feat_idx, [])
                mean_ablation = np.mean(ablations) if ablations else 0.0
                
                if abs(mean_ablation) >= self.ablation_threshold:
                    verified_absorbing_features.append(feat_idx)
                    verified_cosines.append(np.mean(cosines))
                    verified_ablations.append(mean_ablation)
        else:
            verified_absorbing_features = list(absorbing_features_dict.keys())
            verified_cosines = [np.mean(cosines) for cosines in absorbing_features_dict.values()]
            verified_ablations = []
        
        # Calculate absorption rate
        num_absorbed = len(absorbed_indices)
        num_positive = positive_tokens.sum().item()
        absorption_rate = num_absorbed / num_positive if num_positive > 0 else 0.0
        
        # Calculate classification metrics
        predictions = main_active.float()
        labels = token_labels.float()
        
        tp = ((predictions == 1) & (labels == 1)).sum().item()
        fp = ((predictions == 1) & (labels == 0)).sum().item()
        fn = ((predictions == 0) & (labels == 1)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Get example absorbed tokens
        absorbed_token_ids = [tokens[i].item() for i in absorbed_indices[:10]]
        absorbed_token_strings = [tokenizer.decode([tid]) for tid in absorbed_token_ids]
        
        return AbsorptionAnalysis(
            main_feature_idx=main_feature_idx,
            letter="",  # Set by caller
            precision=precision,
            recall=recall,
            f1=f1,
            absorption_rate=absorption_rate,
            num_absorbed_tokens=num_absorbed,
            num_total_tokens=num_positive,
            absorbing_features=verified_absorbing_features,
            absorbing_feature_cosines=verified_cosines,
            absorbing_feature_ablations=verified_ablations,
            absorbed_token_ids=absorbed_token_ids,
            absorbed_token_strings=absorbed_token_strings,
        )
    
    def _get_feature_activations(
        self,
        tokens: torch.Tensor,
        feature_idx: int,
    ) -> torch.Tensor:
        """Get activations for a specific feature across tokens."""
        all_feature_acts = self._get_all_feature_activations(tokens)
        return all_feature_acts[:, feature_idx]
    
    def _get_all_feature_activations(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get all feature activations for given tokens.
        
        Args:
            tokens: Token IDs (n_samples,)
        
        Returns:
            Feature activations (n_samples, n_features)
        """
        with torch.no_grad():
            # Add batch dimension if needed
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(1)  # (n_samples, 1)
            
            # Run model to get activations at layer
            _, cache = self.model.run_with_cache(tokens, stop_at_layer=self.layer+1)
            
            # Get residual stream activations (use as proxy for mlp_in)
            hook_name = f"blocks.{self.layer}.hook_resid_pre"
            activations = cache[hook_name][:, 0, :]  # (batch, d_model)
            
            # Encode with transcoder
            features, _ = self.transcoder.compute_activations(activations)
            
            return features
    
    def _create_icl_prompt(self, token_id: int, tokenizer) -> torch.Tensor:
        """Create simple ICL prompt for ablation test."""
        # Simplified version - in practice you'd want proper ICL prompt
        token_str = tokenizer.decode([token_id])
        prompt = f"The word {token_str} has the first letter:"
        return tokenizer.encode(prompt, return_tensors='pt').to(self.device)
    
    def _compute_ablation_for_token(
        self,
        prompt_tokens: torch.Tensor,
        feature_idx: int,
        target_token_id: int,
    ) -> float:
        """Compute ablation effect for a feature on a specific token."""
        from .ablation import compute_ablation_effect
        
        # Token position is at the start (position 0 in single-token case)
        # Simplified - in practice you'd identify the correct position
        token_position = 0
        
        return compute_ablation_effect(
            self.model,
            self.transcoder,
            prompt_tokens,
            feature_idx,
            token_position,
            target_token_id,
            self.layer,
            use_integrated_gradients=True,
        )


def compute_absorption_score(
    model,
    transcoder,
    layer: int,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    probe_direction: torch.Tensor,
    main_feature_idx: int,
    tokenizer,
    cosine_threshold: float = 0.3,
    ablation_threshold: float = 0.01,
    check_causality: bool = True,
) -> AbsorptionAnalysis:
    """
    Convenience function to compute absorption score.
    
    Args:
        model: Language model
        transcoder: Matryoshka transcoder
        layer: Layer to analyze
        tokens: Token IDs
        labels: Binary labels for concept
        probe_direction: Linear probe direction
        main_feature_idx: Main feature that should track concept
        tokenizer: Tokenizer
        cosine_threshold: Cosine similarity threshold
        ablation_threshold: Ablation effect threshold
        check_causality: Whether to verify causality
    
    Returns:
        AbsorptionAnalysis
    """
    metric = AbsorptionMetric(
        model,
        transcoder,
        layer,
        probe_direction,
        cosine_threshold,
        ablation_threshold,
    )
    
    return metric.detect_absorption(
        tokens,
        labels,
        main_feature_idx,
        tokenizer,
        check_causality=check_causality,
    )

