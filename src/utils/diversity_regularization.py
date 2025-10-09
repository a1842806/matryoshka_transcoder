"""
Diversity regularization utilities to reduce feature duplication.

Based on latest research on feature redundancy mitigation including:
- Centered Kernel Alignment (CKA) for measuring feature similarity
- Orthogonality penalties for decoder weights
- Correlation penalties for activations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple

class FeatureDiversityRegularizer:
    """
    Implements multiple diversity regularization strategies to reduce feature duplication.
    """
    
    def __init__(
        self,
        orthogonality_weight: float = 0.01,
        correlation_weight: float = 0.005,
        cka_weight: float = 0.001,
        position_diversity_weight: float = 0.01
    ):
        """
        Initialize diversity regularizer.
        
        Args:
            orthogonality_weight: Weight for decoder orthogonality penalty
            correlation_weight: Weight for activation correlation penalty
            cka_weight: Weight for Centered Kernel Alignment penalty
            position_diversity_weight: Weight for position diversity penalty
        """
        self.orthogonality_weight = orthogonality_weight
        self.correlation_weight = correlation_weight
        self.cka_weight = cka_weight
        self.position_diversity_weight = position_diversity_weight
        
        # Track feature statistics
        self.feature_activations_history = []
        self.position_activations_history = []
    
    def compute_orthogonality_penalty(self, decoder_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute orthogonality penalty for decoder weights.
        
        Encourages decoder weight vectors to be orthogonal to each other.
        """
        # Normalize weights
        normalized_weights = F.normalize(decoder_weights, p=2, dim=1)
        
        # Compute gram matrix (cosine similarities)
        gram_matrix = torch.mm(normalized_weights, normalized_weights.t())
        
        # Remove diagonal (self-similarity)
        mask = torch.eye(gram_matrix.size(0), device=gram_matrix.device).bool()
        off_diagonal = gram_matrix.masked_select(~mask)
        
        # Penalty is the mean squared off-diagonal elements
        penalty = torch.mean(off_diagonal ** 2)
        
        return penalty
    
    def compute_correlation_penalty(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Compute correlation penalty for feature activations.
        
        Penalizes high correlations between feature activations.
        """
        if activations.size(0) < 2:
            return torch.tensor(0.0, device=activations.device)
        
        # Compute correlation matrix
        correlation_matrix = torch.corrcoef(activations)
        
        # Remove diagonal and NaN values
        mask = torch.eye(correlation_matrix.size(0), device=correlation_matrix.device).bool()
        off_diagonal = correlation_matrix.masked_select(~mask)
        
        # Remove NaN values
        valid_correlations = off_diagonal[torch.isfinite(off_diagonal)]
        
        if len(valid_correlations) == 0:
            return torch.tensor(0.0, device=activations.device)
        
        # Penalty is the mean squared correlation
        penalty = torch.mean(valid_correlations ** 2)
        
        return penalty
    
    def compute_position_diversity_penalty(self, activations: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute position diversity penalty to reduce BOS bias.
        
        Encourages features to activate at diverse positions.
        """
        if activations.size(0) == 0:
            return torch.tensor(0.0, device=activations.device)
        
        # Find top-k activated features
        top_k = min(10, activations.size(0))
        top_activations, top_indices = torch.topk(activations, top_k)
        
        # Get positions of top features
        top_positions = positions[top_indices]
        
        # Penalty for having too many features at position 0 (BOS bias)
        bos_count = torch.sum(top_positions == 0).float()
        bos_penalty = bos_count / top_k
        
        # Penalty for position clustering (low position diversity)
        position_variance = torch.var(top_positions.float())
        diversity_penalty = 1.0 / (1.0 + position_variance)
        
        total_penalty = bos_penalty + diversity_penalty
        
        return total_penalty
    
    def compute_cka_penalty(self, activations1: torch.Tensor, activations2: torch.Tensor) -> torch.Tensor:
        """
        Compute Centered Kernel Alignment (CKA) penalty.
        
        Measures similarity between two sets of feature activations.
        """
        if activations1.size(0) != activations2.size(0):
            return torch.tensor(0.0, device=activations1.device)
        
        # Center the activations
        activations1_centered = activations1 - torch.mean(activations1, dim=0, keepdim=True)
        activations2_centered = activations2 - torch.mean(activations2, dim=0, keepdim=True)
        
        # Compute kernel matrices
        kernel1 = torch.mm(activations1_centered, activations1_centered.t())
        kernel2 = torch.mm(activations2_centered, activations2_centered.t())
        
        # Compute CKA
        hsic = torch.sum(kernel1 * kernel2)
        normalization = torch.sqrt(torch.sum(kernel1 * kernel1) * torch.sum(kernel2 * kernel2))
        
        if normalization == 0:
            return torch.tensor(0.0, device=activations1.device)
        
        cka = hsic / normalization
        
        # Return penalty (higher CKA = higher penalty)
        return cka
    
    def compute_total_penalty(
        self,
        decoder_weights: torch.Tensor,
        activations: torch.Tensor,
        positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total diversity penalty.
        
        Returns:
            total_penalty: Combined diversity penalty
            penalty_breakdown: Dictionary with individual penalty components
        """
        penalty_breakdown = {}
        
        # Orthogonality penalty
        if self.orthogonality_weight > 0:
            ortho_penalty = self.compute_orthogonality_penalty(decoder_weights)
            penalty_breakdown['orthogonality'] = ortho_penalty
        
        # Correlation penalty
        if self.correlation_weight > 0:
            corr_penalty = self.compute_correlation_penalty(activations)
            penalty_breakdown['correlation'] = corr_penalty
        
        # Position diversity penalty
        if self.position_diversity_weight > 0 and positions is not None:
            pos_penalty = self.compute_position_diversity_penalty(activations, positions)
            penalty_breakdown['position_diversity'] = pos_penalty
        
        # Compute weighted total
        total_penalty = torch.tensor(0.0, device=decoder_weights.device)
        
        if 'orthogonality' in penalty_breakdown:
            total_penalty += self.orthogonality_weight * penalty_breakdown['orthogonality']
        
        if 'correlation' in penalty_breakdown:
            total_penalty += self.correlation_weight * penalty_breakdown['correlation']
        
        if 'position_diversity' in penalty_breakdown:
            total_penalty += self.position_diversity_weight * penalty_breakdown['position_diversity']
        
        return total_penalty, penalty_breakdown
    
    def update_statistics(self, activations: torch.Tensor, positions: torch.Tensor):
        """Update internal statistics for adaptive regularization."""
        self.feature_activations_history.append(activations.detach().cpu())
        self.position_activations_history.append(positions.detach().cpu())
        
        # Keep only recent history
        max_history = 100
        if len(self.feature_activations_history) > max_history:
            self.feature_activations_history = self.feature_activations_history[-max_history:]
            self.position_activations_history = self.position_activations_history[-max_history:]

class AdaptiveDiversityRegularizer(FeatureDiversityRegularizer):
    """
    Adaptive diversity regularizer that adjusts weights based on observed redundancy.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.redundancy_threshold = 0.8
        self.adaptive_weights = True
    
    def compute_adaptive_penalty(
        self,
        decoder_weights: torch.Tensor,
        activations: torch.Tensor,
        positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute adaptive diversity penalty based on observed redundancy.
        """
        # Compute base penalties
        total_penalty, penalty_breakdown = self.compute_total_penalty(
            decoder_weights, activations, positions
        )
        
        # Adjust weights based on history if available
        if len(self.feature_activations_history) > 10:
            # Compute recent correlation trends
            recent_activations = torch.stack(self.feature_activations_history[-10:])
            recent_correlation = self.compute_correlation_penalty(recent_activations.mean(0))
            
            # Increase correlation penalty if high redundancy detected
            if recent_correlation > self.redundancy_threshold:
                penalty_breakdown['correlation'] *= 2.0
                total_penalty += self.correlation_weight * penalty_breakdown['correlation']
        
        return total_penalty, penalty_breakdown

def create_diversity_regularizer(config: Dict) -> FeatureDiversityRegularizer:
    """
    Create diversity regularizer based on configuration.
    
    Args:
        config: Configuration dictionary with regularization weights
        
    Returns:
        Configured diversity regularizer
    """
    regularizer_type = config.get('diversity_regularizer_type', 'standard')
    
    if regularizer_type == 'adaptive':
        return AdaptiveDiversityRegularizer(
            orthogonality_weight=config.get('orthogonality_weight', 0.01),
            correlation_weight=config.get('correlation_weight', 0.005),
            cka_weight=config.get('cka_weight', 0.001),
            position_diversity_weight=config.get('position_diversity_weight', 0.01)
        )
    else:
        return FeatureDiversityRegularizer(
            orthogonality_weight=config.get('orthogonality_weight', 0.01),
            correlation_weight=config.get('correlation_weight', 0.005),
            cka_weight=config.get('cka_weight', 0.001),
            position_diversity_weight=config.get('position_diversity_weight', 0.01)
        )
