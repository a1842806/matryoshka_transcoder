"""
Position-stratified sampling to reduce BOS bias and promote feature diversity.

This module implements sampling strategies that ensure features activate
at diverse positions rather than clustering at sequence beginnings.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

class PositionStratifiedSampler:
    """
    Implements position-stratified sampling to reduce BOS bias.
    """
    
    def __init__(
        self,
        position_bins: int = 10,
        min_samples_per_bin: int = 2,
        bos_penalty_factor: float = 0.5,
        max_bos_ratio: float = 0.3
    ):
        """
        Initialize position-stratified sampler.
        
        Args:
            position_bins: Number of position bins for stratification
            min_samples_per_bin: Minimum samples required per position bin
            bos_penalty_factor: Factor to reduce BOS position sampling
            max_bos_ratio: Maximum ratio of samples allowed at position 0
        """
        self.position_bins = position_bins
        self.min_samples_per_bin = min_samples_per_bin
        self.bos_penalty_factor = bos_penalty_factor
        self.max_bos_ratio = max_bos_ratio
        
        # Track sampling statistics
        self.position_counts = defaultdict(int)
        self.total_samples = 0
    
    def compute_position_weights(self, sequence_length: int) -> torch.Tensor:
        """
        Compute sampling weights for different positions.
        
        Reduces weight for BOS positions and promotes diversity.
        """
        weights = torch.ones(sequence_length)
        
        # Apply BOS penalty
        weights[0] *= self.bos_penalty_factor
        
        # Apply position diversity bonus
        for i in range(1, sequence_length):
            # Increase weight for underrepresented positions
            if self.position_counts[i] < self.min_samples_per_bin:
                weights[i] *= 2.0
        
        return weights
    
    def stratify_samples(
        self,
        activations: torch.Tensor,
        positions: torch.Tensor,
        max_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stratify samples across positions to reduce bias.
        
        Args:
            activations: Feature activation values
            positions: Corresponding positions
            max_samples: Maximum number of samples to select
            
        Returns:
            Selected activations and positions
        """
        if len(activations) <= max_samples:
            return activations, positions
        
        # Group by position bins
        position_bins = self._create_position_bins(positions)
        
        # Select samples from each bin
        selected_indices = []
        samples_per_bin = max_samples // len(position_bins)
        
        for bin_positions in position_bins:
            if len(bin_positions) == 0:
                continue
            
            # Get activations for this position bin
            bin_activations = activations[bin_positions]
            
            # Select top samples from this bin
            bin_samples = min(samples_per_bin, len(bin_positions))
            _, top_indices = torch.topk(bin_activations, bin_samples)
            
            # Add to selected samples
            selected_indices.extend(bin_positions[top_indices])
        
        # If we need more samples, fill from remaining high-activation samples
        if len(selected_indices) < max_samples:
            remaining_indices = set(range(len(activations))) - set(selected_indices)
            remaining_activations = activations[list(remaining_indices)]
            
            additional_samples = max_samples - len(selected_indices)
            _, additional_indices = torch.topk(remaining_activations, additional_samples)
            
            selected_indices.extend([list(remaining_indices)[i] for i in additional_indices])
        
        selected_indices = torch.tensor(selected_indices[:max_samples])
        
        return activations[selected_indices], positions[selected_indices]
    
    def _create_position_bins(self, positions: torch.Tensor) -> List[List[int]]:
        """Create position bins for stratification."""
        max_position = positions.max().item()
        bin_size = max(1, max_position // self.position_bins)
        
        bins = [[] for _ in range(self.position_bins)]
        
        for i, pos in enumerate(positions):
            bin_idx = min(pos.item() // bin_size, self.position_bins - 1)
            bins[bin_idx].append(i)
        
        return bins
    
    def update_statistics(self, positions: torch.Tensor):
        """Update position sampling statistics."""
        for pos in positions:
            self.position_counts[pos.item()] += 1
            self.total_samples += 1
    
    def get_position_distribution(self) -> Dict[int, float]:
        """Get current position distribution."""
        if self.total_samples == 0:
            return {}
        
        return {
            pos: count / self.total_samples
            for pos, count in self.position_counts.items()
        }

class AdaptivePositionSampler(PositionStratifiedSampler):
    """
    Adaptive position sampler that adjusts sampling based on observed bias.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bias_threshold = 0.5
        self.adaptive_penalty = True
    
    def compute_adaptive_weights(self, sequence_length: int) -> torch.Tensor:
        """Compute adaptive position weights based on observed bias."""
        weights = self.compute_position_weights(sequence_length)
        
        if self.adaptive_penalty and self.total_samples > 100:
            # Check for BOS bias
            bos_count = self.position_counts.get(0, 0)
            bos_ratio = bos_count / self.total_samples
            
            if bos_ratio > self.max_bos_ratio:
                # Increase BOS penalty
                weights[0] *= 0.1
                
                # Increase weights for underrepresented positions
                for pos in range(1, sequence_length):
                    pos_count = self.position_counts.get(pos, 0)
                    pos_ratio = pos_count / self.total_samples
                    
                    if pos_ratio < 0.01:  # Very underrepresented
                        weights[pos] *= 3.0
        
        return weights

class NonMaximumSuppressionSampler:
    """
    Non-maximum suppression sampler to reduce redundant samples.
    """
    
    def __init__(self, overlap_threshold: float = 0.8, position_tolerance: int = 1):
        """
        Initialize NMS sampler.
        
        Args:
            overlap_threshold: Threshold for considering samples as overlapping
            position_tolerance: Position tolerance for overlap detection
        """
        self.overlap_threshold = overlap_threshold
        self.position_tolerance = position_tolerance
    
    def suppress_overlapping_samples(
        self,
        activations: torch.Tensor,
        positions: torch.Tensor,
        contexts: List[str],
        max_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Apply non-maximum suppression to remove overlapping samples.
        
        Args:
            activations: Feature activation values
            positions: Corresponding positions
            contexts: Corresponding context strings
            max_samples: Maximum number of samples to keep
            
        Returns:
            Suppressed activations, positions, and contexts
        """
        if len(activations) <= max_samples:
            return activations, positions, contexts
        
        # Sort by activation strength (descending)
        sorted_indices = torch.argsort(activations, descending=True)
        
        selected_indices = []
        suppressed_indices = set()
        
        for idx in sorted_indices:
            if len(selected_indices) >= max_samples:
                break
            
            if idx.item() in suppressed_indices:
                continue
            
            # Check for overlap with already selected samples
            is_overlapping = False
            
            for selected_idx in selected_indices:
                if self._is_overlapping(
                    positions[idx], positions[selected_idx],
                    contexts[idx], contexts[selected_idx]
                ):
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                selected_indices.append(idx.item())
            else:
                suppressed_indices.add(idx.item())
        
        selected_indices = torch.tensor(selected_indices)
        
        return (
            activations[selected_indices],
            positions[selected_indices],
            [contexts[i] for i in selected_indices]
        )
    
    def _is_overlapping(
        self,
        pos1: int,
        pos2: int,
        context1: str,
        context2: str
    ) -> bool:
        """Check if two samples are overlapping."""
        # Position overlap
        if abs(pos1 - pos2) <= self.position_tolerance:
            # Context overlap (simple word overlap check)
            words1 = set(context1.lower().split())
            words2 = set(context2.lower().split())
            
            if len(words1) > 0 and len(words2) > 0:
                overlap = len(words1.intersection(words2)) / len(words1.union(words2))
                if overlap > self.overlap_threshold:
                    return True
        
        return False

def create_position_sampler(config: Dict) -> PositionStratifiedSampler:
    """
    Create position sampler based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured position sampler
    """
    sampler_type = config.get('position_sampler_type', 'stratified')
    
    if sampler_type == 'adaptive':
        return AdaptivePositionSampler(
            position_bins=config.get('position_bins', 10),
            min_samples_per_bin=config.get('min_samples_per_bin', 2),
            bos_penalty_factor=config.get('bos_penalty_factor', 0.5),
            max_bos_ratio=config.get('max_bos_ratio', 0.3)
        )
    elif sampler_type == 'nms':
        return NonMaximumSuppressionSampler(
            overlap_threshold=config.get('overlap_threshold', 0.8),
            position_tolerance=config.get('position_tolerance', 1)
        )
    else:
        return PositionStratifiedSampler(
            position_bins=config.get('position_bins', 10),
            min_samples_per_bin=config.get('min_samples_per_bin', 2),
            bos_penalty_factor=config.get('bos_penalty_factor', 0.5),
            max_bos_ratio=config.get('max_bos_ratio', 0.3)
        )
