"""
Real-time feature correlation monitoring during training.

Implements Canonical Correlation Analysis (CCA) and other methods
to detect and track feature redundancy during training.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
import json
import os
from pathlib import Path

class FeatureCorrelationMonitor:
    """
    Monitors feature correlations during training to detect redundancy.
    """
    
    def __init__(
        self,
        correlation_threshold: float = 0.8,
        window_size: int = 100,
        save_frequency: int = 1000,
        output_dir: str = "feature_correlation_logs"
    ):
        """
        Initialize feature correlation monitor.
        
        Args:
            correlation_threshold: Threshold for flagging high correlations
            window_size: Number of recent activations to consider
            save_frequency: How often to save correlation reports
            output_dir: Directory to save correlation logs
        """
        self.correlation_threshold = correlation_threshold
        self.window_size = window_size
        self.save_frequency = save_frequency
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Storage for activation history
        self.activation_history = deque(maxlen=window_size)
        self.step_history = deque(maxlen=window_size)
        
        # Correlation tracking
        self.correlation_history = []
        self.high_correlation_pairs = defaultdict(int)
        self.feature_statistics = defaultdict(list)
        
        # Step counter
        self.step_count = 0
    
    def update(self, activations: torch.Tensor, step: int):
        """
        Update monitor with new activation data.
        
        Args:
            activations: Feature activations [batch_size, num_features]
            step: Current training step
        """
        self.step_count = step
        
        # Store activations
        self.activation_history.append(activations.detach().cpu())
        self.step_history.append(step)
        
        # Compute correlations if we have enough data
        if len(self.activation_history) >= 10:
            self._compute_correlations()
        
        # Save reports periodically
        if step % self.save_frequency == 0:
            self._save_correlation_report()
    
    def _compute_correlations(self):
        """Compute feature correlations from recent history."""
        if len(self.activation_history) < 2:
            return
        
        # Concatenate recent activations
        recent_activations = torch.cat(list(self.activation_history), dim=0)
        
        # Compute correlation matrix
        correlation_matrix = torch.corrcoef(recent_activations.T)
        
        # Find high correlations
        high_corr_pairs = self._find_high_correlations(correlation_matrix)
        
        # Update statistics
        self.correlation_history.append({
            'step': self.step_count,
            'correlation_matrix': correlation_matrix.numpy(),
            'high_correlation_pairs': high_corr_pairs,
            'mean_correlation': torch.mean(torch.abs(correlation_matrix)).item(),
            'max_correlation': torch.max(torch.abs(correlation_matrix)).item()
        })
        
        # Track persistent high correlations
        for pair in high_corr_pairs:
            self.high_correlation_pairs[pair] += 1
    
    def _find_high_correlations(self, correlation_matrix: torch.Tensor) -> List[Tuple[int, int, float]]:
        """Find pairs of features with high correlations."""
        high_corr_pairs = []
        
        # Get upper triangle (avoid duplicates and diagonal)
        n_features = correlation_matrix.size(0)
        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr = correlation_matrix[i, j].item()
                if abs(corr) > self.correlation_threshold:
                    high_corr_pairs.append((i, j, corr))
        
        return high_corr_pairs
    
    def get_redundant_features(self, persistence_threshold: int = 5) -> List[Tuple[int, int, float]]:
        """
        Get features that consistently show high correlation.
        
        Args:
            persistence_threshold: Number of times a pair must show high correlation
            
        Returns:
            List of (feature1, feature2, average_correlation) tuples
        """
        redundant_pairs = []
        
        for (feat1, feat2), count in self.high_correlation_pairs.items():
            if count >= persistence_threshold:
                # Compute average correlation for this pair
                avg_corr = self._compute_average_correlation(feat1, feat2)
                redundant_pairs.append((feat1, feat2, avg_corr))
        
        # Sort by correlation strength
        redundant_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return redundant_pairs
    
    def _compute_average_correlation(self, feat1: int, feat2: int) -> float:
        """Compute average correlation between two features."""
        correlations = []
        
        for record in self.correlation_history:
            corr_matrix = torch.tensor(record['correlation_matrix'])
            if feat1 < corr_matrix.size(0) and feat2 < corr_matrix.size(1):
                correlations.append(corr_matrix[feat1, feat2].item())
        
        return np.mean(correlations) if correlations else 0.0
    
    def _save_correlation_report(self):
        """Save correlation report to disk."""
        report = {
            'step': self.step_count,
            'total_features': len(self.activation_history[0]) if self.activation_history else 0,
            'correlation_threshold': self.correlation_threshold,
            'redundant_features': self.get_redundant_features(),
            'correlation_statistics': self._compute_correlation_statistics()
        }
        
        report_path = os.path.join(self.output_dir, f"correlation_report_step_{self.step_count}.json")
        
        with open(report_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_report = self._convert_for_json(report)
            json.dump(json_report, f, indent=2)
        
        print(f"💾 Correlation report saved to: {report_path}")
    
    def _compute_correlation_statistics(self) -> Dict[str, float]:
        """Compute correlation statistics."""
        if not self.correlation_history:
            return {}
        
        mean_correlations = [record['mean_correlation'] for record in self.correlation_history]
        max_correlations = [record['max_correlation'] for record in self.correlation_history]
        
        return {
            'average_mean_correlation': np.mean(mean_correlations),
            'average_max_correlation': np.mean(max_correlations),
            'trend_mean_correlation': np.polyfit(range(len(mean_correlations)), mean_correlations, 1)[0],
            'trend_max_correlation': np.polyfit(range(len(max_correlations)), max_correlations, 1)[0]
        }
    
    def _convert_for_json(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-serializable objects for JSON."""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def get_correlation_summary(self) -> Dict[str, Any]:
        """Get summary of correlation analysis."""
        return {
            'step': self.step_count,
            'total_samples': len(self.activation_history),
            'redundant_features': self.get_redundant_features(),
            'correlation_statistics': self._compute_correlation_statistics(),
            'high_correlation_pairs_count': len(self.high_correlation_pairs)
        }

class CanonicalCorrelationAnalyzer:
    """
    Implements Canonical Correlation Analysis (CCA) for feature redundancy detection.
    """
    
    def __init__(self, n_components: int = 10):
        """
        Initialize CCA analyzer.
        
        Args:
            n_components: Number of canonical components to compute
        """
        self.n_components = n_components
    
    def compute_cca(self, X: torch.Tensor, Y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute Canonical Correlation Analysis between two sets of features.
        
        Args:
            X: First set of features [n_samples, n_features_X]
            Y: Second set of features [n_samples, n_features_Y]
            
        Returns:
            Dictionary with CCA results
        """
        # Center the data
        X_centered = X - torch.mean(X, dim=0, keepdim=True)
        Y_centered = Y - torch.mean(Y, dim=0, keepdim=True)
        
        # Compute covariance matrices
        C_xx = torch.mm(X_centered.T, X_centered) / (X.size(0) - 1)
        C_yy = torch.mm(Y_centered.T, Y_centered) / (Y.size(0) - 1)
        C_xy = torch.mm(X_centered.T, Y_centered) / (X.size(0) - 1)
        C_yx = C_xy.T
        
        # Solve generalized eigenvalue problem
        try:
            # Compute CCA using SVD approach
            C_xx_inv_sqrt = torch.inverse(torch.cholesky(C_xx + 1e-6 * torch.eye(C_xx.size(0))))
            C_yy_inv_sqrt = torch.inverse(torch.cholesky(C_yy + 1e-6 * torch.eye(C_yy.size(0))))
            
            # Compute correlation matrix
            R = torch.mm(torch.mm(C_xx_inv_sqrt.T, C_xy), C_yy_inv_sqrt)
            
            # SVD of correlation matrix
            U, S, V = torch.svd(R)
            
            # Canonical correlations
            canonical_correlations = S[:self.n_components]
            
            # Canonical vectors
            canonical_vectors_X = torch.mm(C_xx_inv_sqrt, U[:, :self.n_components])
            canonical_vectors_Y = torch.mm(C_yy_inv_sqrt, V[:, :self.n_components])
            
            return {
                'canonical_correlations': canonical_correlations,
                'canonical_vectors_X': canonical_vectors_X,
                'canonical_vectors_Y': canonical_vectors_Y,
                'explained_variance': canonical_correlations ** 2
            }
            
        except Exception as e:
            print(f"Error in CCA computation: {e}")
            return {
                'canonical_correlations': torch.zeros(self.n_components),
                'canonical_vectors_X': torch.zeros(X.size(1), self.n_components),
                'canonical_vectors_Y': torch.zeros(Y.size(1), self.n_components),
                'explained_variance': torch.zeros(self.n_components)
            }

def create_correlation_monitor(config: Dict) -> FeatureCorrelationMonitor:
    """
    Create correlation monitor based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured correlation monitor
    """
    return FeatureCorrelationMonitor(
        correlation_threshold=config.get('correlation_threshold', 0.8),
        window_size=config.get('correlation_window_size', 100),
        save_frequency=config.get('correlation_save_frequency', 1000),
        output_dir=config.get('correlation_output_dir', 'feature_correlation_logs')
    )
