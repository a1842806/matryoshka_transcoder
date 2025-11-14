"""Configuration for Sparse Probing evaluation."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SparseProbingConfig:
    """Configuration for sparse probing evaluation."""
    
    # Model configuration
    model_name: str = "gemma-2-2b"
    layers_to_eval: List[int] = field(default_factory=lambda: list(range(18)))  # Layers 0-17
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    
    # Dataset configuration
    num_samples: int = 10000  # Number of vocabulary samples for evaluation
    batch_size: int = 32
    
    # First-letter task configuration (reused from absorption)
    letters: List[str] = field(default_factory=lambda: list("abcdefghijklmnopqrstuvwxyz"))
    min_tokens_per_letter: int = 100  # Minimum tokens starting with each letter
    
    # Feature selection
    max_features: Optional[int] = None  # If None, evaluate all features. If set, evaluate top N by activation frequency
    min_feature_activation_rate: float = 0.01  # Minimum fraction of samples where feature activates
    
    # Probe training
    probe_lr: float = 1e-3
    probe_epochs: int = 10
    probe_l1_penalty: float = 0.01
    train_test_split: float = 0.8  # Fraction of data for training
    
    # Evaluation
    activation_threshold: float = 0.0  # Threshold for feature activation (0 = any activation)
    top_k_features: int = 10  # Number of top features to report per concept
    
    # Output
    results_dir: str = "results/sparse_probing"
    save_detailed_results: bool = True
    verbose: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if len(self.layers_to_eval) == 0:
            raise ValueError("Must specify at least one layer to evaluate")
        
        if not 0 < self.train_test_split < 1:
            raise ValueError("train_test_split must be between 0 and 1")
        
        if self.num_samples < self.min_tokens_per_letter * len(self.letters):
            self.num_samples = self.min_tokens_per_letter * len(self.letters)

