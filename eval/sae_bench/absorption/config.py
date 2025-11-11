"""Configuration for Absorption Score evaluation."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AbsorptionConfig:
    """Configuration for absorption score evaluation."""
    
    # Model configuration
    model_name: str = "gemma-2-2b"
    layers_to_eval: List[int] = field(default_factory=lambda: list(range(18)))  # Layers 0-17
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    
    # Dataset configuration
    dataset_path: str = "HuggingFaceFW/fineweb-edu"
    dataset_split: str = "train"
    num_samples: int = 10000  # Number of vocabulary samples for evaluation
    batch_size: int = 32
    seq_len: int = 128
    
    # First-letter task configuration
    letters: List[str] = field(default_factory=lambda: list("abcdefghijklmnopqrstuvwxyz"))
    min_tokens_per_letter: int = 100  # Minimum tokens starting with each letter
    
    # Linear probe training
    probe_lr: float = 1e-3
    probe_epochs: int = 10
    probe_l1_penalty: float = 0.01  # For k-sparse probe selection
    
    # Feature selection
    selection_method: str = "cosine_similarity"  # or "k_sparse_probe"
    k_sparse: int = 1  # Number of features to select for k-sparse probing
    cosine_sim_threshold: float = 0.5  # Minimum cosine similarity
    
    # Absorption detection
    absorption_threshold: float = 0.3  # Cosine similarity threshold for absorption
    ablation_threshold: float = 0.01  # Minimum ablation effect to consider causal
    use_integrated_gradients: bool = True  # Use IG approximation for speed
    
    # Metrics
    calculate_precision_recall: bool = True
    calculate_f1: bool = True
    calculate_absorption_rate: bool = True
    
    # Output
    results_dir: str = "results/absorption"
    save_detailed_results: bool = True
    verbose: bool = True
    
    # ICL prompt template for first-letter task
    icl_template: str = "{word1} has the first letter: {letter1}\n{word2} has the first letter: {letter2}\n{word3} has the first letter:"
    
    def __post_init__(self):
        """Validate configuration."""
        if len(self.layers_to_eval) == 0:
            raise ValueError("Must specify at least one layer to evaluate")
        
        if self.selection_method not in ["cosine_similarity", "k_sparse_probe"]:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
        
        if self.num_samples < self.min_tokens_per_letter * len(self.letters):
            self.num_samples = self.min_tokens_per_letter * len(self.letters)

