"""Configuration for Absorption Score evaluation."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class AbsorptionConfig:
    """Configuration for absorption score evaluation."""
    
    # Model configuration
    model_name: str = "gemma-2-2b"
    layers_to_eval: List[int] = field(default_factory=lambda: list(range(18)))  # Layers 0-17
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    
    # Dataset configuration
    num_samples: int = 10000  # Number of vocabulary samples for probe training
    batch_size: int = 10  # Batch size for absorption calculation
    
    # First-letter task configuration
    letters: List[str] = field(default_factory=lambda: list("abcdefghijklmnopqrstuvwxyz"))
    
    # Linear probe training
    probe_lr: float = 0.01
    probe_epochs: int = 100
    
    # Output
    results_dir: str = "results/absorption"
    verbose: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if len(self.layers_to_eval) == 0:
            raise ValueError("Must specify at least one layer to evaluate")
