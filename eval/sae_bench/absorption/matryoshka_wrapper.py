"""
Wrapper for Matryoshka Transcoder to work with absorption evaluation.

Provides a unified interface compatible with SAE Bench evaluation framework.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from pathlib import Path
import sys
import os

# Add project root and src to path BEFORE any other imports
_current_file = os.path.abspath(__file__)
_project_root = os.path.abspath(os.path.join(os.path.dirname(_current_file), "..", "..", ".."))
_src_path = os.path.join(_project_root, "src")

# Ensure src is in path for relative imports in sae.py
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Now safe to import
try:
    from models.sae import MatryoshkaTranscoder
except ImportError:
    # Fallback to full path
    from src.models.sae import MatryoshkaTranscoder


class MatryoshkaTranscoderWrapper:
    """
    Wrapper for Matryoshka Transcoder to provide SAE-like interface.
    
    This wrapper ensures the transcoder works seamlessly with the
    absorption evaluation code, which expects encode() and decode() methods.
    """
    
    def __init__(
        self,
        transcoder: MatryoshkaTranscoder,
        layer: int,
        prefix_level: Optional[int] = None,
    ):
        """
        Args:
            transcoder: Matryoshka transcoder model
            layer: Layer number this transcoder operates on
            prefix_level: Which prefix size to use (0-4), None uses full size
        """
        self.transcoder = transcoder
        self.layer = layer
        self.prefix_level = prefix_level
        
        # Determine active feature count based on prefix level
        if prefix_level is None:
            self.n_features = transcoder.prefix_sizes[-1]  # Full size
            self.active_features = slice(None)
        else:
            self.n_features = transcoder.prefix_sizes[prefix_level]
            self.active_features = slice(0, self.n_features)
        
        self.d_model = transcoder.source_act_size
        self.device = next(transcoder.parameters()).device
        self.dtype = next(transcoder.parameters()).dtype
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode activations to sparse features.
        
        Args:
            x: Input activations (batch, d_model) or (batch, seq, d_model)
        
        Returns:
            Sparse feature activations (batch, n_features) or (batch, seq, n_features)
        """
        original_shape = x.shape
        is_3d = len(original_shape) == 3
        
        if is_3d:
            batch, seq, d_model = original_shape
            x = x.view(batch * seq, d_model)
        
        # Get sparse features from transcoder
        with torch.no_grad():
            _, features_topk = self.transcoder.compute_activations(x)
        
        # Apply prefix masking if needed
        if self.prefix_level is not None:
            features = features_topk[:, self.active_features]
        else:
            features = features_topk
        
        if is_3d:
            features = features.view(batch, seq, -1)
        
        return features
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode features back to activation space.
        
        Args:
            features: Feature activations (batch, n_features) or (batch, seq, n_features)
        
        Returns:
            Reconstructed activations (batch, d_model) or (batch, seq, d_model)
        """
        original_shape = features.shape
        is_3d = len(original_shape) == 3
        
        if is_3d:
            batch, seq, n_feat = original_shape
            features = features.view(batch * seq, n_feat)
        
        # Pad features if using prefix
        if self.prefix_level is not None:
            full_features = torch.zeros(
                features.shape[0],
                self.transcoder.prefix_sizes[-1],
                device=features.device,
                dtype=features.dtype,
            )
            full_features[:, self.active_features] = features
            features = full_features
        
        # Decode through transcoder
        with torch.no_grad():
            reconstructed = features @ self.transcoder.W_dec + self.transcoder.b_dec
        
        if is_3d:
            reconstructed = reconstructed.view(batch, seq, -1)
        
        return reconstructed
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode then decode.
        
        Args:
            x: Input activations
        
        Returns:
            Tuple of (features, reconstruction)
        """
        features = self.encode(x)
        reconstruction = self.decode(features)
        return features, reconstruction
    
    @property
    def W_enc(self) -> torch.Tensor:
        """Encoder weights."""
        return self.transcoder.W_enc[:, self.active_features]
    
    @property
    def W_dec(self) -> torch.Tensor:
        """Decoder weights."""
        return self.transcoder.W_dec[self.active_features, :]
    
    @property
    def b_enc(self) -> torch.Tensor:
        """Encoder bias."""
        return self.transcoder.b_enc[self.active_features]
    
    @property
    def b_dec(self) -> torch.Tensor:
        """Decoder bias."""
        return self.transcoder.b_dec
    
    def to(self, device):
        """Move to device."""
        self.transcoder = self.transcoder.to(device)
        self.device = device
        return self
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        layer: int,
        prefix_level: Optional[int] = None,
        device: str = "cuda",
    ) -> "MatryoshkaTranscoderWrapper":
        """
        Load transcoder from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            layer: Layer number
            prefix_level: Prefix level to use
            device: Device to load on
        
        Returns:
            MatryoshkaTranscoderWrapper instance
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract config and model state
        if 'config' in checkpoint:
            cfg = checkpoint['config']
        else:
            raise ValueError("Checkpoint must contain 'config' key")
        
        # Create transcoder
        transcoder = MatryoshkaTranscoder(cfg)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            transcoder.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            transcoder.load_state_dict(checkpoint['state_dict'])
        else:
            raise ValueError("Checkpoint must contain 'model_state_dict' or 'state_dict'")
        
        transcoder = transcoder.to(device)
        transcoder.eval()
        
        return cls(transcoder, layer, prefix_level)
    
    def get_feature_stats(self) -> Dict[str, Any]:
        """Get statistics about the transcoder."""
        return {
            'layer': self.layer,
            'n_features': self.n_features,
            'total_features': self.transcoder.prefix_sizes[-1],
            'prefix_level': self.prefix_level,
            'prefix_sizes': self.transcoder.prefix_sizes,
            'd_model': self.d_model,
            'device': str(self.device),
            'dtype': str(self.dtype),
        }


def load_matryoshka_transcoder(
    model_name: str,
    layer: int,
    results_dir: str = "results",
    checkpoint_name: str = "final.pt",
    prefix_level: Optional[int] = None,
    device: str = "cuda",
) -> MatryoshkaTranscoderWrapper:
    """
    Load a trained Matryoshka transcoder.
    
    Args:
        model_name: Model name (e.g., "gemma-2-2b")
        layer: Layer number
        results_dir: Base results directory
        checkpoint_name: Name of checkpoint file
        prefix_level: Prefix level to use (None = full)
        device: Device to load on
    
    Returns:
        MatryoshkaTranscoderWrapper
    """
    # Find checkpoint path
    checkpoint_path = Path(results_dir) / model_name / f"layer{layer}"
    
    # Look for checkpoint in subdirectories
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No results found for {model_name} layer {layer}")
    
    # Find the most recent training run
    subdirs = [d for d in checkpoint_path.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No training runs found in {checkpoint_path}")
    
    # Use the most recent (or specified) run
    latest_dir = max(subdirs, key=lambda d: d.stat().st_mtime)
    checkpoint_file = latest_dir / "checkpoints" / checkpoint_name
    
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    
    print(f"Loading transcoder from: {checkpoint_file}")
    
    return MatryoshkaTranscoderWrapper.from_checkpoint(
        str(checkpoint_file),
        layer,
        prefix_level,
        device,
    )

