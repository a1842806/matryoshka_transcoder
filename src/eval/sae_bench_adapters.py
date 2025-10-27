import torch
from typing import Optional

from src.models.sae import MatryoshkaTranscoder
from src.adapters.gemma_scope_transcoder import GemmaScopeTranscoderAdapter


class OurTranscoderAdapter(torch.nn.Module):
    """
    Adapter exposing a common interface for our Matryoshka transcoder:
    - forward(x): returns reconstruction of target activations
    - encode(x): returns sparse features (when available)
    - get_feature_vectors(): returns decoder feature vectors for absorption metrics
    """

    def __init__(self, model: MatryoshkaTranscoder):
        super().__init__()
        self.model = model.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode then decode to avoid recomputing losses here
        acts = self.model.encode(x)
        return self.model.decode(acts)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode(x)

    @torch.no_grad()
    def get_feature_vectors(self) -> torch.Tensor:
        # Return normalized decoder weights as feature vectors: (dict_size, d)
        W = self.model.W_dec.detach()
        return torch.nn.functional.normalize(W, p=2, dim=-1)


class GoogleTranscoderAdapterWrapper(torch.nn.Module):
    """
    Adapter wrapper around Gemma Scope transcoder artifacts.
    Provides:
    - forward(x): linear mapping (mlp_in -> mlp_out)
    - get_feature_vectors(): returns None (no sparse features available)
    """

    def __init__(self, scope_adapter: GemmaScopeTranscoderAdapter):
        super().__init__()
        self.scope = scope_adapter.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scope(x)

    @torch.no_grad()
    def get_feature_vectors(self) -> Optional[torch.Tensor]:
        return None


