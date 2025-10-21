import os
import glob
import time
from typing import Optional, Dict, Any

import torch


class GemmaScopeTranscoderAdapter(torch.nn.Module):
    """
    Lightweight adapter to use Google's Gemma Scope 2B pretrained transcoders
    (e.g., MLP_in → MLP_out mappings) inside this codebase.

    This class tries to load a weight matrix (and optional bias) from a local
    snapshot of the HF repo and exposes a simple forward(x_in) → x_out_hat.

    Notes:
    - If the HF artifact exposes only a direct mapping without sparse latents,
      sparsity/feature metrics are not applicable and should be marked N/A.
    - Input normalization (e.g., Gemma RMSNorm) is not performed here; the
      caller should provide the exact activations (e.g., resid_mid vs ln2).
    """

    def __init__(
        self,
        *,
        layer: int,
        repo_dir: str,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        allow_patterns: Optional[list] = None,
    ) -> None:
        super().__init__()
        self.layer = int(layer)
        self.device = device
        self.dtype = dtype

        # Defer actual weight loading to a helper so we can give a clearer error
        state = self._discover_and_load(repo_dir, allow_patterns)
        if state is None:
            raise FileNotFoundError(
                f"Could not find a Gemma Scope transcoder for layer {self.layer} under {repo_dir}."
            )

        # Detect weight and bias
        W, b = self._extract_linear_params(state)
        if W is None:
            raise RuntimeError("No usable linear weights found in the Gemma Scope artifact.")

        # Register as parameters/buffers so .to(device/dtype) works uniformly
        self.register_buffer("W", W.to(device=self.device, dtype=self.dtype))
        self.register_buffer("b", b.to(device=self.device, dtype=self.dtype) if b is not None else None)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """
        Apply the loaded linear mapping. Supports weight in either layout:
        - (in_dim, out_dim): y = x @ W + b
        - (out_dim, in_dim): y = x @ W.T + b
        """
        if self.W.dim() != 2:
            raise RuntimeError(f"Unexpected W dim: {self.W.shape}")

        in_dim = x_in.shape[-1]
        if self.W.shape[0] == in_dim:
            y = x_in @ self.W
        elif self.W.shape[1] == in_dim:
            y = x_in @ self.W.t()
        else:
            # Try the alternative if batch-major transposed tensors were saved
            if self.W.shape[0] == in_dim:
                y = x_in @ self.W
            else:
                raise RuntimeError(
                    f"Input dim {in_dim} not compatible with W {tuple(self.W.shape)}"
                )

        if self.b is not None:
            y = y + self.b
        return y

    def _discover_and_load(
        self, repo_dir: str, allow_patterns: Optional[list]
    ) -> Optional[Dict[str, Any]]:
        """
        Heuristically discover files for the given layer and load state.

        We accept multiple common formats: .pt, .pth, .bin, .safetensors (via torch)
        when possible. If multiple files are present, prefer ones that match the
        layer number.
        """
        candidates = []
        patterns = ["**/*.pt", "**/*.pth", "**/*.bin", "**/*.safetensors"]
        if allow_patterns:
            patterns = allow_patterns

        for pat in patterns:
            for path in glob.glob(os.path.join(repo_dir, pat), recursive=True):
                base = os.path.basename(path).lower()
                # Prefer files that obviously correspond to this layer
                score = 0
                if f"{self.layer}" in base:
                    score += 2
                if any(k in base for k in ["transcoder", "mlp", "scope"]):
                    score += 1
                candidates.append((score, path))

        if not candidates:
            return None

        # Highest score first, then latest mtime
        candidates.sort(key=lambda x: (x[0], os.path.getmtime(x[1])), reverse=True)

        last_err = None
        for _, path in candidates:
            try:
                # Try torch.load first (works for .pt/.pth and some .bin)
                state = torch.load(path, map_location="cpu")
                if isinstance(state, dict):
                    return state
            except Exception as e:
                last_err = e
                continue

        # No luck
        if last_err is not None:
            print(f"Warning: failed to load any candidate: {last_err}")
        return None

    @staticmethod
    def _extract_linear_params(state: Dict[str, Any]) -> (Optional[torch.Tensor], Optional[torch.Tensor]):
        """
        Extract a single linear layer weight and bias from a generic state dict.
        Tries common key names from various training setups.
        """
        # Direct tensors (rare)
        if isinstance(state, torch.Tensor):
            return state, None

        if not isinstance(state, dict):
            return None, None

        # If state looks like a nested dict (e.g., {"model": {...}}), flatten one level
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        elif "model" in state and isinstance(state["model"], dict):
            state = state["model"]

        # Common patterns
        weight_keys = [
            "weight", "W", "W_out", "W_linear", "W_dec", "W_proj", "proj.weight",
        ]
        bias_keys = ["bias", "b", "b_out", "b_dec", "proj.bias"]

        W = None
        b = None

        for k in weight_keys:
            if k in state and isinstance(state[k], torch.Tensor):
                W = state[k]
                break

        if W is None:
            # Fallback: search for the largest 2D tensor as weight
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and v.dim() == 2:
                    if W is None or (v.numel() > W.numel()):
                        W = v

        for k in bias_keys:
            if k in state and isinstance(state[k], torch.Tensor):
                b = state[k]
                break

        return W, b


def time_forward(module: torch.nn.Module, x: torch.Tensor, repeat: int = 5) -> float:
    """Simple CPU/GPU-agnostic forward timing helper (seconds per call)."""
    # Warmup
    with torch.no_grad():
        _ = module(x)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeat):
            _ = module(x)
    end = time.perf_counter()
    return (end - start) / max(1, repeat)


