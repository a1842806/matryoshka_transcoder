from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.models.sae import MatryoshkaTranscoder
from src.utils.config import compute_fvu

from .config import AutoInterpSAEConfig, NPZTranscoderSpec


def _norm_decoder_rows(weights: torch.Tensor) -> torch.Tensor:
    return weights / weights.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _coerce_matrix(
    array: np.ndarray, *, expected: Tuple[int, int], name: str
) -> np.ndarray:
    if array.shape == expected:
        return array
    if array.shape == expected[::-1]:
        return array.T
    raise ValueError(
        f"Unexpected shape for {name}: {array.shape} (expected {expected} or {expected[::-1]})"
    )


def _coerce_vector(array: np.ndarray, *, expected: int, name: str) -> np.ndarray:
    if array.shape == (expected,):
        return array
    if array.shape == (expected, 1):
        return array.reshape(-1)
    if array.shape == (1, expected):
        return array.reshape(-1)
    raise ValueError(
        f"Unexpected shape for {name}: {array.shape} (expected {(expected,)})"
    )


@dataclass(slots=True)
class ReconstructionReport:
    fvu: float
    mse: float
    mean_activation: float


class MatryoshkaAutoInterpAdapter(MatryoshkaTranscoder):
    """Wrap :class:`MatryoshkaTranscoder` with the attributes SAEBench expects."""

    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        sae_cfg: AutoInterpSAEConfig,
    ) -> None:
        cfg_local = copy.deepcopy(cfg)
        dict_size = int(cfg_local["dict_size"])
        prefix_sizes = cfg_local.get("prefix_sizes") or [dict_size]
        if prefix_sizes[-1] != dict_size:
            prefix_sizes = list(prefix_sizes)
            prefix_sizes.append(dict_size)
        cfg_local["prefix_sizes"] = prefix_sizes

        top_k = int(cfg_local.get("top_k", dict_size))
        if top_k <= 0 or top_k > dict_size:
            top_k = dict_size
        cfg_local["top_k"] = top_k

        top_k_aux = int(cfg_local.get("top_k_aux", top_k * 2))
        cfg_local["top_k_aux"] = max(top_k_aux, top_k)
        cfg_local.setdefault("n_batches_to_dead", 20)

        super().__init__(cfg_local)

        device = torch.device(cfg_local["device"])
        dtype = cfg_local["dtype"]

        self.cfg = sae_cfg
        self.device = device
        self.dtype = dtype
        self.to(device=device, dtype=dtype)

    def to(self, *args, **kwargs):  # type: ignore[override]
        module = super().to(*args, **kwargs)
        if "device" in kwargs:
            self.device = torch.device(kwargs["device"])
        if "dtype" in kwargs:
            self.dtype = kwargs["dtype"]
        return module

    def get_decoder_row_norms(self) -> torch.Tensor:
        return self.W_dec.detach().norm(dim=-1)

    @torch.no_grad()
    def validate_reconstruction(
        self, source_acts: torch.Tensor, target_acts: torch.Tensor
    ) -> ReconstructionReport:
        encoded = self.encode(source_acts)
        decoded = self.decode(encoded)
        mse = F.mse_loss(decoded, target_acts).item()
        fvu = compute_fvu(target_acts, decoded)
        mean_activation = encoded.abs().mean().item()
        return ReconstructionReport(fvu=fvu, mse=mse, mean_activation=mean_activation)


def _infer_hook_name(model_name: str, site: str, layer: int) -> str:
    from src.utils.config import get_hook_name

    return get_hook_name(site, layer, model_name)


def build_matryoshka_adapter(
    cfg: Mapping[str, Any],
    *,
    state_dict_path: str | Path | None = None,
    hook_layer: int | None = None,
    hook_point: str | None = None,
) -> MatryoshkaAutoInterpAdapter:
    """Instantiate a Matryoshka transcoder compatible with SAEBench AutoInterp."""

    cfg_local = copy.deepcopy(dict(cfg))
    if hook_layer is None:
        hook_layer = cfg_local.get("source_layer", cfg_local["layer"])
    if hook_point is None:
        hook_point = cfg_local.get(
            "source_hook_point",
            _infer_hook_name(cfg_local["model_name"], cfg_local["source_site"], hook_layer),
        )

    sae_cfg = AutoInterpSAEConfig.from_kwargs(
        model_name=cfg_local["model_name"],
        hook_layer=hook_layer,
        hook_name=hook_point,
        d_in=cfg_local.get("source_act_size", cfg_local["act_size"]),
        d_sae=cfg_local["dict_size"],
        dtype=cfg_local["dtype"],
        device=torch.device(cfg_local["device"]),
        context_size=cfg_local.get("seq_len"),
        dataset_path=cfg_local.get("dataset_path", ""),
    )

    adapter = MatryoshkaAutoInterpAdapter(cfg_local, sae_cfg=sae_cfg)

    if state_dict_path is not None:
        state_path = Path(state_dict_path)
        state = torch.load(state_path, map_location=adapter.device)
        adapter.load_state_dict(state, strict=False)

    adapter.W_dec.data = _norm_decoder_rows(adapter.W_dec.data)
    return adapter


class NPZTranscoderAdapter(MatryoshkaAutoInterpAdapter):
    """Adapter that loads a cross-layer transcoder from a ``.npz`` checkpoint."""

    def __init__(self, *, spec: NPZTranscoderSpec):
        dtype = getattr(torch, spec.dtype)
        device = torch.device(spec.device)

        target_layer = spec.target_layer or spec.source_layer
        dict_size = int(spec.d_sae)
        prefix_sizes = list(spec.prefix_sizes) if spec.prefix_sizes else [dict_size]
        if prefix_sizes[-1] != dict_size:
            prefix_sizes.append(dict_size)

        top_k = spec.top_k if spec.top_k is not None else min(512, dict_size)
        if top_k <= 0 or top_k > dict_size:
            top_k = dict_size

        cfg: dict[str, Any] = {
            "model_name": spec.model_name,
            "device": spec.device,
            "dtype": dtype,
            "dict_size": spec.d_sae,
            "source_act_size": spec.d_in,
            "target_act_size": spec.d_out,
            "top_k": top_k,
            "prefix_sizes": prefix_sizes,
            "aux_penalty": spec.aux_penalty or 0.0,
            "layer": spec.source_layer,
            "source_layer": spec.source_layer,
            "target_layer": target_layer,
            "source_site": spec.source_hook_point.split(".")[-1],
            "target_site": spec.target_hook_point.split(".")[-1],
            "seq_len": 128,
        }

        sae_cfg = AutoInterpSAEConfig.from_kwargs(
            model_name=spec.model_name,
            hook_layer=spec.source_layer,
            hook_name=spec.source_hook_point,
            d_in=spec.d_in,
            d_sae=spec.d_sae,
            dtype=dtype,
            device=device,
        )

        super().__init__(cfg, sae_cfg=sae_cfg)
        self.spec = spec
        self.load_npz_weights(spec)
        self.W_dec.data = _norm_decoder_rows(self.W_dec.data)

    def load_npz_weights(self, spec: NPZTranscoderSpec) -> None:
        npz_path = Path(spec.path)
        if not npz_path.exists():
            raise FileNotFoundError(f"Transcoder checkpoint not found: {npz_path}")

        data = np.load(npz_path)

        def fetch(key_candidates: Iterable[str]) -> np.ndarray:
            for name in key_candidates:
                if name in data:
                    return data[name]
            raise KeyError(
                f"Unable to find any of {list(key_candidates)} in {npz_path.name}"
            )

        key_map = spec.weight_key_map

        W_enc_np = fetch([key_map.get("W_enc", "W_enc"), "W_enc", "encoder.W"])
        W_dec_np = fetch([key_map.get("W_dec", "W_dec"), "W_dec", "decoder.W"])
        b_enc_np = fetch([key_map.get("b_enc", "b_enc"), "b_enc", "encoder.b"])
        b_dec_np = fetch([key_map.get("b_dec", "b_dec"), "b_dec", "decoder.b"])

        W_enc_np = _coerce_matrix(W_enc_np, expected=(spec.d_in, spec.d_sae), name="W_enc")
        W_dec_np = _coerce_matrix(W_dec_np, expected=(spec.d_sae, spec.d_out), name="W_dec")
        b_enc_np = _coerce_vector(b_enc_np, expected=spec.d_sae, name="b_enc")
        b_dec_np = _coerce_vector(b_dec_np, expected=spec.d_out, name="b_dec")

        device = self.device
        dtype = self.dtype

        self.W_enc.data.copy_(torch.from_numpy(W_enc_np).to(device=device, dtype=dtype))
        self.W_dec.data.copy_(torch.from_numpy(W_dec_np).to(device=device, dtype=dtype))
        self.b_enc.data.copy_(torch.from_numpy(b_enc_np).to(device=device, dtype=dtype))
        self.b_dec.data.copy_(torch.from_numpy(b_dec_np).to(device=device, dtype=dtype))


def build_npz_transcoder_adapter(spec: NPZTranscoderSpec) -> NPZTranscoderAdapter:
    return NPZTranscoderAdapter(spec=spec)


