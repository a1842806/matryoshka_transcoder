from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Mapping, Sequence


def _dtype_to_str(torch_dtype: "torch.dtype") -> str:
    try:
        return str(torch_dtype).split(".")[-1]
    except Exception:
        return str(torch_dtype)


@dataclass(slots=True)
class AutoInterpSAEConfig:
    """Lightweight config mirroring the fields SAEBench expects on ``sae.cfg``.

    The official SAEBench helpers read a subset of :class:`sae_lens.SAE` config
    attributes (``hook_layer``, ``hook_name``, dimensionalities, dtype/device
    strings, etc.).  We recreate just the required structure so our custom
    transcoders can plug into the evaluation pipeline without rewriting the
    upstream utilities.
    """

    model_name: str
    hook_layer: int
    hook_name: str
    d_in: int
    d_sae: int

    # Optional metadata used by SAEBench when serialising results.
    context_size: int | None = None
    hook_head_index: int | None = None

    # Architecture details (mostly informative; SAEBench only checks presence).
    architecture: str = ""
    apply_b_dec_to_input: bool | None = None
    finetuning_scaling_factor: float | None = None
    activation_fn_str: str = "relu"
    activation_fn_kwargs: Mapping[str, Any] = field(default_factory=dict)
    prepend_bos: bool = True
    normalize_activations: str = "none"

    # Runtime metadata expected by ``sae.cfg``.
    dtype: str = ""
    device: str = ""
    model_from_pretrained_kwargs: Mapping[str, Any] = field(default_factory=dict)

    # Dataset/training provenance (optional; improves downstream bookkeeping).
    dataset_path: str = ""
    dataset_trust_remote_code: bool = True
    seqpos_slice: Sequence[int | None] = field(default_factory=lambda: (None,))
    training_tokens: int = -100_000

    # Compatibility with SAE Lens metadata.
    sae_lens_training_version: str | None = None
    neuronpedia_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        cfg_dict = {**asdict(self)}
        # Remove potentially unserialisable mappings (mirrors SAE Lens helpers).
        if "metadata" in cfg_dict:
            cfg_dict.pop("metadata")
        return cfg_dict

    @classmethod
    def from_kwargs(
        cls,
        *,
        model_name: str,
        hook_layer: int,
        hook_name: str,
        d_in: int,
        d_sae: int,
        dtype: "torch.dtype",
        device: "torch.device" | str,
        **overrides: Any,
    ) -> "AutoInterpSAEConfig":
        import torch

        device_str = str(device) if isinstance(device, str) else device.type
        dtype_str = _dtype_to_str(dtype if isinstance(dtype, torch.dtype) else torch.get_default_dtype())

        return cls(
            model_name=model_name,
            hook_layer=hook_layer,
            hook_name=hook_name,
            d_in=d_in,
            d_sae=d_sae,
            dtype=dtype_str,
            device=device_str,
            **overrides,
        )


@dataclass(slots=True)
class NPZTranscoderSpec:
    """Metadata describing a Gemma transcoder checkpoint stored in ``.npz`` form."""

    path: str
    model_name: str
    source_layer: int
    source_hook_point: str
    target_hook_point: str
    d_in: int
    d_out: int
    d_sae: int
    target_layer: int | None = None
    dtype: str = "bfloat16"
    device: str = "cuda"
    top_k: int | None = None
    prefix_sizes: Sequence[int] | None = None
    aux_penalty: float | None = None
    notes: str | None = None

    # Flexible key mapping for diverse checkpoint layouts.
    weight_key_map: Mapping[str, str] = field(
        default_factory=lambda: {
            "W_enc": "W_enc",
            "W_dec": "W_dec",
            "b_enc": "b_enc",
            "b_dec": "b_dec",
        }
    )

    def torch_dtype(self):
        import torch

        return getattr(torch, self.dtype)

    def torch_device(self):
        import torch

        return torch.device(self.device)


