from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from transformer_lens.hook_points import HookedRootModule
from tqdm import trange

from src.models.transcoder_activation_store import TranscoderActivationsStore


@dataclass(slots=True)
class CachePaths:
    root: Path
    tokens: Path
    metadata: Path
    sparsity: Path


@dataclass(slots=True)
class AutoInterpDatasetSummary:
    cache_root: Path
    tokens_path: Path
    shard_paths: Sequence[Path]
    metadata_path: Path
    sparsity_path: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "cache_root": str(self.cache_root),
            "tokens_path": str(self.tokens_path),
            "shard_paths": [str(path) for path in self.shard_paths],
            "metadata_path": str(self.metadata_path),
            "sparsity_path": str(self.sparsity_path) if self.sparsity_path else None,
        }


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to("cpu")


class AutoInterpActivationCache:
    """Utility to materialise token datasets and metadata for AutoInterp."""

    def __init__(
        self,
        *,
        results_root: Path,
        model_name: str,
        layer: int,
        description: str,
    ) -> None:
        safe_desc = description.replace("/", "-")
        base = (
            results_root
            / model_name
            / f"layer{layer}"
            / "auto_interp"
            / "cache"
            / safe_desc
        )
        self.paths = CachePaths(
            root=base,
            tokens=base / "tokens.pt",
            metadata=base / "metadata.json",
            sparsity=base / "sparsity.pt",
        )
        _ensure_dir(self.paths.root)

    def ensure_token_dataset(
        self,
        store: TranscoderActivationsStore,
        *,
        total_tokens: int,
    ) -> torch.Tensor:
        if self.paths.tokens.exists():
            return torch.load(self.paths.tokens)

        seq_len = store.context_size
        tokens_per_sequence = seq_len
        sequences_needed = math.ceil(total_tokens / tokens_per_sequence)
        tokens_per_batch = store.model_batch_size * seq_len
        batches = math.ceil(sequences_needed * seq_len / tokens_per_batch)

        token_chunks = []
        for _ in trange(batches, desc="Collecting token batches", leave=False):
            token_chunks.append(_to_cpu(store.get_batch_tokens()))

        tokens = torch.cat(token_chunks, dim=0)[:sequences_needed]
        torch.save(tokens, self.paths.tokens)
        return tokens

    def save_metadata(self, **metadata) -> None:
        with open(self.paths.metadata, "w") as f:
            json.dump(metadata, f, indent=2)

    def _shard_path(self, shard_idx: int) -> Path:
        return self.paths.root / f"activations_{shard_idx:02d}.pt"

    def materialise_activation_shards(
        self,
        store: TranscoderActivationsStore,
        *,
        shard_sequences: int,
        max_shards: int,
        flatten: bool = True,
    ) -> list[Path]:
        paths: list[Path] = []
        seq_len = store.context_size
        sequences_per_batch = store.model_batch_size

        for shard_idx in range(max_shards):
            shard_path = self._shard_path(shard_idx)
            if shard_path.exists():
                paths.append(shard_path)
                continue

            collected = 0
            source_batches: list[torch.Tensor] = []
            target_batches: list[torch.Tensor] = []

            while collected < shard_sequences:
                batch_tokens = store.get_batch_tokens()
                source, target = store.get_paired_activations(batch_tokens)
                source_batches.append(_to_cpu(source))
                target_batches.append(_to_cpu(target))
                collected += sequences_per_batch

            source_tensor = torch.cat(source_batches, dim=0)
            target_tensor = torch.cat(target_batches, dim=0)

            if flatten:
                source_tensor = source_tensor.reshape(-1, source_tensor.shape[-1])
                target_tensor = target_tensor.reshape(-1, target_tensor.shape[-1])

            torch.save({"source": source_tensor, "target": target_tensor}, shard_path)
            paths.append(shard_path)

        return paths

    def ensure_sparsity(
        self,
        tokens: torch.Tensor,
        *,
        model,
        sae,
        batch_size: int,
        hook_layer: int,
        hook_name: str,
    ) -> torch.Tensor:
        if self.paths.sparsity.exists():
            return torch.load(self.paths.sparsity)

        from sae_bench.sae_bench_utils.activation_collection import (
            get_feature_activation_sparsity,
        )

        sparsity = get_feature_activation_sparsity(
            tokens=tokens,
            model=model,
            sae=sae,
            batch_size=batch_size,
            layer=hook_layer,
            hook_name=hook_name,
            mask_bos_pad_eos_tokens=True,
        )
        torch.save(_to_cpu(sparsity), self.paths.sparsity)
        return sparsity


def prepare_auto_interp_cache(
    model: HookedRootModule,
    cfg: Mapping[str, Any],
    *,
    results_root: Path,
    description: str,
    total_tokens: int,
    shard_sequences: int,
    max_shards: int,
    flatten: bool = True,
    metadata: Mapping[str, Any] | None = None,
    max_context_length: int | None = 512,
) -> AutoInterpDatasetSummary:
    """Stream activations into cached shards for AutoInterp evaluation."""

    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive")
    if shard_sequences <= 0:
        raise ValueError("shard_sequences must be positive")
    if max_shards <= 0:
        raise ValueError("max_shards must be positive")

    cfg_local = dict(cfg)
    store = TranscoderActivationsStore(model, cfg_local)

    if max_context_length is not None and store.context_size > max_context_length:
        raise ValueError(
            f"Context size {store.context_size} exceeds AutoInterp limit {max_context_length}"
        )

    results_root = Path(results_root)

    cache = AutoInterpActivationCache(
        results_root=results_root,
        model_name=cfg_local["model_name"],
        layer=int(cfg_local.get("source_layer", cfg_local["layer"])),
        description=description,
    )

    tokens = cache.ensure_token_dataset(store, total_tokens=total_tokens)
    num_sequences = int(tokens.shape[0]) if tokens.ndim > 1 else int(tokens.numel())
    sequence_length = int(tokens.shape[1]) if tokens.ndim > 1 else 1
    total_tokens_collected = int(tokens.numel())
    shard_paths = cache.materialise_activation_shards(
        store,
        shard_sequences=shard_sequences,
        max_shards=max_shards,
        flatten=flatten,
    )

    metadata_payload = {
        "model_name": cfg_local["model_name"],
        "layer": int(cfg_local.get("source_layer", cfg_local["layer"])),
        "source_hook_point": cfg_local["source_hook_point"],
        "target_hook_point": cfg_local["target_hook_point"],
        "seq_len": int(store.context_size),
        "model_batch_size": int(store.model_batch_size),
        "total_tokens_requested": int(total_tokens),
        "total_tokens_collected": total_tokens_collected,
        "num_sequences": num_sequences,
        "sequence_length": sequence_length,
        "shard_sequences": int(shard_sequences),
        "max_shards": int(max_shards),
    }
    if metadata is not None:
        metadata_payload.update(metadata)

    cache.save_metadata(**metadata_payload)

    del tokens

    return AutoInterpDatasetSummary(
        cache_root=cache.paths.root,
        tokens_path=cache.paths.tokens,
        shard_paths=shard_paths,
        metadata_path=cache.paths.metadata,
        sparsity_path=cache.paths.sparsity if cache.paths.sparsity.exists() else None,
    )


