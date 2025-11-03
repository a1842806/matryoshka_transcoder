"""Command-line helper to run SAEBench AutoInterp on project transcoders."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from openai import OpenAI
from sae_bench.evals.autointerp.eval_config import AutoInterpEvalConfig
from sae_bench.evals.autointerp.main import AutoInterp
from sae_bench.sae_bench_utils import general_utils
from transformer_lens import HookedTransformer

from eval.sae_bench.activation_pipeline import (
    AutoInterpActivationCache,
    AutoInterpDatasetSummary,
    prepare_auto_interp_cache,
)
from eval.sae_bench.config import NPZTranscoderSpec
from eval.sae_bench.wrappers import (
    MatryoshkaAutoInterpAdapter,
    build_matryoshka_adapter,
    build_npz_transcoder_adapter,
)
from src.utils.config import build_autointerp_eval_cfg, get_default_cfg


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _resolve_dtype(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        try:
            return getattr(torch, value)
        except AttributeError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported torch dtype string: {value}") from exc
    raise TypeError(f"Cannot resolve dtype from value of type {type(value)!r}")


def _resolve_device(value: Any) -> torch.device:
    if isinstance(value, torch.device):
        return value
    if isinstance(value, str):
        return torch.device(value)
    raise TypeError(f"Cannot resolve device from value of type {type(value)!r}")


def _maybe_path(value: str | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def _read_latents(path: Path | None) -> list[int] | None:
    if path is None:
        return None
    with path.open("r") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return [int(x) for x in payload]
    if isinstance(payload, dict) and "latents" in payload:
        return [int(x) for x in payload["latents"]]
    raise ValueError("Override latents file must be a list or contain a 'latents' key")


def _normalize_messages(messages: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    normalized = []
    for message in messages:
        if set(message.keys()) != {"role", "content"}:
            raise ValueError("Messages must contain exactly 'role' and 'content' keys")
        normalized.append({"role": message["role"], "content": message["content"]})
    return normalized


def _make_get_api_response(
    *,
    model_name: str,
    base_url: str | None,
    temperature: float,
    request_timeout: float,
    max_retries: int,
) -> Callable[[AutoInterp, Iterable[dict[str, str]], int, int], tuple[list[str], str]]:
    """Factory that returns a patched ``get_api_response`` bound method."""

    def _call_with_retry(client: OpenAI, **kwargs: Any) -> Any:
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                return client.chat.completions.create(**kwargs)
            except Exception as exc:  # pragma: no cover - network dependent
                last_err = exc
                wait = min(2 ** attempt, 30.0)
                time.sleep(wait)
        assert last_err is not None
        raise last_err

    def get_api_response(
        self: AutoInterp,
        messages: Iterable[dict[str, str]],
        max_tokens: int,
        n_completions: int = 1,
    ) -> tuple[list[str], str]:
        payload = _normalize_messages(messages)
        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)

        result = _call_with_retry(
            client,
            model=model_name,
            messages=payload,
            n=n_completions,
            max_tokens=max_tokens,
            stream=False,
            temperature=temperature,
            timeout=request_timeout,
        )

        responses = [choice.message.content.strip() for choice in result.choices]

        from tabulate import tabulate

        log_table = tabulate(
            [m.values() for m in payload + [{"role": "assistant", "content": responses[0]}]],
            tablefmt="simple_grid",
            maxcolwidths=[None, 120],
        )

        return responses, log_table

    return get_api_response


def _default_results_root(cfg: dict[str, Any]) -> Path:
    root = cfg.get("results_root", "results")
    return Path(root).expanduser().resolve()


def _adapter_from_args(
    *,
    adapter_type: str,
    eval_cfg: dict[str, Any],
    state_dict_path: Path | None,
    npz_spec_path: Path | None,
) -> MatryoshkaAutoInterpAdapter:
    if adapter_type == "matryoshka":
        if state_dict_path is None:
            raise ValueError("--state-dict is required for matryoshka adapter")
        return build_matryoshka_adapter(eval_cfg, state_dict_path=state_dict_path)

    if adapter_type == "npz":
        if npz_spec_path is None:
            raise ValueError("--npz-spec is required for npz adapter")
        spec_payload = _load_json(npz_spec_path)
        spec = NPZTranscoderSpec(**spec_payload)
        return build_npz_transcoder_adapter(spec)

    raise ValueError(f"Unsupported adapter type: {adapter_type}")


def _build_eval_config(
    *,
    eval_cfg: dict[str, Any],
    args: argparse.Namespace,
    override_latents: list[int] | None,
) -> AutoInterpEvalConfig:
    config_kwargs: dict[str, Any] = {
        "model_name": eval_cfg["model_name"],
        "dataset_name": eval_cfg.get("dataset_path", eval_cfg.get("dataset_name", "")),
        "llm_context_size": int(eval_cfg["seq_len"]),
        "total_tokens": int(args.total_tokens),
    }

    if override_latents is not None:
        config_kwargs["override_latents"] = override_latents
    elif args.n_latents is not None:
        config_kwargs["n_latents"] = int(args.n_latents)

    config_kwargs["random_seed"] = args.random_seed if args.random_seed is not None else eval_cfg.get("seed", 42)

    config = AutoInterpEvalConfig(**config_kwargs)

    batch_size = args.llm_batch_size or eval_cfg.get("model_batch_size") or 32
    config.llm_batch_size = int(batch_size)

    base_dtype = args.llm_dtype or eval_cfg.get("model_dtype") or eval_cfg.get("dtype") or "bfloat16"
    if isinstance(base_dtype, torch.dtype):
        base_dtype = general_utils.dtype_to_str(base_dtype)
    config.llm_dtype = str(base_dtype)

    config.scoring = not args.explain_only
    config.use_demos_in_explanation = not args.no_demos

    if args.buffer is not None:
        config.buffer = int(args.buffer)
    if args.act_threshold_frac is not None:
        config.act_threshold_frac = float(args.act_threshold_frac)
    if args.n_top_generation is not None:
        config.n_top_ex_for_generation = int(args.n_top_generation)
    if args.n_top_scoring is not None:
        config.n_top_ex_for_scoring = int(args.n_top_scoring)
    if args.n_random_scoring is not None:
        config.n_random_ex_for_scoring = int(args.n_random_scoring)

    return config


def _prepare_tokens_and_sparsity(
    *,
    model: HookedTransformer,
    eval_cfg: dict[str, Any],
    results_root: Path,
    description: str,
    args: argparse.Namespace,
    sae: MatryoshkaAutoInterpAdapter,
    config: AutoInterpEvalConfig,
) -> tuple[torch.Tensor, torch.Tensor, AutoInterpDatasetSummary]:
    dataset_summary = prepare_auto_interp_cache(
        model,
        eval_cfg,
        results_root=results_root,
        description=description,
        total_tokens=args.total_tokens,
        shard_sequences=args.shard_sequences,
        max_shards=args.max_shards,
        flatten=True,
        metadata={
            "generated_by": "eval/sae_bench/run_autointerp.py",
            "timestamp": int(time.time()),
            "adapter_type": args.adapter,
        },
        max_context_length=args.max_context_length,
    )

    cache = AutoInterpActivationCache(
        results_root=results_root,
        model_name=eval_cfg["model_name"],
        layer=int(eval_cfg.get("source_layer", eval_cfg["layer"])),
        description=description,
    )

    tokens = torch.load(cache.paths.tokens)
    model_device = next(model.parameters()).device
    tokens = tokens.to(model_device)

    sparsity = cache.ensure_sparsity(
        tokens,
        model=model,
        sae=sae,
        batch_size=config.llm_batch_size,
        hook_layer=sae.cfg.hook_layer,
        hook_name=sae.cfg.hook_name,
    )

    return tokens, sparsity, dataset_summary


def _summarise_results(
    results: dict[int, dict[str, Any]],
    *,
    expected_latents: int | None,
) -> dict[str, Any]:
    scores = [entry.get("score") for entry in results.values() if "score" in entry]
    summary: dict[str, Any] = {
        "n_features_with_results": len(results),
        "n_features_with_scores": len(scores),
    }
    if scores:
        summary["mean_score"] = float(statistics.mean(scores))
        if len(scores) > 1:
            summary["stdev_score"] = float(statistics.pstdev(scores))
        if expected_latents:
            summary["coverage"] = float(len(scores) / expected_latents)
    return summary


def _write_outputs(
    *,
    run_dir: Path,
    summary: dict[str, Any],
    details: dict[int, dict[str, Any]],
    eval_config: AutoInterpEvalConfig,
    sae: MatryoshkaAutoInterpAdapter,
    dataset_summary: AutoInterpDatasetSummary,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    details_payload = {
        "latents": {str(k): v for k, v in details.items()},
        "timestamp": int(time.time()),
    }
    (run_dir / "details.json").write_text(json.dumps(details_payload, indent=2))

    cfg_payload = {
        "auto_interp": asdict(eval_config),
        "adapter_cfg": sae.cfg.to_dict(),
        "dataset": dataset_summary.as_dict(),
    }
    (run_dir / "config.json").write_text(json.dumps(cfg_payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAEBench AutoInterp on Matryoshka transcoders")
    parser.add_argument("--base-config", type=str, help="Path to base training config JSON", default=None)
    parser.add_argument("--layer", type=int, required=True, help="Model layer to evaluate")
    parser.add_argument("--description", type=str, default="auto_interp", help="Run descriptor for results folder")
    parser.add_argument("--adapter", choices=["matryoshka", "npz"], default="matryoshka")
    parser.add_argument("--state-dict", type=str, help="Path to Matryoshka state dict (.pt)")
    parser.add_argument("--npz-spec", type=str, help="Path to NPZTranscoderSpec JSON definition")
    parser.add_argument("--results-root", type=str, default=None, help="Root directory for evaluation artifacts")
    parser.add_argument("--total-tokens", type=int, default=2_000_000)
    parser.add_argument("--shard-sequences", type=int, default=2048)
    parser.add_argument("--max-shards", type=int, default=8)
    parser.add_argument("--max-context-length", type=int, default=512)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--n-latents", type=int, default=None)
    parser.add_argument("--override-latents", type=str, default=None, help="Optional JSON file with latent ids")
    parser.add_argument("--llm-batch-size", type=int, default=None)
    parser.add_argument("--llm-dtype", type=str, default=None)
    parser.add_argument("--buffer", type=int, default=None)
    parser.add_argument("--act-threshold-frac", type=float, default=None)
    parser.add_argument("--n-top-generation", type=int, default=None)
    parser.add_argument("--n-top-scoring", type=int, default=None)
    parser.add_argument("--n-random-scoring", type=int, default=None)
    parser.add_argument("--explain-only", action="store_true", help="Skip scoring phase and only gather explanations")
    parser.add_argument("--no-demos", action="store_true", help="Disable exemplar hints in judge prompt")
    parser.add_argument("--source-site", type=str, default="mlp_in")
    parser.add_argument("--target-site", type=str, default="mlp_out")
    parser.add_argument("--api-key-env", type=str, default="AUTO_INTERP_OPENAI_API_KEY")
    parser.add_argument("--api-key-file", type=str, default=None)
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--judge-base-url", type=str, default=None)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-timeout", type=float, default=60.0)
    parser.add_argument("--judge-max-retries", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_cfg = _load_json(Path(args.base_config)) if args.base_config else get_default_cfg()

    eval_cfg = build_autointerp_eval_cfg(
        base_cfg,
        layer=args.layer,
        total_tokens=args.total_tokens,
        context_size=base_cfg.get("seq_len", 128),
        description=args.description,
        source_site=args.source_site,
        target_site=args.target_site,
    )

    results_root = Path(args.results_root).expanduser().resolve() if args.results_root else _default_results_root(eval_cfg)

    device = _resolve_device(eval_cfg.get("device", "cuda"))
    dtype = _resolve_dtype(eval_cfg.get("model_dtype", eval_cfg.get("dtype", torch.bfloat16)))

    model = HookedTransformer.from_pretrained_no_processing(
        eval_cfg["model_name"],
        dtype=dtype,
    ).to(device)
    model.eval()

    state_path = _maybe_path(args.state_dict)
    npz_spec_path = _maybe_path(args.npz_spec)
    adapter = _adapter_from_args(
        adapter_type=args.adapter,
        eval_cfg=eval_cfg,
        state_dict_path=state_path,
        npz_spec_path=npz_spec_path,
    )
    adapter = adapter.to(device=device, dtype=dtype)

    override_latents = _read_latents(_maybe_path(args.override_latents))
    eval_config = _build_eval_config(eval_cfg=eval_cfg, args=args, override_latents=override_latents)

    tokens, sparsity, dataset_summary = _prepare_tokens_and_sparsity(
        model=model,
        eval_cfg=eval_cfg,
        results_root=results_root,
        description=args.description,
        args=args,
        sae=adapter,
        config=eval_config,
    )

    api_key: str | None = None
    if args.api_key_file:
        api_key = Path(args.api_key_file).read_text().strip()
    if api_key is None and args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None:
        raise RuntimeError("OpenAI API key not provided. Use --api-key-file or set an env var.")

    autointerp = AutoInterp(
        cfg=eval_config,
        model=model,
        sae=adapter,
        tokenized_dataset=tokens,
        sparsity=sparsity,
        device=str(device),
        api_key=api_key,
    )

    patched = _make_get_api_response(
        model_name=args.judge_model,
        base_url=args.judge_base_url,
        temperature=args.judge_temperature,
        request_timeout=args.judge_timeout,
        max_retries=args.judge_max_retries,
    )
    autointerp.get_api_response = patched.__get__(autointerp, AutoInterp)  # type: ignore[attr-defined]

    results = asyncio.run(autointerp.run())

    summary = _summarise_results(results, expected_latents=eval_config.n_latents)

    run_dir = (
        results_root
        / eval_cfg["model_name"]
        / f"layer{eval_cfg.get('source_layer', eval_cfg['layer'])}"
        / "auto_interp"
        / args.description
    )

    _write_outputs(
        run_dir=run_dir,
        summary=summary,
        details=results,
        eval_config=eval_config,
        sae=adapter,
        dataset_summary=dataset_summary,
    )

    print(json.dumps({"results_dir": str(run_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()


