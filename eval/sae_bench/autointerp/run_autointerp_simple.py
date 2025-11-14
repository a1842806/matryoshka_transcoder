"""Simplified AutoInterp evaluation for local execution.

Usage:
    python eval/sae_bench/run_autointerp_simple.py \
        --matryoshka results/gemma-2-2b/layer8/checkpoints/final.pt \
        --layer 8 \
        --n-latents 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from sae_bench.evals.autointerp.eval_config import AutoInterpEvalConfig
from sae_bench.evals.autointerp.main import AutoInterp
from transformer_lens import HookedTransformer

from eval.sae_bench.wrappers import build_matryoshka_adapter, build_npz_transcoder_adapter
from eval.sae_bench.config import NPZTranscoderSpec
from src.utils.config import get_default_cfg


def load_matryoshka(state_dict_path: Path, layer: int) -> Any:
    """Load Matryoshka transcoder from checkpoint.
    
    Note: Using hook_resid_mid instead of hook_mlp_in because Gemma-2's
    hook_mlp_in is not cached by TransformerLens. resid_mid is pre-layernorm
    version of mlp_in, very similar for evaluation purposes.
    """
    cfg = get_default_cfg()
    cfg["model_name"] = "gemma-2-2b"
    cfg["layer"] = layer
    cfg["source_layer"] = layer
    cfg["target_layer"] = layer
    cfg["source_site"] = "resid_mid"  # Use resid_mid (cached) instead of mlp_in (not cached)
    cfg["target_site"] = "mlp_out"
    cfg["dict_size"] = 18432  # Adjust based on your model
    cfg["top_k"] = 96
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["dtype"] = torch.bfloat16  # Use bfloat16 for stability
    
    # CRITICAL: Set correct activation dimensions for Gemma-2-2B!
    cfg["act_size"] = 2304  # Gemma-2-2B activation size (not GPT-2's 768)
    cfg["source_act_size"] = 2304
    cfg["target_act_size"] = 2304
    
    return build_matryoshka_adapter(cfg, state_dict_path=state_dict_path)


def load_google_transcoder(npz_path: Path, layer: int) -> Any:
    """Load Google's transcoder from NPZ file.
    
    Note: Using hook_resid_mid for source instead of hook_mlp_in
    because hook_mlp_in is not cached by TransformerLens for Gemma-2.
    """
    spec = NPZTranscoderSpec(
        path=str(npz_path),
        model_name="gemma-2-2b",
        source_layer=layer,
        source_hook_point=f"blocks.{layer}.hook_resid_mid",  # Use resid_mid (cached)
        target_hook_point=f"blocks.{layer}.hook_mlp_out",
        d_in=2304,  # Gemma-2-2B activation size
        d_out=2304,
        d_sae=16384,  # Adjust based on Google's transcoder size
        dtype="float16",
        device="cuda" if torch.cuda.is_available() else "cpu",
        top_k=96,
    )
    return build_npz_transcoder_adapter(spec)


def prepare_data(
    model: HookedTransformer,
    sae: Any,
    config: AutoInterpEvalConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare tokenized data and compute sparsity on-the-fly."""
    from datasets import load_dataset
    
    # Load dataset and tokenize
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    
    # Tokenize enough sequences
    seq_len = config.llm_context_size
    n_sequences = (config.total_tokens + seq_len - 1) // seq_len
    
    tokens_list = []
    for i, example in enumerate(dataset):
        if i >= n_sequences:
            break
        text = example["text"]
        tokens = model.to_tokens(text, prepend_bos=True)
        if tokens.shape[1] >= seq_len:
            tokens_list.append(tokens[:, :seq_len])
    
    tokens = torch.cat(tokens_list, dim=0)[:n_sequences]
    
    # Move tokens to the same device as the model to ensure GPU is used
    tokens = tokens.to(next(model.parameters()).device, non_blocking=True)
    
    # Compute actual feature activation sparsity (matches paper methodology)
    print(f"[2/3] Computing feature sparsity...")
    from sae_bench.sae_bench_utils.activation_collection import (
        get_feature_activation_sparsity,
    )
    
    sparsity = get_feature_activation_sparsity(
        tokens=tokens,
        model=model,
        sae=sae,
        batch_size=config.llm_batch_size,
        layer=sae.cfg.hook_layer,
        hook_name=sae.cfg.hook_name,
        mask_bos_pad_eos_tokens=True,
    )
    print(f"✓ Computed sparsity for {len(sparsity)} features (mean: {sparsity.mean():.4f})")
    
    return tokens, sparsity


def evaluate_transcoder(
    name: str,
    sae: Any,
    model: HookedTransformer,
    n_latents: int,
    total_tokens: int,
    api_key: str,
) -> dict[str, Any]:
    """Evaluate a single transcoder."""
    
    # Create config
    config = AutoInterpEvalConfig(
        model_name="gemma-2-2b",
        dataset_name="HuggingFaceFW/fineweb-edu",
        llm_context_size=128,
        total_tokens=total_tokens,
        n_latents=n_latents,
        llm_batch_size=32,
        llm_dtype="bfloat16",
        random_seed=42,
    )
    
    print(f"\n{'='*60}")
    print(f"EVALUATING: {name}")
    print(f"{'='*60}")
    
    # Prepare data
    print(f"[1/3] Preparing tokenized dataset ({config.total_tokens:,} tokens)...")
    start_time = time.time()
    tokens, sparsity = prepare_data(model, sae, config)
    prep_time = time.time() - start_time
    print(f"✓ Data + sparsity prepared in {prep_time:.1f}s (tokens shape: {tokens.shape})")
    
    # Run AutoInterp
    print(f"[3/3] Running AutoInterp on {n_latents} features...")
    print(f"      Estimated time: ~{n_latents * 2} minutes (2 min/feature)")
    
    # Add rate limiting to avoid OpenAI 429 errors (500 RPM limit)
    # With 500 RPM limit, we can make ~8 requests per second
    # Add a small delay to stay under limit (0.15s = ~6-7 req/s with safety margin)
    import functools
    rate_limit_delay = 0.15  # seconds between requests
    last_request_time = [0.0]  # Use list to allow modification in nested function
    
    autointerp = AutoInterp(
        cfg=config,
        model=model,
        sae=sae,
        tokenized_dataset=tokens,
        sparsity=sparsity,
        device=str(sae.device),
        api_key=api_key,
    )
    
    # Wrap get_api_response with rate limiting
    original_get_api_response = autointerp.get_api_response
    
    @functools.wraps(original_get_api_response)
    def rate_limited_get_api_response(*args, **kwargs):
        """Add delay between API requests to avoid rate limits."""
        import time
        elapsed = time.time() - last_request_time[0]
        if elapsed < rate_limit_delay:
            time.sleep(rate_limit_delay - elapsed)
        last_request_time[0] = time.time()
        return original_get_api_response(*args, **kwargs)
    
    autointerp.get_api_response = rate_limited_get_api_response
    
    eval_start = time.time()
    results = asyncio.run(autointerp.run())
    eval_time = time.time() - eval_start
    print(f"✓ AutoInterp completed in {eval_time/60:.1f} minutes ({eval_time/len(results):.1f}s per feature)")
    
    # Compute summary statistics
    scores = [v["score"] for v in results.values() if "score" in v]
    
    summary = {
        "name": name,
        "n_evaluated": len(results),
        "n_scored": len(scores),
    }
    
    if scores:
        summary["mean_score"] = float(statistics.mean(scores))
        summary["stdev_score"] = float(statistics.stdev(scores)) if len(scores) > 1 else 0.0
        summary["min_score"] = float(min(scores))
        summary["max_score"] = float(max(scores))
        # Pass rate (>= 0.5 is typically considered good)
        summary["pass_rate"] = sum(1 for s in scores if s >= 0.5) / len(scores)
    
    return {"summary": summary, "details": results}


def main():
    parser = argparse.ArgumentParser(description="Simple AutoInterp comparison")
    parser.add_argument("--matryoshka", type=str, required=True, 
                       help="Path to Matryoshka checkpoint (.pt)")
    parser.add_argument("--google", type=str, required=False, default=None,
                       help="Optional path to Google transcoder (.npz); if omitted, only Matryoshka is evaluated")
    parser.add_argument("--layer", type=int, required=True,
                       help="Layer to evaluate")
    parser.add_argument("--n-latents", type=int, default=50,
                       help="Number of features to evaluate (default: 50)")
    parser.add_argument("--total-tokens", type=int, default=2_000_000,
                       help="Total tokens for evaluation (default: 2M)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for results (default: results/saebench/autointerp/[layer])")
    
    args = parser.parse_args()
    
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable must be set")
    
    # Load model
    print("="*60)
    print("SETUP PHASE")
    print("="*60)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available! Cannot run on GPU.")
    
    device = "cuda:0"
    print(f"Loading Gemma-2-2B model on {device}...")
    
    model = HookedTransformer.from_pretrained_no_processing(
        "gemma-2-2b",
        dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    print(f"✓ Gemma model loaded on {device}")
    
    # Load transcoders  
    print("Loading Matryoshka transcoder...")
    matryoshka = load_matryoshka(Path(args.matryoshka), args.layer)
    matryoshka = matryoshka.to(device=device, dtype=torch.bfloat16)
    matryoshka.eval()  # CRITICAL: Set to eval mode for threshold-based sparsification
    print(f"✓ Matryoshka transcoder loaded on {device}")
    
    # Verify threshold was loaded from checkpoint
    threshold_val = matryoshka.threshold.item() if hasattr(matryoshka, 'threshold') else 0.0
    print(f"  Threshold value: {threshold_val:.6f} ({'✓ loaded from checkpoint' if threshold_val > 0.0 else '⚠ WARNING: threshold is 0.0 - all ReLU activations will pass through'})")
    
    google = None
    if args.google:
        print("Loading Google transcoder...")
        google = load_google_transcoder(Path(args.google), args.layer)
        google = google.to(device=device, dtype=torch.bfloat16)
        google.eval()  # Set to eval mode
        print(f"✓ Google transcoder loaded on {device}")
    
    # Evaluate both
    matryoshka_results = evaluate_transcoder(
        "Matryoshka",
        matryoshka,
        model,
        args.n_latents,
        args.total_tokens,
        api_key=api_key,
    )
    
    google_results = None
    if google is not None:
        google_results = evaluate_transcoder(
            "Google",
            google,
            model,
            args.n_latents,
            args.total_tokens,
            api_key=api_key,
        )
    
    # Save results to results/saebench/autointerp/[layer_number] format
    if args.output_dir is None:
        output_dir = Path("results") / "saebench" / "autointerp" / str(args.layer)
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "matryoshka_results.json", "w") as f:
        json.dump(matryoshka_results, f, indent=2)
    
    if google_results is not None:
        with open(output_dir / "google_results.json", "w") as f:
            json.dump(google_results, f, indent=2)
    
    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    m_sum = matryoshka_results["summary"]
    g_sum = google_results["summary"] if google_results is not None else None
    
    if g_sum is not None:
        print(f"\n{'Metric':<20} {'Matryoshka':>15} {'Google':>15} {'Difference':>15}")
        print("-" * 70)
        if "mean_score" in m_sum and "mean_score" in g_sum:
            print(f"{'Mean Score':<20} {m_sum['mean_score']:>15.4f} {g_sum['mean_score']:>15.4f} "
                  f"{m_sum['mean_score'] - g_sum['mean_score']:>+15.4f}")
            print(f"{'Std Dev':<20} {m_sum['stdev_score']:>15.4f} {g_sum['stdev_score']:>15.4f} "
                  f"{m_sum['stdev_score'] - g_sum['stdev_score']:>+15.4f}")
            print(f"{'Pass Rate':<20} {m_sum['pass_rate']:>15.2%} {g_sum['pass_rate']:>15.2%} "
                  f"{m_sum['pass_rate'] - g_sum['pass_rate']:>+15.2%}")
        print(f"{'Features Scored':<20} {m_sum['n_scored']:>15} {g_sum['n_scored']:>15}")
    else:
        print("Evaluated only Matryoshka. Google transcoder not provided.")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
