"""Simplified AutoInterp evaluation for local execution with local LLM support.

Usage (with local Mistral):
    python eval/sae_bench/run_autointerp_simple.py \
        --matryoshka results/gemma-2-2b/layer8/checkpoints/final.pt \
        --layer 8 \
        --n-latents 50 \
        --use-local-llm \
        --local-model mistralai/Mistral-7B-Instruct-v0.2

Usage (with OpenAI):
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
from typing import Any, Iterable

import torch
from sae_bench.evals.autointerp.eval_config import AutoInterpEvalConfig
from sae_bench.evals.autointerp.main import AutoInterp
from transformer_lens import HookedTransformer

from eval.sae_bench.wrappers import build_matryoshka_adapter, build_npz_transcoder_adapter
from eval.sae_bench.config import NPZTranscoderSpec
from src.utils.config import get_default_cfg


class LocalLLMWrapper:
    """Wrapper for local LLM inference using transformers (single-GPU)."""
    
    def __init__(self, model_name: str, device_ids: list[int] = [0, 1]):
        print(f"Loading local LLM: {model_name}")
        print(f"Target GPU devices: {device_ids}")
        self.model_name = model_name
        
        # Use transformers directly (simple single-GPU for Mistral-7B)
        print("Loading with transformers on single GPU...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set pad_token to avoid CUDA asserts
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # For generation
        
        self.device = f"cuda:{device_ids[0]}"  # Use first GPU
        
        print(f"Loading {model_name} on {self.device}...")
        
        # Simple: load entire model on one GPU (Mistral-7B fits easily on 24GB)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(self.device)
        
        self.backend = "transformers"
        self.device_ids = device_ids
        
        # Verify GPU placement
        total_params = sum(p.numel() for p in self.model.parameters())
        on_device = all(str(p.device) == self.device for p in self.model.parameters())
        
        print(f"✓ Loaded {model_name}: {total_params/1e9:.1f}B params on {self.device}")
        if not on_device:
            print(f"  ⚠ WARNING: Some parameters not on {self.device}!")
        
        # Force GPU memory allocation with proper test
        print("Testing GPU inference...")
        test_prompt = "Hello"
        dummy_input = self.tokenizer(
            test_prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        with torch.no_grad():
            _ = self.model.generate(**dummy_input, max_new_tokens=5)
        print("✓ GPU memory allocated and working")
    
    def generate(self, messages: list[dict[str, str]], n_completions: int = 1) -> list[str]:
        """Generate completions from messages."""
        prompt = self._format_prompt_mixtral(messages)
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)
        
        responses = []
        for _ in range(n_completions):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0 if n_completions == 1 else 0.7,
                do_sample=n_completions > 1,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the assistant's response
            response = response[len(prompt):].strip()
            responses.append(response)
        
        return responses
    
    def _format_prompt_mixtral(self, messages: list[dict[str, str]]) -> str:
        """Format messages for Mixtral instruction format."""
        formatted = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                formatted.append(f"<s>[INST] {content}")
            elif role == "user":
                if formatted:
                    formatted.append(f"{content} [/INST]")
                else:
                    formatted.append(f"<s>[INST] {content} [/INST]")
            elif role == "assistant":
                formatted.append(f"{content}</s>")
        
        # If the last message is from user, we're ready for assistant response
        if messages[-1]["role"] == "user":
            return " ".join(formatted) + " "
        
        return " ".join(formatted)


def make_local_llm_api_response(local_llm: LocalLLMWrapper):
    """Create a patched get_api_response method for AutoInterp using local LLM."""
    
    def get_api_response(
        self: AutoInterp,
        messages: Iterable[dict[str, str]],
        max_tokens: int,
        n_completions: int = 1,
    ) -> tuple[list[str], str]:
        """Patched method to use local LLM instead of OpenAI."""
        
        message_list = list(messages)
        responses = local_llm.generate(message_list, n_completions)
        
        # Format log for display
        from tabulate import tabulate
        log_table = tabulate(
            [m.values() for m in message_list + [{"role": "assistant", "content": responses[0]}]],
            tablefmt="simple_grid",
            maxcolwidths=[None, 120],
        )
        
        return responses, log_table
    
    return get_api_response


def load_matryoshka(state_dict_path: Path, layer: int) -> Any:
    """Load Matryoshka transcoder from checkpoint."""
    cfg = get_default_cfg()
    cfg["model_name"] = "gemma-2-2b"
    cfg["layer"] = layer
    cfg["source_layer"] = layer
    cfg["target_layer"] = layer
    cfg["source_site"] = "mlp_in"
    cfg["target_site"] = "mlp_out"
    cfg["dict_size"] = 18432  # Adjust based on your model
    cfg["top_k"] = 96
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["dtype"] = torch.float16
    
    # CRITICAL: Set correct activation dimensions for Gemma-2-2B!
    cfg["act_size"] = 2304  # Gemma-2-2B activation size (not GPT-2's 768)
    cfg["source_act_size"] = 2304
    cfg["target_act_size"] = 2304
    
    return build_matryoshka_adapter(cfg, state_dict_path=state_dict_path)


def load_google_transcoder(npz_path: Path, layer: int) -> Any:
    """Load Google's transcoder from NPZ file."""
    spec = NPZTranscoderSpec(
        path=str(npz_path),
        model_name="gemma-2-2b",
        source_layer=layer,
        source_hook_point=f"blocks.{layer}.hook_mlp_in",
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
    from sae_bench.sae_bench_utils.activation_collection import get_feature_activation_sparsity
    
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
    
    # Compute sparsity (use smaller batch size to avoid OOM)
    sparsity = get_feature_activation_sparsity(
        tokens=tokens,
        model=model,
        sae=sae,
        batch_size=8,  # Reduced from 32 to avoid OOM with large models
        layer=sae.cfg.hook_layer,
        hook_name=sae.cfg.hook_name,
        mask_bos_pad_eos_tokens=True,
    )
    
    return tokens, sparsity


def evaluate_transcoder(
    name: str,
    sae: Any,
    model: HookedTransformer,
    n_latents: int,
    total_tokens: int,
    api_key: str | None = None,
    local_llm: LocalLLMWrapper | None = None,
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
    print(f"✓ Data prepared in {prep_time:.1f}s (tokens shape: {tokens.shape})")
    
    # Run AutoInterp
    print(f"[2/3] Running AutoInterp on {n_latents} features...")
    if local_llm is not None:
        print(f"      Using local LLM: {local_llm.model_name} ({local_llm.backend})")
    print(f"      Estimated time: ~{n_latents * 2} minutes (2 min/feature)")
    
    autointerp = AutoInterp(
        cfg=config,
        model=model,
        sae=sae,
        tokenized_dataset=tokens,
        sparsity=sparsity,
        device=str(sae.device),
        api_key=api_key or "dummy",  # Dummy key if using local LLM
    )
    
    # Patch with local LLM if provided
    if local_llm is not None:
        patched_method = make_local_llm_api_response(local_llm)
        autointerp.get_api_response = patched_method.__get__(autointerp, AutoInterp)
    
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
    parser.add_argument("--output-dir", type=str, default="autointerp_results",
                       help="Output directory for results")
    
    # Local LLM options
    parser.add_argument("--use-local-llm", action="store_true",
                       help="Use local LLM instead of OpenAI API")
    parser.add_argument("--local-model", type=str, 
                       default="mistralai/Mistral-7B-Instruct-v0.2",
                       help="HuggingFace model name for local LLM")
    parser.add_argument("--local-gpus", type=int, nargs="+", default=[0, 1],
                       help="GPU IDs to use for local LLM (default: 0 1)")
    
    args = parser.parse_args()
    
    # Initialize LLM (local or API)
    local_llm = None
    api_key = None
    
    if args.use_local_llm:
        print(f"Using local LLM: {args.local_model}")
        print(f"GPU devices: {args.local_gpus}")
        local_llm = LocalLLMWrapper(args.local_model, device_ids=args.local_gpus)
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable must be set (or use --use-local-llm)")
    
    # Load model
    print("="*60)
    print("SETUP PHASE")
    print("="*60)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available! Cannot run on GPU.")
    
    # Use GPU 1 for Gemma to avoid OOM (Mistral is on GPU 0)
    device = "cuda:1" if len(args.local_gpus) > 1 and args.use_local_llm else "cuda:0"
    print(f"Loading Gemma-2-2B model on {device}...")
    print(f"  (Mistral LLM is on cuda:{args.local_gpus[0] if args.use_local_llm else 'N/A'})")
    
    model = HookedTransformer.from_pretrained_no_processing(
        "gemma-2-2b",
        dtype=torch.float16,
    ).to(device)
    model.eval()
    print(f"✓ Gemma model loaded on {device}")
    
    # Load transcoders  
    print("Loading Matryoshka transcoder...")
    matryoshka = load_matryoshka(Path(args.matryoshka), args.layer)
    matryoshka = matryoshka.to(device=device, dtype=torch.float16)
    print(f"✓ Matryoshka transcoder loaded on {device}")
    
    google = None
    if args.google:
        print("Loading Google transcoder...")
        google = load_google_transcoder(Path(args.google), args.layer)
        google = google.to(device=device, dtype=torch.float16)
        print(f"✓ Google transcoder loaded on {device}")
    
    # Evaluate both
    matryoshka_results = evaluate_transcoder(
        "Matryoshka",
        matryoshka,
        model,
        args.n_latents,
        args.total_tokens,
        api_key=api_key,
        local_llm=local_llm,
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
            local_llm=local_llm,
        )
    
    # Save results
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
