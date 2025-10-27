#!/usr/bin/env python3
import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn.functional as F

from transformer_lens import HookedTransformer

from utils.config import get_default_cfg, compute_fvu, get_hook_name
from models.sae import MatryoshkaTranscoder
from models.transcoder_activation_store import TranscoderActivationsStore, create_transcoder_config
from adapters.gemma_scope_transcoder import GemmaScopeTranscoderAdapter, time_forward

def make_output_dir(base_dir: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, f"compare_layer17_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def load_matryoshka(checkpoint_dir: str, device: torch.device, dtype: torch.dtype) -> Tuple[MatryoshkaTranscoder, Dict[str, Any]]:
    ckpt_dir = Path(checkpoint_dir)
    cfg_path = ckpt_dir / "config.json"
    state_path = ckpt_dir / "sae.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing sae.pt under {checkpoint_dir}")

    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
    else:
        cfg = get_default_cfg()

    cfg.setdefault("model_name", "gemma-2-2b")
    cfg.setdefault("device", str(device))
    cfg["device"] = device
    cfg.setdefault("dtype", dtype)
    cfg["dtype"] = dtype

    layer = int(cfg.get("layer", 17))
    src_site = cfg.get("source_site", "resid_mid")
    tgt_site = cfg.get("target_site", "mlp_out")
    cfg["source_hook_point"] = cfg.get("source_hook_point", get_hook_name(src_site, layer, cfg["model_name"]))
    cfg["target_hook_point"] = cfg.get("target_hook_point", get_hook_name(tgt_site, layer, cfg["model_name"]))

    model = MatryoshkaTranscoder(cfg).to(device=device, dtype=dtype)
    model.load_state_dict(torch.load(str(state_path), map_location=device))
    model.eval()
    return model, cfg

@torch.no_grad()
def batch_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
  
    Y = y_true.reshape(-1, y_true.shape[-1]).float()
    Yh = y_pred.reshape(-1, y_pred.shape[-1]).float()

    mse = F.mse_loss(Yh, Y).item()
    mae = (Yh - Y).abs().mean().item()

    var_res = (Yh - Y).pow(2).mean().item()
    var_orig = Y.var().item() + 1e-8
    fvu = var_res / var_orig
    r2 = 1.0 - fvu

    Yn = F.normalize(Y, p=2, dim=-1)
    Yhn = F.normalize(Yh, p=2, dim=-1)
    cos = (Yn * Yhn).sum(dim=-1).mean().item()

    return {"mse": mse, "mae": mae, "fvu": fvu, "r2": r2, "cos": cos}

def aggregate_metrics(running: Dict[str, float], new: Dict[str, float], n: int) -> Dict[str, float]:
  
    for k, v in new.items():
        if k not in running:
            running[k] = 0.0
            running[f"count_{k}"] = 0
        running[k] += v * n
        running[f"count_{k}"] += n
    return running

def finalize_metrics(running: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for k, v in running.items():
        if k.startswith("count_"):
            continue
        cnt = running.get(f"count_{k}", 1)
        out[k] = v / max(1, cnt)
    return out

def evaluate_models(
    model: HookedTransformer,
    store: TranscoderActivationsStore,
    matryoshka: MatryoshkaTranscoder,
    scope_adapter: GemmaScopeTranscoderAdapter,
    total_batches: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    results = {
        "matryoshka": {"metrics": {}, "sparsity": {}},
        "gemma_scope": {"metrics": {}, "sparsity": {}},
    }

    run_m = {}
    run_g = {}

    l0_sum = 0.0
    l0_count = 0

    for _ in range(total_batches):
        src, tgt = store.next_batch()  # (B, D)
        src = src.to(device=device, dtype=dtype)
        tgt = tgt.to(device=device, dtype=dtype)

        acts = matryoshka.encode(src)
        pred_m = matryoshka.decode(acts)
        bm = batch_metrics(tgt, pred_m)
        run_m = aggregate_metrics(run_m, bm, n=src.shape[0])

        l0 = (acts > 0).sum(dim=-1).float().mean().item()
        l0_sum += l0 * src.shape[0]
        l0_count += src.shape[0]

        pred_g = scope_adapter(src)
        bg = batch_metrics(tgt, pred_g)
        run_g = aggregate_metrics(run_g, bg, n=src.shape[0])

    results["matryoshka"]["metrics"] = finalize_metrics(run_m)
    results["matryoshka"]["sparsity"] = {"avg_l0": (l0_sum / max(1, l0_count))}
    results["gemma_scope"]["metrics"] = finalize_metrics(run_g)
    results["gemma_scope"]["sparsity"] = {"avg_l0": None}

    with torch.no_grad():
        src, _ = store.next_batch()
        src = src.to(device=device, dtype=dtype)
      
        def fn_m(x):
            a = matryoshka.encode(x)
            return matryoshka.decode(a)
        tm = time_forward(type("F", (), {"__call__": lambda self, x: fn_m(x)})(), src, repeat=5)
      
        tg = time_forward(scope_adapter, src, repeat=5)
    results["matryoshka"]["efficiency"] = {"sec_per_forward": tm}
    results["gemma_scope"]["efficiency"] = {"sec_per_forward": tg}

    return results

@torch.no_grad()
def compute_behavioral_perplexity(
    model: HookedTransformer,
    store: TranscoderActivationsStore,
    matryoshka: MatryoshkaTranscoder,
    scope_adapter: GemmaScopeTranscoderAdapter,
    num_batches: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    """
    Compute baseline perplexity and perplexity when replacing layer-17 mlp_out
    with each transcoder's reconstruction. Returns per-model dict with ppl and delta.
    """
    layer = 17
    src_hook = f"blocks.{layer}.hook_resid_mid"
    tgt_hook = f"blocks.{layer}.hook_mlp_out"

    def ce_loss(logits: torch.Tensor, tokens: torch.Tensor) -> float:
      
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = tokens[:, 1:].contiguous()
        return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).item()

    agg = {
        "baseline": {"loss_sum": 0.0, "count": 0},
        "matryoshka": {"loss_sum": 0.0, "count": 0},
        "gemma_scope": {"loss_sum": 0.0, "count": 0},
    }

    for _ in range(num_batches):
        tokens = store.get_batch_tokens()  # (B, T)
        tokens = tokens.to(device=device)

        logits = model(tokens)
        bl = ce_loss(logits, tokens)
        agg["baseline"]["loss_sum"] += bl * tokens.size(0)
        agg["baseline"]["count"] += tokens.size(0)

        resid_mid_cache = {"val": None}

        def hook_resid_mid(act, hook):
            resid_mid_cache["val"] = act
            return act

        def hook_mlp_out_mat(act, hook):
            src = resid_mid_cache["val"]  # (B, T, D)
            a = matryoshka.encode(src)
            y = matryoshka.decode(a)
            return y

        logits_m = model.run_with_hooks(
            tokens,
            fwd_hooks=[(src_hook, hook_resid_mid), (tgt_hook, hook_mlp_out_mat)],
        )
        lm = ce_loss(logits_m, tokens)
        agg["matryoshka"]["loss_sum"] += lm * tokens.size(0)
        agg["matryoshka"]["count"] += tokens.size(0)

        resid_mid_cache = {"val": None}

        def hook_resid_mid2(act, hook):
            resid_mid_cache["val"] = act
            return act

        def hook_mlp_out_scope(act, hook):
            src = resid_mid_cache["val"]  # (B, T, D)
            B, T, D = src.shape
            flat = src.reshape(B * T, D)
            y = scope_adapter(flat).reshape(B, T, -1)
            return y

        logits_g = model.run_with_hooks(
            tokens,
            fwd_hooks=[(src_hook, hook_resid_mid2), (tgt_hook, hook_mlp_out_scope)],
        )
        lg = ce_loss(logits_g, tokens)
        agg["gemma_scope"]["loss_sum"] += lg * tokens.size(0)
        agg["gemma_scope"]["count"] += tokens.size(0)

    out = {}
    base_loss = agg["baseline"]["loss_sum"] / max(1, agg["baseline"]["count"])
    base_ppl = float(torch.exp(torch.tensor(base_loss)))

    for name in ["matryoshka", "gemma_scope"]:
        l = agg[name]["loss_sum"] / max(1, agg[name]["count"])
        ppl = float(torch.exp(torch.tensor(l)))
        out[name] = {"ppl": ppl, "delta_vs_baseline": ppl - base_ppl, "baseline_ppl": base_ppl}

    return out

def save_metrics(out_dir: str, results: Dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    import csv
    csv_path = os.path.join(out_dir, "reconstruction.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "mse", "mae", "fvu", "r2", "cos"])
        for name in ["matryoshka", "gemma_scope"]:
            m = results[name]["metrics"]
            writer.writerow([name, m.get("mse"), m.get("mae"), m.get("fvu"), m.get("r2"), m.get("cos")])

    csv_path = os.path.join(out_dir, "sparsity.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "avg_l0"])
        for name in ["matryoshka", "gemma_scope"]:
            s = results[name].get("sparsity", {})
            writer.writerow([name, s.get("avg_l0")])

    beh = results.get("behavioral")
    if beh is not None:
        csv_path = os.path.join(out_dir, "behavioral.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "ppl", "delta_vs_baseline", "baseline_ppl"])
            for name in ["matryoshka", "gemma_scope"]:
                b = beh.get(name, {})
                writer.writerow([name, b.get("ppl"), b.get("delta_vs_baseline"), b.get("baseline_ppl")])

def main():
    parser = argparse.ArgumentParser(description="Compare Matryoshka vs Gemma Scope transcoders at layer 17 on FineWeb-EDU")
    parser.add_argument("--matryoshka_checkpoint", type=str, required=True, help="Path to your Matryoshka checkpoint dir (contains sae.pt, config.json)")
    parser.add_argument("--gemma_scope_dir", type=str, required=True, help="Local path to Gemma Scope 2B pretrained transcoders snapshot")
    parser.add_argument("--dataset", type=str, required=True, help="FineWeb-EDU dataset path or HF dataset id")
    parser.add_argument("--tokens", type=int, default=int(2e6), help="Approximate number of tokens for evaluation")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--model_batch_size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=2000, help="Override computed batches (advanced)")
    parser.add_argument("--behavioral_batches", type=int, default=128, help="Batches for behavioral ppl replacement eval")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"]) 
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    device = torch.device(args.device)
    dtype = dtype_map[args.dtype]

    model = HookedTransformer.from_pretrained("gemma-2-2b", device=device, dtype=dtype)

    cfg = get_default_cfg()
    cfg["model_name"] = "gemma-2-2b"
    cfg["device"] = device
    cfg["dtype"] = dtype
    cfg["seq_len"] = args.seq_len
    cfg["model_batch_size"] = args.model_batch_size
    cfg["dataset_path"] = args.dataset
    cfg["batch_size"] = 1024
    cfg = create_transcoder_config(
        cfg,
        source_layer=17,
        target_layer=17,
        source_site="mlp_in",
        target_site="mlp_out",
    )

    store = TranscoderActivationsStore(model, cfg)

    matryoshka, mat_cfg = load_matryoshka(args.matryoshka_checkpoint, device, dtype)

    scope_adapter = GemmaScopeTranscoderAdapter(layer=17, repo_dir=args.gemma_scope_dir, device=device, dtype=dtype)

    if args.batches:
        total_batches = args.batches
    else:
        total_batches = max(1, int(args.tokens // (cfg["batch_size"])))

    results = evaluate_models(model, store, matryoshka, scope_adapter, total_batches, device, dtype)

    behavioral = compute_behavioral_perplexity(
        model, store, matryoshka, scope_adapter, args.behavioral_batches, device, dtype
    )
    results["behavioral"] = behavioral

    out_dir = make_output_dir(os.path.join("metrics", "comparisons"))
    save_metrics(out_dir, results)
    print(f"Saved comparison to {out_dir}")

if __name__ == "__main__":
    main()

