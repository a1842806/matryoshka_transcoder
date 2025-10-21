import os
import json
import ast
import argparse
from typing import Dict, Any

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from src.utils.config import get_default_cfg
from src.models.sae import MatryoshkaTranscoder
from src.models.transcoder_activation_store import TranscoderActivationsStore, create_transcoder_config
from src.adapters.gemma_scope_transcoder import GemmaScopeTranscoderAdapter
from src.eval.sae_bench_adapters import OurTranscoderAdapter, GoogleTranscoderAdapterWrapper


@torch.no_grad()
def compute_absorption_from_features(feature_vecs: torch.Tensor, subset: int = 8192) -> float:
    # feature_vecs: (n_features, d) L2-normalized
    if feature_vecs is None or feature_vecs.numel() == 0:
        return float('nan')
    V = feature_vecs
    if subset and V.shape[0] > subset:
        idx = torch.randperm(V.shape[0])[:subset]
        V = V[idx]
    S = (V @ V.T).abs()
    # remove diagonal
    S = S - torch.eye(S.shape[0], device=S.device)
    # fraction of pairs with cosine > 0.9
    m = S.numel() - S.shape[0]
    return (S > 0.9).sum().item() / max(1, m)


def save_results(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="SAE Bench-style eval for Gemma 2B L17")
    parser.add_argument("--which", type=str, choices=["ours", "google"], required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, help="Our checkpoint dir (for --which ours)")
    parser.add_argument("--google_dir", type=str, help="Local snapshot dir (for --which google)")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--batches", type=int, default=500)
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    device = torch.device(args.device)
    dtype = dtype_map[args.dtype]

    # Load model
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=device, dtype=dtype)

    # Config for mlp_in -> mlp_out at layer 17
    cfg = get_default_cfg()
    cfg["model_name"] = "gemma-2-2b"
    cfg["device"] = device
    cfg["dtype"] = dtype
    cfg["dataset_path"] = args.dataset
    cfg["batch_size"] = 1024
    cfg["seq_len"] = 128
    cfg["model_batch_size"] = 256
    cfg = create_transcoder_config(cfg, source_layer=17, target_layer=17, source_site="mlp_in", target_site="mlp_out")

    store = TranscoderActivationsStore(model, cfg)

    # Build adapter
    if args.which == "ours":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for --which ours")
        # Load our transcoder
        with open(os.path.join(args.checkpoint, "config.json"), "r") as f:
            ck_cfg = json.load(f)
        # Coerce types from JSON serialization
        def _maybe_int(x):
            try:
                return int(x)
            except Exception:
                return x
        def _maybe_float(x):
            try:
                return float(x)
            except Exception:
                return x
        # dtypes
        if isinstance(ck_cfg.get("dtype"), str):
            try:
                ck_cfg["dtype"] = getattr(torch, ck_cfg["dtype"].split(".")[-1])
            except Exception:
                ck_cfg["dtype"] = dtype
        if isinstance(ck_cfg.get("model_dtype"), str):
            try:
                ck_cfg["model_dtype"] = getattr(torch, ck_cfg["model_dtype"].split(".")[-1])
            except Exception:
                ck_cfg["model_dtype"] = dtype
        # numeric coercions
        for k in [
            "dict_size", "top_k", "top_k_aux", "n_batches_to_dead",
            "source_act_size", "target_act_size", "act_size"
        ]:
            if k in ck_cfg:
                ck_cfg[k] = _maybe_int(ck_cfg[k])
        for k in ["lr", "aux_penalty", "min_lr"]:
            if k in ck_cfg:
                v = ck_cfg[k]
                if isinstance(v, str) and "/" in v:
                    # handle simple fractions like "1/32"
                    try:
                        num, den = v.split("/")
                        v = float(num) / float(den)
                    except Exception:
                        pass
                ck_cfg[k] = _maybe_float(v)
        gs = ck_cfg.get("group_sizes")
        if isinstance(gs, str):
            try:
                ck_cfg["group_sizes"] = json.loads(gs)
            except Exception:
                try:
                    ck_cfg["group_sizes"] = ast.literal_eval(gs)
                except Exception:
                    pass
        ck_cfg["device"] = device
        transcoder = MatryoshkaTranscoder(ck_cfg).to(device=device, dtype=dtype)
        transcoder.load_state_dict(torch.load(os.path.join(args.checkpoint, "sae.pt"), map_location=device))
        adapter = OurTranscoderAdapter(transcoder)
        out_dir = os.path.join("analysis_results", "sae_bench", "gemma-2-2b", "l17", "ours")
    else:
        if not args.google_dir:
            raise ValueError("--google_dir is required for --which google")
        scope = GemmaScopeTranscoderAdapter(layer=17, repo_dir=args.google_dir, device=device, dtype=dtype)
        adapter = GoogleTranscoderAdapterWrapper(scope)
        out_dir = os.path.join("analysis_results", "sae_bench", "gemma-2-2b", "l17", "google")

    # Reconstruction metrics over batches
    running = {"mse": 0.0, "mae": 0.0, "cos": 0.0, "count": 0}
    for _ in range(args.batches):
        src, tgt = store.next_batch()
        src = src.to(device=device, dtype=dtype)
        tgt = tgt.to(device=device, dtype=dtype)
        pred = adapter(src)
        mse = F.mse_loss(pred, tgt).item()
        mae = (pred - tgt).abs().mean().item()
        cos = F.cosine_similarity(pred.reshape(-1, pred.shape[-1]), tgt.reshape(-1, tgt.shape[-1]), dim=-1).mean().item()
        n = src.shape[0]
        running["mse"] += mse * n
        running["mae"] += mae * n
        running["cos"] += cos * n
        running["count"] += n

    metrics = {k: (running[k] / max(1, running["count"])) for k in ["mse", "mae", "cos"]}

    # Absorption proxy (only for ours where features exist)
    absorption = None
    if hasattr(adapter, "get_feature_vectors"):
        feat_vecs = adapter.get_feature_vectors()
        if feat_vecs is not None:
            absorption = compute_absorption_from_features(feat_vecs)

    results = {
        "reconstruction": metrics,
        "absorption_score": absorption,
    }

    save_results(os.path.join(out_dir, "metrics.json"), results)
    print(f"Saved SAE Bench-style metrics to {out_dir}/metrics.json")


if __name__ == "__main__":
    main()


