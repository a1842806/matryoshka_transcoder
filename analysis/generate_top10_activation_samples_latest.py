#!/usr/bin/env python3
import os
import json
from pathlib import Path


def find_latest_samples_dir(base_dir: str) -> str:
    candidates = []
    base = Path(base_dir)
    if not base.exists():
        return ""
    for p in base.rglob("*_activation_samples"):
        if (p / "collection_summary.json").exists():
            try:
                mtime = (p / "collection_summary.json").stat().st_mtime
            except Exception:
                mtime = p.stat().st_mtime
            candidates.append((mtime, p))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: x[0], reverse=True)
    return str(candidates[0][1])


def collect_top10_per_feature(samples_dir: str) -> dict:
    out = {}
    for fname in os.listdir(samples_dir):
        if not (fname.startswith("feature_") and fname.endswith("_samples.json")):
            continue
        fpath = os.path.join(samples_dir, fname)
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        stats = data.get("statistics", {})
        feat_idx = stats.get("feature_idx")
        samples = data.get("top_samples", [])
        samples.sort(key=lambda s: s.get("activation_value", 0.0), reverse=True)
        top10 = samples[:10]
        if feat_idx is None:
            # derive from filename if missing
            try:
                feat_idx = int(fname.split("_")[1])
            except Exception:
                continue
        out[str(feat_idx)] = [
            {
                "activation_value": s.get("activation_value"),
                "text": s.get("text"),
                "context_text": s.get("context_text"),
                "position": s.get("position"),
                "tokens": s.get("tokens"),
                "context_tokens": s.get("context_tokens"),
            }
            for s in top10
        ]
    return out


def main():
    project_root = str(Path(__file__).resolve().parents[1])
    base_dir = os.path.join(project_root, "checkpoints", "transcoder")
    latest_dir = find_latest_samples_dir(base_dir)
    if not latest_dir:
        print("No activation_samples directory found.")
        return
    result = collect_top10_per_feature(latest_dir)
    out_path = os.path.join(latest_dir, "top10_per_feature.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved top-10 samples per feature to: {out_path}")
    print(f"Features processed: {len(result)} (source: {latest_dir})")


if __name__ == "__main__":
    main()


