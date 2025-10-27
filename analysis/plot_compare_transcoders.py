#!/usr/bin/env python3
import os
import json
import argparse
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot comparison results")
    parser.add_argument("--results", type=str, required=True, help="Path to results.json produced by compare_transcoders.py")
    args = parser.parse_args()

    with open(args.results, "r") as f:
        data = json.load(f)

    out_dir = os.path.dirname(args.results)

    # Reconstruction metrics bar plot
    fig, ax = plt.subplots(figsize=(6, 4))
    names = ["matryoshka", "gemma_scope"]
    metrics = ["fvu", "mse", "cos"]
    x = range(len(names))

    width = 0.25
    for i, m in enumerate(metrics):
        vals = [data[n]["metrics"].get(m, None) for n in names]
        ax.bar([xi + i * width for xi in x], vals, width=width, label=m)

    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(names)
    ax.set_title("Reconstruction (Layer 17)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reconstruction.png"), dpi=200)
    plt.close(fig)

    # Behavioral ppl
    if "behavioral" in data:
        beh = data["behavioral"]
        fig, ax = plt.subplots(figsize=(6, 4))
        vals = [beh.get(n, {}).get("ppl", None) for n in names]
        ax.bar(names, vals)
        ax.set_title("Perplexity with Layer-17 Replacement")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "behavioral_ppl.png"), dpi=200)
        plt.close(fig)

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()


