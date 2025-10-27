import os
import json
import argparse
import csv


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Compare SAE Bench-style metrics (L17)")
    parser.add_argument("--ours", type=str, required=True)
    parser.add_argument("--google", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=os.path.join("analysis_results", "sae_bench", "gemma-2-2b", "l17"))
    args = parser.parse_args()

    ours = load_json(args.ours)
    google = load_json(args.google)

    out = {
        "ours": ours,
        "google": google,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "comparison.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Compact CSV
    with open(os.path.join(args.out_dir, "comparison.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "mse", "mae", "cos", "absorption_score"])
        def row(name, obj):
            rec = obj.get("reconstruction", {})
            return [name, rec.get("mse"), rec.get("mae"), rec.get("cos"), obj.get("absorption_score")]
        writer.writerow(row("ours", ours))
        writer.writerow(row("google", google))

    print(f"Saved comparison to {args.out_dir}/comparison.json and comparison.csv")


if __name__ == "__main__":
    main()


