"""Compare AutoInterp evaluation runs and print side-by-side summaries."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tabulate import tabulate


@dataclass(slots=True)
class AutoInterpSummary:
    label: str
    path: Path
    mean_score: float | None
    stdev_score: float | None
    coverage: float | None
    pass_rate: float | None
    n_features: int
    n_scored: int

    @classmethod
    def from_metrics(
        cls,
        label: str,
        path: Path,
        *,
        pass_threshold: float,
    ) -> "AutoInterpSummary":
        metrics_path = path / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Expected metrics.json at {metrics_path}")

        payload = json.loads(metrics_path.read_text())

        pass_rate = None
        details_path = path / "details.json"
        if details_path.is_file():
            details_payload = json.loads(details_path.read_text())
            latents = details_payload.get("latents", {})
            scores = [v.get("score") for v in latents.values() if isinstance(v, dict)]
            scored = [score for score in scores if isinstance(score, (int, float))]
            if scored:
                passes = [score for score in scored if score >= pass_threshold]
                pass_rate = len(passes) / len(scored)

        return cls(
            label=label,
            path=path,
            mean_score=payload.get("mean_score"),
            stdev_score=payload.get("stdev_score"),
            coverage=payload.get("coverage"),
            pass_rate=pass_rate,
            n_features=payload.get("n_features_with_results", 0),
            n_scored=payload.get("n_features_with_scores", 0),
        )

    def as_row(self) -> list[Any]:
        return [
            self.label,
            self.mean_score,
            self.stdev_score,
            self.coverage,
            self.pass_rate,
            self.n_features,
            self.n_scored,
        ]


def _delta_rows(summaries: Iterable[AutoInterpSummary]) -> list[list[Any]]:
    summaries = list(summaries)
    if len(summaries) < 2:
        return []

    baseline, *others = summaries
    rows = []
    for item in others:
        rows.append(
            [
                f"{item.label} - {baseline.label}",
                _delta(item.mean_score, baseline.mean_score),
                _delta(item.stdev_score, baseline.stdev_score),
                _delta(item.coverage, baseline.coverage),
                _delta(item.pass_rate, baseline.pass_rate),
                item.n_features - baseline.n_features,
                item.n_scored - baseline.n_scored,
            ]
        )
    return rows


def _delta(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None:
        return None
    return round(current - baseline, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two or more AutoInterp result folders", prog="compare_autointerp"
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help="Paths to AutoInterp run directories (expect metrics.json/config.json)",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        help="Optional labels for the runs (defaults to folder names)",
    )
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=0.5,
        help="Score threshold used to compute pass rate (default: 0.5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.labels and len(args.labels) != len(args.runs):
        raise ValueError("Number of labels must match number of runs")

    summaries: list[AutoInterpSummary] = []
    for idx, run_path_str in enumerate(args.runs):
        path = Path(run_path_str).expanduser().resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"Run path is not a directory: {path}")

        label = args.labels[idx] if args.labels else path.name
        summaries.append(
            AutoInterpSummary.from_metrics(label, path, pass_threshold=args.pass_threshold)
        )

    headers = [
        "run",
        "mean_score",
        "stdev",
        "coverage",
        f"pass_rate(>{args.pass_threshold})",
        "n_features",
        "n_scored",
    ]
    summary_table = tabulate([s.as_row() for s in summaries], headers=headers, tablefmt="github")
    print("Summary\n=======")
    print(summary_table)

    delta_rows = _delta_rows(summaries)
    if delta_rows:
        delta_table = tabulate(delta_rows, headers=headers, tablefmt="github")
        print("\nDeltas (vs first run)\n=====================")
        print(delta_table)


if __name__ == "__main__":
    main()


