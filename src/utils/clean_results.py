"""Utilities for saving training runs in a standardized layout."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _make_json_serializable(obj: Any) -> Any:
    """Convert objects that are not JSON serializable into strings."""
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    return obj


class CleanResultsManager:
    """Helper class for writing training artifacts to disk."""

    def __init__(self, base_dir: str = "results") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_experiment_dir(
        self,
        model_name: str,
        layer: int,
        steps: int,
        description: str = "",
    ) -> Path:
        """Create `results/{model}/{layerX}/{steps}/` and return the path."""

        layer_dir = self.base_dir / model_name / f"layer{layer}"
        run_dir = layer_dir / str(steps)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)

        return run_dir

    def save_checkpoint(
        self,
        experiment_dir: Path,
        model: torch.nn.Module,
        config: Dict[str, Any],
        step: int,
        metrics: Optional[Dict[str, Any]] = None,
        is_final: bool = False,
    ) -> None:
        """Persist model state, config, and metrics to the run directory."""

        checkpoints_dir = experiment_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)

        checkpoint_name = "final.pt" if is_final else f"step_{step}.pt"
        checkpoint_path = checkpoints_dir / checkpoint_name
        torch.save(model.state_dict(), checkpoint_path)

        config_copy = _make_json_serializable(config)
        if is_final:
            config_copy["completed_at"] = datetime.utcnow().isoformat() + "Z"
        else:
            config_copy.pop("completed_at", None)

        config_path = experiment_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config_copy, f, indent=2)

        if metrics:
            metrics_copy = _make_json_serializable(metrics)
            if is_final:
                metrics_path = experiment_dir / "metrics.json"
            else:
                interim_dir = experiment_dir / "metrics"
                interim_dir.mkdir(exist_ok=True)
                metrics_path = interim_dir / f"step_{step}.json"
            with metrics_path.open("w", encoding="utf-8") as f:
                json.dump(metrics_copy, f, indent=2)

    def save_activation_samples(
        self,
        experiment_dir: Path,
        sample_collector: Any,
        top_k_features: int = 100,
        samples_per_feature: int = 10,
    ) -> None:
        samples_dir = experiment_dir / "activation_samples"
        samples_dir.mkdir(exist_ok=True)
        sample_collector.save_samples(str(samples_dir), top_k_features, samples_per_feature)

    def list_experiments(
        self,
        model_name: Optional[str] = None,
        layer: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return a dictionary describing saved runs."""

        experiments: Dict[str, Any] = {}

        for model_dir in self.base_dir.iterdir():
            if not model_dir.is_dir():
                continue
            if model_name and model_dir.name != model_name:
                continue

            model_runs: Dict[int, Any] = {}

            for layer_dir in model_dir.iterdir():
                if not layer_dir.is_dir() or not layer_dir.name.startswith("layer"):
                    continue
                try:
                    layer_num = int(layer_dir.name.replace("layer", ""))
                except ValueError:
                    continue
                if layer is not None and layer_num != layer:
                    continue

                runs = []
                for run_dir in sorted(layer_dir.iterdir()):
                    if not run_dir.is_dir():
                        continue

                    runs.append(
                        {
                            "steps": run_dir.name,
                            "path": str(run_dir),
                            "has_final_checkpoint": (run_dir / "checkpoints" / "final.pt").exists(),
                            "has_metrics": (run_dir / "metrics.json").exists(),
                        }
                    )

                if runs:
                    model_runs[layer_num] = runs

            if model_runs:
                experiments[model_dir.name] = model_runs

        return experiments

