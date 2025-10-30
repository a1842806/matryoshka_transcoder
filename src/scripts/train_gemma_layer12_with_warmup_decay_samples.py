"""Training entry point for Gemma-2-2B layer 12 Matryoshka transcoders."""

import os
import sys

import torch
from transformer_lens import HookedTransformer

# Allow running the script directly without installing the package.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.sae import MatryoshkaTranscoder
from src.models.transcoder_activation_store import (
    TranscoderActivationsStore,
    create_transcoder_config,
)
from src.training.training import train_transcoder
from src.utils.config import get_default_cfg, post_init_cfg


def build_config() -> dict:
    cfg = get_default_cfg()
    cfg.update(
        {
            "model_name": "gemma-2-2b",
            "dataset_path": "HuggingFaceFW/fineweb-edu",
            "layer": 12,
            "num_tokens": int(2e7),
            "model_batch_size": 4,
            "batch_size": 1024,
            "seq_len": 64,
            "lr": 3e-4,
            "model_dtype": torch.bfloat16,
            "dtype": torch.bfloat16,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "scheduler_type": "warmup_decay",
            "warmup_steps": 1000,
            "sae_type": "matryoshka-transcoder",
            "dict_size": 36864,
            "prefix_sizes": [4608, 9216, 18432, 27648, 36864],
            "top_k": 96,
            "l1_coeff": 0.0,
            "aux_penalty": 1 / 32,
            "n_batches_to_dead": 20,
            "top_k_aux": 512,
            "save_activation_samples": True,
            "sample_collection_freq": 500,
            "max_samples_per_feature": 200,
            "sample_context_size": 30,
            "sample_activation_threshold": 0.2,
            "top_features_to_save": 200,
            "samples_per_feature_to_save": 20,
            "checkpoint_freq": 2000,
            "perf_log_freq": 500,
            "wandb_project": "gemma-2-2b-layer12-interpretability",
        }
    )

    cfg["min_lr"] = cfg["lr"] * 0.01

    cfg = create_transcoder_config(
        cfg,
        source_layer=12,
        target_layer=12,
        source_site="resid_mid",
        target_site="mlp_out",
    )

    return post_init_cfg(cfg)


def main() -> None:
    cfg = build_config()
    expected_steps = int(cfg["num_tokens"] // cfg["batch_size"])
    run_dir_hint = f"results/{cfg['model_name']}/layer{cfg['layer']}/{expected_steps}"

    print(
        f"Training Matryoshka transcoder: model={cfg['model_name']} layer={cfg['layer']} "
        f"steps≈{expected_steps}"
    )
    print(f"Artifacts will be stored under {run_dir_hint}")

    model = HookedTransformer.from_pretrained_no_processing(cfg["model_name"], dtype=cfg["model_dtype"])
    model = model.to(cfg["device"])

    activation_store = TranscoderActivationsStore(model, cfg)
    transcoder = MatryoshkaTranscoder(cfg).to(cfg["device"])

    try:
        train_transcoder(transcoder, activation_store, model, cfg)
    except Exception as exc:  # pragma: no cover - used for CLI feedback
        print(f"Training failed: {exc}")
        raise
    else:
        print(f"Training complete. Inspect {run_dir_hint} for checkpoints, metrics, and samples.")


if __name__ == "__main__":
    main()

