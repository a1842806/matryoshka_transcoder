"""Compact training recipe for Gemma-2-2B layer 5 (≈16k steps)."""

import os
import sys

import torch
from transformer_lens import HookedTransformer

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
            "layer": 5,
            "num_tokens": 16_384_000,
            "model_batch_size": 4,
            "batch_size": 1024,
            "seq_len": 128,
            "lr": 4e-4,
            "model_dtype": torch.bfloat16,
            "dtype": torch.bfloat16,
            "device": "cuda:1" if torch.cuda.device_count() > 1 else "cuda" if torch.cuda.is_available() else "cpu",
            "scheduler_type": "warmup_decay",
            "warmup_steps": 1024,
            "dict_size": 18432,
            "prefix_sizes": [2304, 4608, 9216, 13824, 18432],
            "top_k": 96,
            "aux_penalty": 1 / 64,
            "n_batches_to_dead": 20,
            "top_k_aux": 256,
            "save_activation_samples": True,
            "sample_collection_freq": 100,
            "max_samples_per_feature": 100,
            "sample_context_size": 20,
            "sample_activation_threshold": 0.1,
            "top_features_to_save": 100,
            "samples_per_feature_to_save": 10,
            "perf_log_freq": 25,
            "checkpoint_freq": 3000,
            "wandb_project": "gemma-2-2b-layer5-16k-steps",
            "experiment_description": "16k-steps-layer5",
            "loss_recovered_freq": 3000,
            "loss_recovered_n_batches": 5,
        }
    )

    cfg["min_lr"] = cfg["lr"] * 0.01

    cfg["source_act_size"] = 2304
    cfg["target_act_size"] = 2304

    cfg = create_transcoder_config(
        cfg,
        source_layer=5,
        target_layer=5,
        source_site="mlp_in",
        target_site="mlp_out",
    )

    return post_init_cfg(cfg)


def main() -> None:
    cfg = build_config()
    expected_steps = int(cfg["num_tokens"] // cfg["batch_size"])
    run_dir_hint = f"results/{cfg['model_name']}/layer{cfg['layer']}/{expected_steps}"

    print(
        f"Training Matryoshka transcoder: model={cfg['model_name']} "
        f"layer={cfg['layer']} steps≈{expected_steps}"
    )
    print(f"Artifacts will be stored under {run_dir_hint}")

    model = HookedTransformer.from_pretrained_no_processing(cfg["model_name"], dtype=cfg["model_dtype"])
    model = model.to(cfg["device"])

    activation_store = TranscoderActivationsStore(model, cfg)
    transcoder = MatryoshkaTranscoder(cfg).to(cfg["device"])

    try:
        train_transcoder(transcoder, activation_store, model, cfg)
    except Exception as exc:  # pragma: no cover
        print(f"Training failed: {exc}")
        raise
    else:
        print(f"Training complete. Inspect {run_dir_hint} for checkpoints, metrics, and samples.")


if __name__ == "__main__":
    main()

